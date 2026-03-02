"""
train_carrydiff.py
──────────────────────────────────────────────────────────────────────────────
Trains the 4 CarryDiff parameters from data.

The question: if we treat CARRY_DETECT_THRESH, CARRY_OUT_THRESH, TEN, and
SCALE as learnable scalars (initialized arbitrarily), do they converge to
values that correctly separate carries?

Valid convergence means:
  CARRY_DETECT_THRESH  ∈ (0, 9)   — any value in this interval works
  CARRY_OUT_THRESH     ∈ (9, 10)  — any value in this interval works
  TEN                  ≈ 10       — must be close to base
  SCALE                >> 1       — must be large enough for hard thresholding

USAGE
─────
  python train_carrydiff.py --device cuda
  python train_carrydiff.py --device cpu   # fast enough, ~60s

OUTPUT
──────
  Logs parameter values and accuracy every 500 steps.
  Final plot saved to results/carrydiff_threshold_training.png
"""

import argparse
import math
import time
import random
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# ── Learnable CarryDiff ────────────────────────────────────────────────────────

class LearnableCarryDiff(nn.Module):
    """
    CarryDiff with all 4 parameters as nn.Parameter.
    Identical computation to the hand-coded version — only the values are learned.
    """

    def __init__(self, init_mode='random'):
        super().__init__()

        if init_mode == 'correct':
            # Initialise at the known-correct values (sanity check)
            self.carry_detect_thresh = nn.Parameter(torch.tensor(8.5))
            self.carry_out_thresh    = nn.Parameter(torch.tensor(9.5))
            self.ten                 = nn.Parameter(torch.tensor(10.0))
            self.scale               = nn.Parameter(torch.tensor(100.0))  # start lower

        elif init_mode == 'random':
            # Random initialisation — the interesting experiment
            # Deliberately bad starting points to test convergence
            self.carry_detect_thresh = nn.Parameter(torch.tensor(float(random.uniform(1, 7))))
            self.carry_out_thresh    = nn.Parameter(torch.tensor(float(random.uniform(3, 8))))
            self.ten                 = nn.Parameter(torch.tensor(float(random.uniform(5, 15))))
            self.scale               = nn.Parameter(torch.tensor(float(random.uniform(1, 20))))

        elif init_mode == 'adversarial':
            # Start at clearly wrong values
            self.carry_detect_thresh = nn.Parameter(torch.tensor(5.0))   # wrong: inside gap
            self.carry_out_thresh    = nn.Parameter(torch.tensor(5.0))   # wrong: way off
            self.ten                 = nn.Parameter(torch.tensor(8.0))   # wrong base
            self.scale               = nn.Parameter(torch.tensor(1.0))   # way too small

        elif init_mode == 'dead':
            # Wrong threshold + very high scale -> saturated, near-zero gradients.
            self.carry_detect_thresh = nn.Parameter(torch.tensor(10.5))  # wrong: outside valid (0,9)
            self.carry_out_thresh    = nn.Parameter(torch.tensor(9.5))   # nominally correct
            self.ten                 = nn.Parameter(torch.tensor(10.0))   # correct base
            self.scale               = nn.Parameter(torch.tensor(100.0))  # hard saturation

    def _sigmoid(self, x):
        return torch.sigmoid(x)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        a, b: [B, 10] float tensors of digits (0–9)
        returns: [B, 11] float tensor of output digits
        """
        B = a.shape[0]
        z = torch.zeros(B, 1, device=a.device, dtype=a.dtype)

        a11 = torch.cat([z, a], dim=1)   # [B, 11]
        b11 = torch.cat([z, b], dim=1)
        raw = a11 + b11                   # column sums, no carry

        # Initial state: raw mod 10 (fully noised)
        carry0 = self._sigmoid(self.scale * (raw - self.carry_out_thresh))
        x = raw - self.ten * carry0

        # 10 denoising steps
        for _ in range(10):
            # Shift right: each position looks at its right neighbour
            raw_right = torch.cat([raw[:, 1:], z], dim=1)
            x_right   = torch.cat([x[:, 1:],   z], dim=1)

            carry_in  = self._sigmoid(self.scale * (raw_right - x_right - self.carry_detect_thresh))
            col       = raw + carry_in
            carry_out = self._sigmoid(self.scale * (col - self.carry_out_thresh))
            x         = col - self.ten * carry_out

        return x   # [B, 11]


# ── Data generation ────────────────────────────────────────────────────────────

def make_batch(batch_size: int, device: str) -> tuple:
    """Generate random 10-digit pairs and their digit-level answers."""
    a = torch.randint(0, 10_000_000_000, (batch_size,), dtype=torch.int64)
    b = torch.randint(0, 10_000_000_000, (batch_size,), dtype=torch.int64)
    c = a + b

    # Extract digits: MSD first, 10 digits for a and b, 11 for c
    powers10 = torch.tensor([10**i for i in range(9, -1, -1)], dtype=torch.int64)
    powers11 = torch.tensor([10**i for i in range(10, -1, -1)], dtype=torch.int64)

    a_dig = ((a[:, None] // powers10[None, :]) % 10).float()
    b_dig = ((b[:, None] // powers10[None, :]) % 10).float()
    c_dig = ((c[:, None] // powers11[None, :]) % 10).float()

    return a_dig.to(device), b_dig.to(device), c_dig.to(device)


def exact_accuracy(model, n=2000, device='cpu') -> float:
    """Fraction of (a,b) pairs where all 11 output digits are exactly correct."""
    model.eval()
    with torch.no_grad():
        a, b, c = make_batch(n, device)
        pred = model(a, b)
        pred_round = pred.round().long().clamp(0, 9)
        c_int = c.long()
        correct = (pred_round == c_int).all(dim=1).float().mean().item()
    model.train()
    return correct


# ── Training ───────────────────────────────────────────────────────────────────

def train(args):
    device = args.device
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    results_dir = Path('results/carrydiff_thresholds')
    results_dir.mkdir(parents=True, exist_ok=True)

    # Run all three init modes
    init_modes = ['correct', 'random', 'adversarial', 'dead']
    if args.init != 'all':
        init_modes = [args.init]

    all_histories = {}

    for init_mode in init_modes:
        print(f"\n{'='*65}")
        print(f"  Learnable CarryDiff — init: {init_mode}")
        print(f"{'='*65}")

        torch.manual_seed(args.seed)
        random.seed(args.seed)

        model = LearnableCarryDiff(init_mode=init_mode).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # LR schedule: warmup then cosine decay
        def lr_lambda(step):
            if step < 200:
                return step / 200
            progress = (step - 200) / max(1, args.steps - 200)
            return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        history = {
            'step': [], 'loss': [], 'acc': [],
            'cdt': [], 'cot': [], 'ten': [], 'scale': []
        }

        print(f"\n  Initial params:")
        print(f"    carry_detect_thresh = {model.carry_detect_thresh.item():.4f}  (valid: 0 < x < 9)")
        print(f"    carry_out_thresh    = {model.carry_out_thresh.item():.4f}  (valid: 9 < x < 10)")
        print(f"    ten                 = {model.ten.item():.4f}  (valid: ≈ 10)")
        print(f"    scale               = {model.scale.item():.4f}  (valid: >> 1)")
        print()

        header = f"{'step':>7}  {'loss':>8}  {'acc':>7}  {'cdt':>7}  {'cot':>7}  {'ten':>7}  {'scale':>8}  {'valid?':>6}"
        print(f"  {header}")
        print(f"  {'─'*72}")

        t0 = time.time()

        for step in range(args.steps):
            a, b, c = make_batch(args.batch_size, device)
            pred = model(a, b)

            # MSE loss on digit values
            loss = nn.functional.mse_loss(pred, c)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 500 == 0 or step == args.steps - 1:
                acc = exact_accuracy(model, n=1000, device=device)

                cdt   = model.carry_detect_thresh.item()
                cot   = model.carry_out_thresh.item()
                ten   = model.ten.item()
                scale = model.scale.item()

                valid_cdt   = 0 < cdt < 9
                valid_cot   = 9 < cot < 10
                valid_ten   = abs(ten - 10) < 0.5
                valid_scale = scale > 10
                all_valid   = valid_cdt and valid_cot and valid_ten and valid_scale

                history['step'].append(step)
                history['loss'].append(loss.item())
                history['acc'].append(acc)
                history['cdt'].append(cdt)
                history['cot'].append(cot)
                history['ten'].append(ten)
                history['scale'].append(scale)

                valid_str = '✓ all' if all_valid else f'{"✓" if valid_cdt else "✗"}cdt {"✓" if valid_cot else "✗"}cot {"✓" if valid_ten else "✗"}ten {"✓" if valid_scale else "✗"}sc'

                print(f"  {step:>7,}  {loss.item():>8.4f}  {100*acc:>6.1f}%  {cdt:>7.4f}  {cot:>7.4f}  {ten:>7.4f}  {scale:>8.2f}  {valid_str}")

        elapsed = time.time() - t0
        final_acc = exact_accuracy(model, n=2000, device=device)
        print(f"\n  Final accuracy: {100*final_acc:.2f}%  ({elapsed:.1f}s)")
        print(f"  Final params:")
        print(f"    carry_detect_thresh = {model.carry_detect_thresh.item():.6f}  {'✓ valid (0,9)' if 0 < model.carry_detect_thresh.item() < 9 else '✗ INVALID'}")
        print(f"    carry_out_thresh    = {model.carry_out_thresh.item():.6f}  {'✓ valid (9,10)' if 9 < model.carry_out_thresh.item() < 10 else '✗ INVALID'}")
        print(f"    ten                 = {model.ten.item():.6f}  {'✓ ≈10' if abs(model.ten.item()-10) < 0.5 else '✗ drifted'}")
        print(f"    scale               = {model.scale.item():.4f}  {'✓ large' if model.scale.item() > 10 else '✗ too small'}")

        all_histories[init_mode] = history

        # Save checkpoint
        torch.save({
            'model_state': model.state_dict(),
            'history': history,
            'final_acc': final_acc,
            'init_mode': init_mode,
        }, results_dir / f'carrydiff_{init_mode}.pt')

    # ── Try to plot ────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('Learnable CarryDiff: Do thresholds converge to valid separators?',
                     fontsize=12, fontweight='bold')

        colors = {
            'correct': '#4a9eff',
            'random': '#ff7f50',
            'adversarial': '#50c878',
            'dead': '#8d6e63',
        }

        for init_mode, hist in all_histories.items():
            col = colors.get(init_mode, 'gray')
            steps = hist['step']

            axes[0,0].plot(steps, hist['acc'], color=col, label=init_mode, linewidth=2)
            axes[0,1].plot(steps, hist['cdt'], color=col, label=init_mode, linewidth=2)
            axes[0,2].plot(steps, hist['cot'], color=col, label=init_mode, linewidth=2)
            axes[1,0].plot(steps, hist['loss'], color=col, label=init_mode, linewidth=2)
            axes[1,1].plot(steps, hist['ten'], color=col, label=init_mode, linewidth=2)
            axes[1,2].plot(steps, [math.log10(max(s, 0.01)) for s in hist['scale']],
                          color=col, label=init_mode, linewidth=2)

        axes[0,0].set_title('Exact Accuracy'); axes[0,0].set_ylabel('accuracy'); axes[0,0].legend()
        axes[0,0].set_ylim(-0.05, 1.05)

        axes[0,1].set_title('carry_detect_thresh')
        axes[0,1].axhline(0, color='red', linestyle='--', alpha=0.4, linewidth=1)
        axes[0,1].axhline(9, color='red', linestyle='--', alpha=0.4, linewidth=1)
        axes[0,1].fill_between([0, args.steps], 0, 9, alpha=0.07, color='green')
        axes[0,1].text(args.steps*0.5, 4.5, 'valid region (0,9)', ha='center', fontsize=8, color='green')

        axes[0,2].set_title('carry_out_thresh')
        axes[0,2].axhline(9, color='red', linestyle='--', alpha=0.4, linewidth=1)
        axes[0,2].axhline(10, color='red', linestyle='--', alpha=0.4, linewidth=1)
        axes[0,2].fill_between([0, args.steps], 9, 10, alpha=0.15, color='green')
        axes[0,2].text(args.steps*0.5, 9.5, 'valid region (9,10)', ha='center', fontsize=8, color='green')

        axes[1,0].set_title('MSE Loss'); axes[1,0].set_ylabel('loss')
        axes[1,1].set_title('ten (should → 10)')
        axes[1,1].axhline(10, color='red', linestyle='--', alpha=0.4, linewidth=1)
        axes[1,2].set_title('log10(scale) (should → large)')
        axes[1,2].axhline(1, color='red', linestyle='--', alpha=0.4, linewidth=1)

        for ax in axes.flat:
            ax.set_xlabel('step')
            ax.grid(alpha=0.2)

        plt.tight_layout()
        plot_path = results_dir / 'threshold_training.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n  Plot saved to: {plot_path}")

    except ImportError:
        print("\n  (matplotlib not available — skipping plot)")

    return all_histories


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--device',     type=str,   default='cuda')
    p.add_argument('--steps',      type=int,   default=5000)
    p.add_argument('--batch-size', type=int,   default=512)
    p.add_argument('--lr',         type=float, default=3e-2)
    p.add_argument('--seed',       type=int,   default=42)
    p.add_argument('--init',       type=str,   default='all',
                   choices=['all', 'correct', 'random', 'adversarial', 'dead'])
    args = p.parse_args()
    train(args)
