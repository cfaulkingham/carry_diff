#!/usr/bin/env python3

import numpy as np
import random

NUM_DIGITS = 10
T_STEPS    = 10    # carry-propagation steps = max carry chain in 10-digit addition

# ── 4 Parameters ──────────────────────────────────────────────────────────────
CARRY_DETECT_THRESH = np.float64(6.5)    # 1 param
CARRY_OUT_THRESH    = np.float64(9.5)    # 1 param
TEN                 = np.float64(10.0)   # 1 param
SCALE               = np.float64(1e4)    # 1 param
# ──────────────────────────────────────────────────────────────────────────────

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)   # sigmoid saturation is fine


def _sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


class CarryDiff:
    """
    4-parameter model for 10-digit addition.

    Architecture: iterative carry relaxation, T=10 parallel propagation steps.
    No attention. No FFN. No embeddings.

    Each propagation step, for every output position i in parallel:
      1. carry_in[i]  = sigmoid(SCALE × (raw[i+1] − x_t[i+1] − 8.5))
      2. col[i]       = raw[i] + carry_in[i]
      3. carry_out[i] = sigmoid(SCALE × (col[i] − 9.5))
      4. x_new[i]     = col[i] − 10 × carry_out[i]
    """

    def _step(self, x: np.ndarray, raw: np.ndarray) -> np.ndarray:
        """One parallel carry-propagation step. x, raw: [B, 11] float64."""
        B = x.shape[0]
        pad = np.zeros((B, 1), dtype=np.float64)

        # Each position looks one slot to the right
        raw_right = np.concatenate([raw[:, 1:], pad], axis=1)
        x_right   = np.concatenate([x[:, 1:],   pad], axis=1)

        # Detect whether the right column generated a carry
        carry_in  = _sigmoid(SCALE * (raw_right - x_right - CARRY_DETECT_THRESH))

        # New column sum and output digit
        col       = raw + carry_in
        carry_out = _sigmoid(SCALE * (col - CARRY_OUT_THRESH))
        return col - TEN * carry_out

    def forward(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        a, b : [B, 10] int arrays, left-to-right (most significant first)
        returns: [B, 11] int array of the sum's digits
        """
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        B = a.shape[0]
        z = np.zeros((B, 1), dtype=np.float64)

        # Prepend a zero slot for the potential overflow digit (position 0)
        a11 = np.concatenate([z, a], axis=1)   # [B, 11]
        b11 = np.concatenate([z, b], axis=1)   # [B, 11]
        raw = a11 + b11                         # column-wise raw sums, no carry

        # Initial state: raw sums mod 10, no carries resolved yet
        carry0 = _sigmoid(SCALE * (raw - CARRY_OUT_THRESH))
        x = raw - TEN * carry0   # ≈ raw % 10

        # Iterative carry propagation: T parallel relaxation steps
        for _ in range(T_STEPS):
            x = self._step(x, raw)

        return np.round(x).astype(int)


# ── AdderBoard-compatible interface ───────────────────────────────────────────

def build_model():
    model = CarryDiff()
    metadata = {
        "name":         "CarryDiff",
        "author":       "Claude (Anthropic)",
        "params":       4,
        "architecture": f"Iterative carry relaxation, T={T_STEPS} parallel propagation steps, no attn/FFN/embed",
        "tricks": [
            "Carry uncertainty resolved via iterative relaxation",
            "Parallel position updates (all 11 slots per step)",
            "(raw_sum − digit) ∈ {9,10} iff carry — threshold at 6.5",
            "Differentiable modulo via sharp sigmoid",
            "float64 precision",
        ],
    }
    return model, metadata


def _encode(n: int) -> np.ndarray:
    return np.array([int(c) for c in f"{n:010d}"], dtype=np.int64)


def add(model: CarryDiff, a: int, b: int) -> int:
    digits = model.forward(_encode(a)[None, :], _encode(b)[None, :])[0]
    return int("".join(str(d) for d in digits))


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, meta = build_model()

    print("=" * 60)
    print(f"  {meta['name']}")
    print("=" * 60)
    print(f"  Parameters  : {meta['params']}")
    print(f"  Architecture: {meta['architecture']}")
    print()
    print("  Breakdown:")
    print(f"    CARRY_DETECT_THRESH = 6.5     (1 param)")
    print(f"    CARRY_OUT_THRESH    = 9.5     (1 param)")
    print(f"    TEN                 = 10.0    (1 param)")
    print(f"    SCALE               = 1e4     (1 param)")
    print(f"    {'─'*34}")
    print(f"    Total               = 4")
    print()

    print("  Edge cases:")
    tests = [
        (0,               0),
        (9_999_999_999,   1),
        (9_999_999_999,   9_999_999_999),
        (1_234_567_890,   9_876_543_210),
        (5_000_000_000,   5_000_000_000),
        (1,               9_999_999_998),
        (1_000_000_000,   9_000_000_000),
    ]
    all_ok = True
    for a, b in tests:
        got = add(model, a, b)
        ok  = got == a + b
        all_ok = all_ok and ok
        print(f"    {'✓' if ok else '✗'}  {a:>13,} + {b:>13,}  =  {got:>14,}")
    print(f"  → {'ALL PASS' if all_ok else 'FAILURES FOUND'}")
    print()

    print("  Random accuracy (seed=2025, n=10,000)...")
    random.seed(2025)
    correct = 0
    for _ in range(10_000):
        a = random.randint(0, 9_999_999_999)
        b = random.randint(0, 9_999_999_999)
        if add(model, a, b) == a + b:
            correct += 1
    pct = 100 * correct / 10_000
    print(f"  → {correct}/10000  ({pct:.3f}%)  {'QUALIFIES (≥99%)' if pct >= 99 else 'FAILS'}")
