import numpy as nu
import re, random
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ----------------------------
# Constants (no magic numbers)
# ----------------------------
FOOTPRINT = 147  # nucleosome footprint (bp)
SMOOTH_WINDOW = 10  # boxcar smoothing window length
OCCUPANCY_KERNEL = 146  # integration kernel length for occupancy

# Fast ASCII -> base index lookup (A,C,G,T) -> (0,1,2,3)
_ASCII_TO_IDX = nu.full(256, -1, dtype=nu.int8)
_ASCII_TO_IDX[ord("A")] = 0
_ASCII_TO_IDX[ord("C")] = 1
_ASCII_TO_IDX[ord("G")] = 2
_ASCII_TO_IDX[ord("T")] = 3


@dataclass(frozen=True)
class NucPositionResult:
    """Named outputs for clarity."""

    energy_raw: nu.ndarray
    energy_smoothed: nu.ndarray
    dyad_probability: nu.ndarray
    occupancy: nu.ndarray


def CleanSeq(dna: str) -> str:
    """Uppercase, convert U->T, and keep only A/C/G/T."""
    dna = dna.upper().replace("U", "T")
    return re.sub(r"[^GATC]", "", dna)


def encode_seq(seq: str) -> nu.ndarray:
    """Encode DNA string to integer array using a fast lookup table.
    Assumes the sequence contains only A/C/G/T. Use CleanSeq beforehand.
    """
    b = nu.frombuffer(seq.encode("ascii"), dtype=nu.uint8)
    return _ASCII_TO_IDX[b]


def getweight(w: int, p: float, b: float) -> nu.ndarray:
    """Vectorized dinucleotide probability weights.
    Returns array with shape (4, 4, w) where [prev][curr][s].
    """
    x = nu.arange(w, dtype=nu.float64)
    s = b * nu.sin(2 * nu.pi * x / p)

    weights = nu.empty((4, 4, w), dtype=nu.float64)

    # Row A (prev=A)
    weights[0, 0] = 0.25 + s  # AA
    weights[0, 1] = 0.25 - s / 3  # AC
    weights[0, 2] = 0.25 - s / 3  # AG
    weights[0, 3] = 0.25 - s / 3  # AT

    # Row C (prev=C)
    weights[1, 0] = 0.25  # CA
    weights[1, 1] = 0.25  # CC
    weights[1, 2] = 0.25  # CG
    weights[1, 3] = 0.25  # CT

    # Row G (prev=G)
    weights[2, 0] = 0.25 + s / 3  # GA
    weights[2, 1] = 0.25 - s  # GC
    weights[2, 2] = 0.25 + s / 3  # GG
    weights[2, 3] = 0.25 + s / 3  # GT

    # Row T (prev=T)
    weights[3, 0] = 0.25 + s  # TA
    weights[3, 1] = 0.25 - s  # TC
    weights[3, 2] = 0.25 - s  # TG
    weights[3, 3] = 0.25 + s  # TT

    return weights


def calcE(seq: str, w: int, amplitude: float, period: float) -> nu.ndarray:
    """Compute energy-like score E using vectorized operations.
    - seq: DNA sequence string (A/C/G/T)
    - w: window size (bp)
    - amplitude, period: periodical weight params
    Returns: array of length len(seq) - w
    """
    idx = encode_seq(seq)
    L = idx.size
    num_win = L - w
    if num_win <= 0:
        return nu.array([], dtype=nu.float64)

    # Precompute weights and their logs once (avoid log in loop)
    weights = getweight(w, period, amplitude)  # shape (4,4,w)
    log_weights = nu.log(nu.clip(weights, 1e-300, None))

    log_p_f = nu.zeros(num_win, dtype=nu.float64)
    log_p_r = nu.zeros(num_win, dtype=nu.float64)

    i = nu.arange(num_win)
    for s in range(w):
        # Forward: prob_array[prev=idx[i+s-1], curr=idx[i+s], s]
        prev_f = idx[i + s - 1]
        curr_f = idx[i + s]
        log_p_f += log_weights[prev_f, curr_f, s]

        # Reverse: prob_array[3-idx[i+w-s], 3-idx[i+w-s-1], s]
        a = idx[i + w - s]
        b = idx[i + w - s - 1]
        rprev = 3 - a
        rcurr = 3 - b
        log_p_r += log_weights[rprev, rcurr, s]

    # Convert back from log-space and scale
    p_f = nu.exp(log_p_f) * (4.0**w)
    p_r = nu.exp(log_p_r) * (4.0**w)

    # Align reverse as in original code (shift by -1)
    p_r = nu.roll(p_r, -1)

    # Energy combination
    E = (p_r * nu.log(p_r) + p_f * nu.log(p_f)) / (p_r + p_f)
    return E


def smooth(x: nu.ndarray, window_len: int) -> nu.ndarray:
    """Simple boxcar smoothing with reflection padding."""
    if window_len <= 1 or x.size == 0:
        return x
    s = nu.r_[x[window_len - 1 : 0 : -1], x, x[-1:-window_len:-1]]
    w = nu.ones(window_len, dtype=nu.float64)
    y = nu.convolve(w / w.sum(), s, mode="valid")
    return y[
        len(x[window_len - 1 : 0 : -1]) : len(x[window_len - 1 : 0 : -1]) + len(x) + 1
    ]


def vanderlick(Energy: nu.ndarray, mu: float) -> nu.ndarray:
    """Optimized O(n) implementation using sliding window sums.
    P = forward * reversed(backward), with footprint exclusion.
    """
    E_out = Energy - mu

    n = E_out.size
    forward = nu.zeros(n, dtype=nu.float64)

    # forward[i] = exp(E_out[i] - sum(forward[max(i-FOOTPRINT,0):i]))
    sum_prev = 0.0
    for i in range(n):
        forward[i] = nu.exp(E_out[i] - sum_prev)
        sum_prev += forward[i]
        if i >= FOOTPRINT:
            sum_prev -= forward[i - FOOTPRINT]

    # backward as in original but with sliding dot product
    backward = nu.zeros(n, dtype=nu.float64)
    r_forward = forward[::-1]

    # backward[i] = 1 - sum(r_forward[max(i-FOOTPRINT,0):i] * backward[max(i-FOOTPRINT,0):i])
    sum_prod = 0.0
    for i in range(n):
        backward[i] = 1.0 - sum_prod
        sum_prod += r_forward[i] * backward[i]
        if i >= FOOTPRINT:
            sum_prod -= r_forward[i - FOOTPRINT] * backward[i - FOOTPRINT]

    return forward * backward[::-1]


def CreateDNA(dnalength: int) -> str:
    """Create a random sequence flanking a 601 sequence to the requested length."""
    dna601 = "ACAGGATGTATATATCTGACACGTGCCTGGAGACTAGGGAGTAATCCCCTTGGCGGTTAAAACGCGGGGGACAGCGCGTACGTGCGTTTAAGCGGTGCTAGAGCTGTCTACGACCAATTGAGCGGCCTCGGCACCGGGATTCTCCAG"
    flanklength = (dnalength - len(dna601)) // 2
    dna = (
        "".join(random.choice("ACGT") for _ in range(flanklength))
        + dna601
        + "".join(random.choice("ACGT") for _ in range(flanklength))
    )
    return CleanSeq(dna)


def CalcNucPositions(
    sequence: str,
    w: int,
    chemical_potential: float,
    amplitude: float,
    period: float,
) -> NucPositionResult:
    """Compute energies, smoothed energies, dyad probability, and occupancy.
    Returns a NucPositionResult.
    """
    sequence = CleanSeq(sequence)

    energy_raw = calcE(sequence, w, amplitude, period)
    energy_smoothed = smooth(energy_raw, SMOOTH_WINDOW)
    dyad_probability = vanderlick(energy_smoothed, chemical_potential)

    # Pad probability to align with original behavior (asymmetric padding)
    left_pad = (w + 1) // 2
    right_pad = w // 2
    dyad_probability = nu.concatenate(
        (
            nu.zeros(left_pad, dtype=nu.float64),
            dyad_probability,
            nu.zeros(right_pad, dtype=nu.float64),
        )
    )

    occupancy = nu.convolve(
        dyad_probability, nu.ones(OCCUPANCY_KERNEL, dtype=nu.float64), mode="same"
    )

    return NucPositionResult(
        energy_raw=energy_raw,
        energy_smoothed=energy_smoothed,
        dyad_probability=dyad_probability,
        occupancy=occupancy,
    )


# Default params
footprint = 147
chemical_potential = -8.5
amplitude = 0.2
period = 10.1

if __name__ == "__main__":
    seq = CreateDNA(2000)
    res = CalcNucPositions(seq, footprint, chemical_potential, amplitude, period)
    plt.plot(res.occupancy)
    plt.xlabel("position (bp)")
    plt.ylabel("P")
    plt.show()
