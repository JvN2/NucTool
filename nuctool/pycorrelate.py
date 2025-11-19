import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from tqdm import tqdm
from numba import jit, njit


def pnormalize(g: np.ndarray, t: np.ndarray, u: np.ndarray, bins: np.ndarray) -> None:
    if t.size > 0 and u.size > 0:
        tmax = np.max(t)
        tmin = np.min(t)
        umax = np.max(u)
        umin = np.min(u)
        duration = max(tmax, umax) - min(tmin, umin)

        for i in range(len(bins) - 1):
            tau = 0.5 * (bins[i] + bins[i + 1])
            a = np.sum(t <= (umax - tau))
            b = np.sum(u >= (tmin + tau))
            if a * b > 0:
                g[i] *= (duration - tau) / (a * b)


@njit
def binary_search_left(arr, value):
    """Binary search for left insertion point (equivalent to searchsorted)"""
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < value:
            left = mid + 1
        else:
            right = mid
    return left


@njit
def pcorrelate_core(t, u, bins):
    """Optimized numba-accelerated core correlation computation"""
    nbins = len(bins) - 1
    counts = np.zeros(nbins, dtype=np.float64)

    for ti in t:
        # For each t[i], find counts in each bin using binary search
        for k in range(nbins):
            left_edge = ti + bins[k]
            right_edge = ti + bins[k + 1]

            # Count elements in u that fall within [left_edge, right_edge)
            left_idx = binary_search_left(u, left_edge)
            right_idx = binary_search_left(u, right_edge)

            counts[k] += right_idx - left_idx

    return counts


def pcorrelate(
    t: np.ndarray, u: np.ndarray, bins: np.ndarray, normalize: bool
) -> np.ndarray:
    # Ensure float arrays and sort u
    t = np.asarray(t, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    bins = np.asarray(bins, dtype=np.float64)
    u = np.sort(u)

    # Use numba-accelerated core
    counts = pcorrelate_core(t, u, bins)

    diff = bins[1:] - bins[:-1]
    g = counts / diff
    if normalize:
        pnormalize(g, t, u, bins)
    return g


def auto_correlate(t: np.ndarray, bins: np.ndarray, normalize: bool) -> np.ndarray:
    return pcorrelate(t, t, bins, normalize)


if __name__ == "__main__":
    # Example input data
    bins = np.linspace(0, 1000, 1000).astype(float)
    correlation = np.zeros_like(bins[:-1])

    n_iterations = 10
    for _ in tqdm(range(n_iterations), desc="Calculating correlation"):
        t = np.cumsum(np.random.normal(167, 20, size=100000))
        correlation += pcorrelate(t, t, bins, normalize=True)

    plt.plot(bins[:-1], correlation / n_iterations, drawstyle="steps-post")
    # plt.xscale("log")
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    plt.ylim(0, 5)
    plt.show()
