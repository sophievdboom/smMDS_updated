import numpy as np
import time


def burstLoc(Arr, mindistance):
    """
    Fast burst localization.

    Assumes Arr is already:
    - 1D
    - sorted
    - unique

    That is true if Arr comes from np.flatnonzero(...)
    """
    t0 = time.perf_counter()
    print("        burstLoc: converting input array...")

    Arr = np.asarray(Arr, dtype=np.int64).ravel()
    print(f"        burstLoc: input size = {Arr.size}")

    if Arr.size == 0:
        print(f"        burstLoc: empty input, done in {time.perf_counter() - t0:.3f} s")
        return np.array([], dtype=int), np.array([], dtype=int)

    t1 = time.perf_counter()
    diffs = np.diff(Arr)
    run_breaks = np.flatnonzero(diffs > 1) + 1

    run_starts_idx = np.concatenate(([0], run_breaks))
    run_ends_idx = np.concatenate((run_breaks, [Arr.size]))

    bStart = Arr[run_starts_idx]
    bLength = run_ends_idx - run_starts_idx

    print(f"        burstLoc: run detection done in {time.perf_counter() - t1:.3f} s")
    print(f"        burstLoc: number of raw runs = {bStart.size}")

    if mindistance <= 1:
        print(f"        burstLoc: total done in {time.perf_counter() - t0:.3f} s")
        return bStart.astype(int), bLength.astype(int)

    t1 = time.perf_counter()
    keep = np.ones(bStart.size, dtype=bool)
    keep[1:] = (bStart[1:] - (bStart[:-1] + bLength[:-1] - 1)) > mindistance

    bStartAcc = bStart[keep]
    bLengthAcc = bLength[keep]

    print(f"        burstLoc: mindistance filter done in {time.perf_counter() - t1:.3f} s")
    print(f"        burstLoc: accepted runs = {bStartAcc.size}")
    print(f"        burstLoc: total done in {time.perf_counter() - t0:.3f} s")

    return bStartAcc.astype(int), bLengthAcc.astype(int)