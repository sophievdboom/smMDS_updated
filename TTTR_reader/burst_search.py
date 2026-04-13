import json
import numpy as np
import pandas as pd
import tttrlib


def burst_loc(indices):
    """
    Reimplementation of the 'burstLoc' idea:
    group consecutive integer indices into runs.

    Returns
    -------
    starts : np.ndarray
        Start index of each run
    lengths : np.ndarray
        Length of each run
    """
    indices = np.asarray(indices, dtype=int).ravel()
    if indices.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    # consecutive runs
    breaks = np.where(np.diff(indices) != 1)[0] + 1
    groups = np.split(indices, breaks)

    starts = np.array([g[0] for g in groups], dtype=int)
    lengths = np.array([len(g) for g in groups], dtype=int)
    return starts, lengths


def simple_moving_average(x, window):
    """
    Simple fallback filter.
    This is NOT guaranteed to match the original Lee filter exactly.
    Use this only as a temporary placeholder if you do not yet port
    leeFilter1D_Add / leeFilter1D_matlab.
    """
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x.copy()
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="same")


def get_macro_time_resolution_s(tttr_obj):
    """
    Prefer the documented header tag route.
    Falls back to top-level JSON field search if needed.
    """
    try:
        return float(tttr_obj.header.tag("MeasDesc_GlobalResolution")["value"])
    except Exception:
        header = json.loads(tttr_obj.header.json)
        if "MeasDesc_GlobalResolution" in header:
            return float(header["MeasDesc_GlobalResolution"])
        for tag in header.get("tags", []):
            if tag.get("name") == "MeasDesc_GlobalResolution":
                return float(tag["value"])
        raise KeyError("Could not find MeasDesc_GlobalResolution in header")


def get_tttr_photons_from_file(ptu_filename, verbose=True):
    """
    Recreate the old Photons array:
    Photons[:, 0] = channel
    Photons[:, 1] = macro time in ns
    """
    tttr = tttrlib.TTTR(ptu_filename)

    routing_channels = np.asarray(tttr.routing_channels)
    macro_times_raw = np.asarray(tttr.macro_times)

    macro_time_resolution_s = get_macro_time_resolution_s(tttr)

    # Convert to ns to match the old smMDS script convention
    macro_times_ns = macro_times_raw.astype(np.float64) * macro_time_resolution_s * 1e9

    photons = np.column_stack((routing_channels.astype(np.int64), macro_times_ns))

    if verbose:
        print("=== STEP 1: READ TTTR DATA ===")
        print(f"File: {ptu_filename}")
        print(f"Number of photons/events: {len(photons)}")
        print(f"Unique routing channels: {np.unique(routing_channels)}")
        print(f"Macro time resolution: {macro_time_resolution_s:.12e} s")
        print("First 10 photons [channel, macro_time_ns]:")
        print(photons[:10])

    return tttr, photons


def recreate_smds_burst_search(
    ptu_filename,
    set_lee_filter=4,
    threshold_iT_signal_ms=0.05,
    threshold_iT_noise_ms=0.05,
    min_phs_burst=5,
    min_phs_noise=100,
    verbose=True,
    use_placeholder_filter=True
):
    """
    Recreate the burst-search logic from the original smMDS code
    using tttrlib as the file reader.

    Notes
    -----
    - This reproduces the logic structure of the old code.
    - The exact match depends on using the SAME Lee filter implementation.
    - For now, this uses a simple moving average if use_placeholder_filter=True.
    """

    tttr, photons = get_tttr_photons_from_file(ptu_filename, verbose=verbose)

    # same variable names as original script
    macroT = photons[:, 1]      # ns
    channel = photons[:, 0]

    if verbose:
        print("\n=== STEP 2: EXTRACT MACRO TIMES + CHANNELS ===")
        print(f"macroT shape: {macroT.shape}")
        print(f"channel shape: {channel.shape}")
        print(f"macroT min/max (ns): {macroT.min():.3f} / {macroT.max():.3f}")

    # Inter-photon times in ns
    interPhT = np.diff(macroT)

    if verbose:
        print("\n=== STEP 3: INTER-PHOTON TIMES ===")
        print(f"interPhT shape: {interPhT.shape}")
        print(f"First 20 inter-photon times (ns):")
        print(interPhT[:20])

    # Filter
    if use_placeholder_filter:
        interLee = simple_moving_average(interPhT, set_lee_filter)
        filter_used = "simple_moving_average PLACEHOLDER"
    else:
        raise NotImplementedError(
            "Port your exact leeFilter1D_Add / leeFilter1D_matlab here for exact matching."
        )

    if verbose:
        print("\n=== STEP 4: FILTERED INTER-PHOTON TIMES ===")
        print(f"Filter used: {filter_used}")
        print(f"First 20 filtered values (ns):")
        print(interLee[:20])

    # Original thresholds were in ms, converted in original code as:
    # interLee < threIT * 1e6
    # because interLee is in ns and 1 ms = 1e6 ns
    signal_upper_ns = threshold_iT_signal_ms * 1e6
    noise_lower_ns = threshold_iT_noise_ms * 1e6

    indexSig = np.where((0.4 < interLee) & (interLee < signal_upper_ns))[0]
    indexSigN = np.where(interLee > noise_lower_ns)[0]

    if verbose:
        print("\n=== STEP 5: THRESHOLDING ===")
        print(f"Signal threshold upper bound: {signal_upper_ns} ns")
        print(f"Noise threshold lower bound: {noise_lower_ns} ns")
        print(f"Number of signal-like indices: {len(indexSig)}")
        print(f"Number of noise-like indices: {len(indexSigN)}")
        print("First 20 signal indices:", indexSig[:20])
        print("First 20 noise indices:", indexSigN[:20])

    if len(indexSig) == 0:
        if verbose:
            print("\nNo signal indices found. No bursts returned.")
        df1 = pd.DataFrame({"channel": channel, "photons": macroT})
        df2 = pd.DataFrame(columns=["Burst timestart", "Burst intensity", "Burst duration"])
        return df1, df2

    bStart, bLength = burst_loc(indexSig)
    bStartN, bLengthN = burst_loc(indexSigN)

    if verbose:
        print("\n=== STEP 6: GROUP CONSECUTIVE INDICES INTO REGIONS ===")
        print(f"Signal regions found: {len(bStart)}")
        print(f"Noise regions found: {len(bStartN)}")
        print("First 10 signal starts:", bStart[:10])
        print("First 10 signal lengths:", bLength[:10])

    # 2nd filter: minimum photons
    bStartLong = bStart[bLength >= min_phs_burst]
    bLengthLong = bLength[bLength >= min_phs_burst].astype(int)

    bStartLongN = bStartN[bLengthN >= min_phs_noise].astype(int)
    bLengthLongN = bLengthN[bLengthN >= min_phs_noise].astype(int)

    if verbose:
        print("\n=== STEP 7: MINIMUM-LENGTH FILTER ===")
        print(f"Bursts kept: {len(bStartLong)}")
        print(f"Noise regions kept: {len(bStartLongN)}")
        print("Burst starts kept:", bStartLong[:10])
        print("Burst lengths kept:", bLengthLong[:10])

    # Collect photons in each burst
    bursts = np.zeros((int(np.sum(bLengthLong)), 3), dtype=float)
    lInd = 0

    for i in range(len(bStartLong)):
        burst_number = np.ones(bLengthLong[i]) * i
        photons2 = photons[bStartLong[i]:(bStartLong[i] + bLengthLong[i])]
        bursts[lInd:lInd + bLengthLong[i], :] = np.concatenate(
            (burst_number[:, np.newaxis], photons2),
            axis=1
        )
        lInd += bLengthLong[i]

    if verbose:
        print("\n=== STEP 8: COLLECT BURST PHOTONS ===")
        print(f"bursts array shape: {bursts.shape}")
        print("First 10 rows [burst_id, channel, macro_time_ns]:")
        print(bursts[:10])

    # Background estimate from long noise regions
    BackN = 0
    BackT = 0.0

    for i in range(len(bStartLongN)):
        gap_photons = photons[bStartLongN[i]: bStartLongN[i] + bLengthLongN[i]]
        BackT += gap_photons[-1, 1] - gap_photons[0, 1]   # ns
        BackN += gap_photons.shape[0]

    BI = BackN / BackT * 1e6 if BackT != 0 else 0.0

    if verbose:
        print("\n=== STEP 9: BACKGROUND ESTIMATE ===")
        print(f"BackN: {BackN}")
        print(f"BackT (ns): {BackT}")
        print(f"BI (counts/ms): {BI}")

    # Burst intensity and duration
    NI = np.zeros(len(bStartLong), dtype=float)
    TBurst = np.zeros(len(bStartLong), dtype=float)

    for i in range(len(bStartLong)):
        N = bursts[bursts[:, 0] == i, 2]
        NI[i] = N.size

        # Original code:
        # TBurst[i] = np.sum(interPhT[bStartLong[i] - 1 : bStartLong[i] + bLengthLong[i] - 1] * 1e-6)
        # interPhT in ns -> ms
        start_idx = max(bStartLong[i] - 1, 0)
        stop_idx = bStartLong[i] + bLengthLong[i] - 1
        TBurst[i] = np.sum(interPhT[start_idx:stop_idx] * 1e-6)

    if verbose:
        print("\n=== STEP 10: FINAL BURST METRICS ===")
        print(f"Number of bursts: {len(NI)}")
        print("First 10 burst intensities (photon counts):", NI[:10])
        print("First 10 burst durations (ms):", TBurst[:10])

    df1 = pd.DataFrame({
        "channel": channel,
        "photons": photons[:, 1],   # ns
    })

    df2 = pd.DataFrame({
        "Burst timestart": photons[bStartLong, 1],  # ns
        "Burst intensity": NI,
        "Burst duration": TBurst,                   # ms
    })

    if verbose:
        print("\n=== STEP 11: DATAFRAMES ===")
        print("Photons dataframe head:")
        print(df1.head())
        print("\nBursts dataframe head:")
        print(df2.head())

    return df1, df2