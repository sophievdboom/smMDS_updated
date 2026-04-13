import json
import time
import numpy as np
import pandas as pd
import tttrlib

from scipy.ndimage import uniform_filter

from leefilter_new import leeFilter1D_matlab
from burstloc_new import burstLoc


def get_macro_resolution_s(tttr_obj):
    header = json.loads(tttr_obj.header.json)

    if "MeasDesc_GlobalResolution" in header:
        return float(header["MeasDesc_GlobalResolution"])

    for tag in header.get("tags", []):
        if tag.get("name") == "MeasDesc_GlobalResolution":
            return float(tag["value"])

    raise KeyError("MeasDesc_GlobalResolution not found in header.")


def _lee_filter_add_with_global_variance(I, window_size, overall_variance):
    """
    Chunk-safe additive Lee filter:
    local mean/variance from the chunk, global variance from the whole trace.
    """
    I = np.asarray(I, dtype=np.float64)
    mean_I = uniform_filter(I, window_size)
    sqr_mean_I = uniform_filter(I ** 2, window_size)
    var_I = sqr_mean_I - mean_I ** 2

    weight_I = var_I / (var_I + overall_variance)
    output_I = mean_I + weight_I * (I - mean_I)
    return output_I


def _combine_mean_variance(n_total, mean_total, m2_total, x):
    """
    Merge running mean/variance statistics with a new batch x.
    """
    x = np.asarray(x, dtype=np.float64)
    n_batch = x.size
    if n_batch == 0:
        return n_total, mean_total, m2_total

    mean_batch = np.mean(x, dtype=np.float64)
    m2_batch = np.sum((x - mean_batch) ** 2, dtype=np.float64)

    if n_total == 0:
        return n_batch, mean_batch, m2_batch

    delta = mean_batch - mean_total
    n_new = n_total + n_batch
    mean_new = mean_total + delta * n_batch / n_new
    m2_new = m2_total + m2_batch + delta * delta * n_total * n_batch / n_new

    return n_new, mean_new, m2_new


class BooleanRunCollector:
    """
    Collect runs of True values from boolean chunks without storing giant index arrays.
    Equivalent in spirit to burstLoc on index arrays.
    """

    def __init__(self):
        self.starts = []
        self.lengths = []

        self.open_run = False
        self.open_start = None
        self.open_length = 0

    def _close_open_run(self):
        if self.open_run:
            self.starts.append(self.open_start)
            self.lengths.append(self.open_length)
            self.open_run = False
            self.open_start = None
            self.open_length = 0

    def consume(self, mask, absolute_offset):
        mask = np.asarray(mask, dtype=bool)
        n = mask.size
        if n == 0:
            return

        # If previous chunk ended with an open run and this chunk starts False,
        # close that run now.
        if self.open_run and not mask[0]:
            self._close_open_run()

        # Find run starts and ends in this chunk
        diff = np.diff(mask.astype(np.int8))
        starts = np.flatnonzero(diff == 1) + 1
        ends = np.flatnonzero(diff == -1) + 1

        if mask[0]:
            starts = np.r_[0, starts]
        if mask[-1]:
            ends = np.r_[ends, n]

        if starts.size == 0:
            return

        run_idx = 0

        # Extend an open run from previous chunk if needed
        if self.open_run and starts[0] == 0:
            first_end = ends[0]
            self.open_length += first_end

            if first_end < n:
                self._close_open_run()
            else:
                # Run continues through the entire chunk
                return

            run_idx = 1

        for s, e in zip(starts[run_idx:], ends[run_idx:]):
            length = int(e - s)
            abs_start = int(absolute_offset + s)

            if e == n:
                # This run stays open into the next chunk
                self.open_run = True
                self.open_start = abs_start
                self.open_length = length
            else:
                self.starts.append(abs_start)
                self.lengths.append(length)

    def finalize(self):
        self._close_open_run()

        if len(self.starts) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        return (
            np.asarray(self.starts, dtype=np.int64),
            np.asarray(self.lengths, dtype=np.int64),
        )


def _select_events(tttr, user_setting):
    """
    Event-selection hook.
    For T2 mode this keeps current behavior.
    Later this can be extended for PIE-T3 microtime gating.

    Returns
    -------
    channels : np.ndarray
    macro_times_raw : np.ndarray
    """
    tttr_mode = str(user_setting.get("tttr_mode", "T2")).upper()
    allowed_channels = user_setting.get("allowed_routing_channels", None)

    channels = np.asarray(tttr.routing_channels)
    macro_times_raw = np.asarray(tttr.macro_times)

    if tttr_mode == "PIE_T3":
        # Reserved for future microtime gating.
        pie_gate = user_setting.get("pie_microtime_gate", None)
        if pie_gate is not None:
            raise NotImplementedError(
                "PIE-T3 microtime gating is not implemented yet. "
                "Leave pie_microtime_gate=None for now."
            )

    if allowed_channels is not None:
        allowed_channels = np.asarray(allowed_channels)
        mask = np.isin(channels, allowed_channels)
        channels = channels[mask]
        macro_times_raw = macro_times_raw[mask]

    return channels, macro_times_raw


def _compute_overall_variance_of_diffs(macro_times_raw, chunk_size):
    """
    Compute variance of inter-photon times in ticks, chunked.
    """
    n_events = len(macro_times_raw)
    n_diff = n_events - 1

    n_total = 0
    mean_total = 0.0
    m2_total = 0.0

    print("\n[3a/9] Computing global inter-photon variance in chunks...")

    for start in range(0, n_diff, chunk_size):
        end = min(start + chunk_size, n_diff)

        # Need macro_times_raw[start:end+1] to make diffs for diff indices start:end
        diffs = np.diff(macro_times_raw[start:end + 1]).astype(np.float64)

        n_total, mean_total, m2_total = _combine_mean_variance(
            n_total, mean_total, m2_total, diffs
        )

        if start == 0 or ((start // chunk_size) + 1) % 10 == 0 or end == n_diff:
            print(
                f"    variance pass: processed diff indices {start:,} to {end:,} "
                f"of {n_diff:,}"
            )

    if n_total < 2:
        return 0.0

    variance = m2_total / n_total
    print(f"    Global inter-photon variance = {variance:.6e} ticks^2")
    return variance


def _chunked_boolean_runs(
    macro_times_raw,
    resolution_s,
    setLeeFilter,
    filter_name,
    signal_upper_ms,
    noise_lower_ms,
    use_noise_regions,
    chunk_size,
    overall_variance,
):
    """
    Chunked Lee filtering + thresholding + run collection.
    Avoids building giant interPhT/interLee/index arrays.
    """
    n_events = len(macro_times_raw)
    n_diff = n_events - 1

    # Convert thresholds to ticks
    tick_s = resolution_s
    lower_bound_ticks = 0.4e-9 / tick_s
    signal_upper_ticks = signal_upper_ms * 1e-3 / tick_s
    noise_lower_ticks = noise_lower_ms * 1e-3 / tick_s

    print("\n[4/9] Chunked Lee filter + thresholding...")
    print(f"    Tick duration: {tick_s:.12e} s")
    print(f"    lower_bound_ticks  = {lower_bound_ticks:.3f}")
    print(f"    signal_upper_ticks = {signal_upper_ticks:.3f}")
    print(f"    noise_lower_ticks  = {noise_lower_ticks:.3f}")
    print(f"    chunk_size         = {chunk_size:,}")

    # Halo for chunk-safe filtering
    halo = max(2, int(setLeeFilter) + 2)

    signal_runs = BooleanRunCollector()
    noise_runs = BooleanRunCollector() if use_noise_regions else None

    first_signal_preview = None
    first_noise_preview = None

    for start in range(0, n_diff, chunk_size):
        end = min(start + chunk_size, n_diff)

        ext_start = max(0, start - halo)
        ext_end = min(n_diff, end + halo)

        diffs_ext = np.diff(macro_times_raw[ext_start:ext_end + 1]).astype(np.float64)

        if filter_name == "addLeefilter":
            interLee_ext = _lee_filter_add_with_global_variance(
                diffs_ext, setLeeFilter, overall_variance
            )
        elif filter_name == "matlabLeefilter":
            interLee_ext = leeFilter1D_matlab(diffs_ext, setLeeFilter)
        else:
            raise RuntimeError("filter name does not exist")

        core_slice = slice(start - ext_start, start - ext_start + (end - start))
        interLee_core = interLee_ext[core_slice]

        signal_mask = (interLee_core > lower_bound_ticks) & (interLee_core < signal_upper_ticks)
        signal_runs.consume(signal_mask, start)

        if use_noise_regions:
            noise_mask = interLee_core > noise_lower_ticks
            noise_runs.consume(noise_mask, start)

        if first_signal_preview is None:
            first_signal_preview = np.flatnonzero(signal_mask)[:20] + start
        if use_noise_regions and first_noise_preview is None:
            first_noise_preview = np.flatnonzero(noise_mask)[:20] + start

        if start == 0 or ((start // chunk_size) + 1) % 10 == 0 or end == n_diff:
            print(
                f"    filter pass: processed diff indices {start:,} to {end:,} "
                f"of {n_diff:,}"
            )

    bStart, bLength = signal_runs.finalize()

    if use_noise_regions:
        bStartN, bLengthN = noise_runs.finalize()
    else:
        bStartN = np.array([], dtype=np.int64)
        bLengthN = np.array([], dtype=np.int64)

    print(f"    Signal runs found: {len(bStart)}")
    print(f"    Noise runs found:  {len(bStartN)}")
    if first_signal_preview is not None:
        print(f"    First 20 signal-like diff indices: {first_signal_preview}")
    if use_noise_regions and first_noise_preview is not None:
        print(f"    First 20 noise-like diff indices:  {first_noise_preview}")

    return bStart, bLength, bStartN, bLengthN


def get_bursts(ptufilename, user_setting=None):
    """
    Memory-optimized burst search using tttrlib.
    Safe for T2 now, and prepared for future PIE-T3 hooks.

    Returns
    -------
    photons_dataframe : pd.DataFrame
        Debug-only photon table (first N photons, not the full stream).
        Important metadata are stored in .attrs:
            - total_photons_in_file
            - macro_resolution_s

    bursts_dataframe : pd.DataFrame
        Columns:
            - Burst timestart  (ns)
            - Burst intensity  (number of photons)
            - Burst duration   (ms)
    """
    total_t0 = time.perf_counter()
    print("\n=== get_bursts: START ===")
    print(f"Input file: {ptufilename}")

    default_setting = {
        "set_lee_filter": 2,
        "threshold_iT_signal": 0.05,
        "threshold_iT_noise": 0.1,
        "min_phs_burst": 10,
        "min_phs_noise": 160,
        "filter_name": "addLeefilter",
        "show_plot": False,
        "output_folder": "output",
        "use_noise_regions": True,
        "tttr_mode": "T2",
        "allowed_routing_channels": None,
        "pie_microtime_gate": None,      # reserved for future PIE-T3
        "diff_chunk_size": 5_000_000,
        "debug_photons_n": 0,            # keep 0 if you do not want debug photon rows
    }

    if user_setting is None:
        user_setting = {}

    for key in default_setting:
        if key not in user_setting:
            user_setting[key] = default_setting[key]

    setLeeFilter = user_setting["set_lee_filter"]
    threIT = user_setting["threshold_iT_signal"]
    threIT2 = user_setting["threshold_iT_noise"]
    minPhs = user_setting["min_phs_burst"]
    minPhsN = user_setting["min_phs_noise"]
    filter_name = user_setting["filter_name"]
    use_noise_regions = user_setting["use_noise_regions"]
    diff_chunk_size = int(user_setting["diff_chunk_size"])
    debug_photons_n = int(user_setting["debug_photons_n"])

    print("Settings loaded:")
    print(f"  set_lee_filter        = {setLeeFilter}")
    print(f"  threshold_iT_signal   = {threIT} ms")
    print(f"  threshold_iT_noise    = {threIT2} ms")
    print(f"  min_phs_burst         = {minPhs}")
    print(f"  min_phs_noise         = {minPhsN}")
    print(f"  filter_name           = {filter_name}")
    print(f"  output_folder         = {user_setting['output_folder']}")
    print(f"  use_noise_regions     = {use_noise_regions}")
    print(f"  tttr_mode             = {user_setting['tttr_mode']}")
    print(f"  allowed_routing_channels = {user_setting['allowed_routing_channels']}")
    print(f"  diff_chunk_size       = {diff_chunk_size:,}")
    print(f"  debug_photons_n       = {debug_photons_n}")

    # ------------------------------------------------------------------
    # 1. Load TTTR file
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    print("\n[1/9] Loading TTTR file...")
    tttr = tttrlib.TTTR(ptufilename)
    print(f"    Done in {time.perf_counter() - t0:.3f} s")

    # ------------------------------------------------------------------
    # 2. Extract selected events
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    print("\n[2/9] Extracting selected channels and macro times...")
    channels, macro_times_raw = _select_events(tttr, user_setting)
    resolution_s = get_macro_resolution_s(tttr)

    print(f"    Done in {time.perf_counter() - t0:.3f} s")
    print(f"    Number of selected events: {len(macro_times_raw):,}")
    print(f"    Unique channels: {np.unique(channels)}")
    print(f"    Macro-time resolution: {resolution_s:.12e} s")

    if len(macro_times_raw) == 0:
        print("    No events found in file.")
        df1 = pd.DataFrame(columns=["channel", "photons"])
        df1.attrs["total_photons_in_file"] = 0
        df1.attrs["macro_resolution_s"] = resolution_s

        df2 = pd.DataFrame(columns=["Burst timestart", "Burst intensity", "Burst duration"])
        print(f"\n=== get_bursts: END ({time.perf_counter() - total_t0:.3f} s) ===")
        return df1, df2

    if len(macro_times_raw) == 1:
        print("    Only one event found, no inter-photon times available.")
        debug_n = min(debug_photons_n, len(macro_times_raw))
        df1 = pd.DataFrame({
            "channel": channels[:debug_n],
            "photons": macro_times_raw[:debug_n].astype(np.float64) * resolution_s * 1e9,
        })
        df1.attrs["total_photons_in_file"] = int(len(macro_times_raw))
        df1.attrs["macro_resolution_s"] = resolution_s

        df2 = pd.DataFrame(columns=["Burst timestart", "Burst intensity", "Burst duration"])
        print(f"\n=== get_bursts: END ({time.perf_counter() - total_t0:.3f} s) ===")
        return df1, df2

    # ------------------------------------------------------------------
    # 3. Compute global variance of inter-photon times (chunked)
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    overall_variance = _compute_overall_variance_of_diffs(macro_times_raw, diff_chunk_size)
    print(f"    Done in {time.perf_counter() - t0:.3f} s")

    # ------------------------------------------------------------------
    # 4-6. Chunked Lee filter, thresholding, run collection
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    bStart, bLength, bStartN, bLengthN = _chunked_boolean_runs(
        macro_times_raw=macro_times_raw,
        resolution_s=resolution_s,
        setLeeFilter=setLeeFilter,
        filter_name=filter_name,
        signal_upper_ms=threIT,
        noise_lower_ms=threIT2,
        use_noise_regions=use_noise_regions,
        chunk_size=diff_chunk_size,
        overall_variance=overall_variance,
    )
    print(f"    Done in {time.perf_counter() - t0:.3f} s")

    # ------------------------------------------------------------------
    # 7. Apply minimum burst/noise length filters
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    print("\n[7/9] Applying minimum burst/noise length filters...")

    bStartLong = bStart[bLength >= minPhs]
    bLengthLong = np.asarray(bLength[bLength >= minPhs], dtype=np.int64)

    if use_noise_regions:
        bStartLongN = np.asarray(bStartN[bLengthN >= minPhsN], dtype=np.int64)
        bLengthLongN = np.asarray(bLengthN[bLengthN >= minPhsN], dtype=np.int64)
    else:
        bStartLongN = np.array([], dtype=np.int64)
        bLengthLongN = np.array([], dtype=np.int64)

    print(f"    Done in {time.perf_counter() - t0:.3f} s")
    print(f"    Bursts kept: {len(bStartLong)}")
    print(f"    Noise regions kept: {len(bStartLongN)}")
    print(f"    First 10 kept burst starts: {bStartLong[:10]}")
    print(f"    First 10 kept burst lengths: {bLengthLong[:10]}")

    # ------------------------------------------------------------------
    # 8. Compute background estimate from noise regions
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    print("\n[8/9] Computing background estimate from noise regions...")

    BackN = 0
    BackT_ticks = 0

    if use_noise_regions:
        for i in range(len(bStartLongN)):
            if i % 1000 == 0 and i > 0:
                print(f"    Processed {i} / {len(bStartLongN)} noise regions...")

            s = bStartLongN[i]
            L = bLengthLongN[i]

            if L > 0:
                # Same logic as old code but without Photons array
                first_tick = macro_times_raw[s]
                last_tick = macro_times_raw[s + L - 1]
                BackT_ticks += (last_tick - first_tick)
                BackN += L

    BackT_ns = BackT_ticks * resolution_s * 1e9
    BI = BackN / BackT_ns * 1e6 if BackT_ns != 0 else 0.0

    print(f"    Done in {time.perf_counter() - t0:.3f} s")
    print(f"    BackN: {BackN}")
    print(f"    BackT (ns): {BackT_ns}")
    print(f"    BI: {BI}")

    # ------------------------------------------------------------------
    # 9. Compute burst intensities and durations
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    print("\n[9/9] Computing burst intensities and durations...")

    NI = bLengthLong.astype(np.float64)
    TBurst = np.zeros(len(bStartLong), dtype=np.float64)

    # Burst start time in ns
    burst_start_ns = macro_times_raw[bStartLong].astype(np.float64) * resolution_s * 1e9

    for i in range(len(bStartLong)):
        if i % 1000 == 0 and i > 0:
            print(f"    Processed {i} / {len(bStartLong)} bursts...")

        s = int(bStartLong[i])
        L = int(bLengthLong[i])

        # Match the old logic without building interPhT:
        # start_idx = max(s - 1, 0)
        # end_idx   = s + L - 1
        # TBurst = sum(interPhT[start_idx:end_idx]) * 1e-6  (when interPhT in ns)
        if s > 0:
            dur_ticks = macro_times_raw[s + L - 1] - macro_times_raw[s - 1]
        else:
            # Equivalent edge handling for first burst
            dur_ticks = macro_times_raw[L - 1] - macro_times_raw[0]

        TBurst[i] = dur_ticks * resolution_s * 1e3  # ms

    df2 = pd.DataFrame({
        "Burst timestart": burst_start_ns,
        "Burst intensity": NI,
        "Burst duration": TBurst,
    })

    # Debug-only photons DataFrame
    debug_n = min(debug_photons_n, len(macro_times_raw))
    if debug_n > 0:
        df1 = pd.DataFrame({
            "channel": channels[:debug_n],
            "photons": macro_times_raw[:debug_n].astype(np.float64) * resolution_s * 1e9,
        })
    else:
        df1 = pd.DataFrame(columns=["channel", "photons"])

    df1.attrs["total_photons_in_file"] = int(len(macro_times_raw))
    df1.attrs["macro_resolution_s"] = resolution_s
    df1.attrs["macro_times_raw_full"] = macro_times_raw
    df1.attrs["channels_full"] = channels

    print(f"    Done in {time.perf_counter() - t0:.3f} s")
    print(f"    debug photons dataframe shape: {df1.shape}")
    print(f"    bursts dataframe shape: {df2.shape}")

    if len(df2) > 0:
        print("    First 5 bursts:")
        print(df2.head())
    else:
        print("    No bursts after filtering.")

    print(f"\n=== get_bursts: END ({time.perf_counter() - total_t0:.3f} s) ===")
    return df1, df2