import json
import numpy as np
import pandas as pd
import tttrlib

from scipy.ndimage import uniform_filter

from leefilter_new import leeFilter1D_matlab


def get_macro_resolution_s(tttr_obj):
    header = json.loads(tttr_obj.header.json)

    if "MeasDesc_GlobalResolution" in header:
        return float(header["MeasDesc_GlobalResolution"])

    for tag in header.get("tags", []):
        if tag.get("name") == "MeasDesc_GlobalResolution":
            return float(tag["value"])

    raise KeyError("MeasDesc_GlobalResolution not found in header.")


def _lee_filter_add_with_global_variance(I, window_size, overall_variance):
    I = np.asarray(I, dtype=np.float64)
    mean_I = uniform_filter(I, window_size)
    sqr_mean_I = uniform_filter(I ** 2, window_size)
    var_I = sqr_mean_I - mean_I ** 2

    weight_I = var_I / (var_I + overall_variance)
    output_I = mean_I + weight_I * (I - mean_I)
    return output_I


def _combine_mean_variance(n_total, mean_total, m2_total, x):
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

        if self.open_run and not mask[0]:
            self._close_open_run()

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
        if self.open_run and starts[0] == 0:
            first_end = ends[0]
            self.open_length += first_end

            if first_end < n:
                self._close_open_run()
            else:
                return

            run_idx = 1

        for s, e in zip(starts[run_idx:], ends[run_idx:]):
            length = int(e - s)
            abs_start = int(absolute_offset + s)

            if e == n:
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


def _normalize_burst_channels_setting(setting_value):
    """
    Convert user-facing burst channel selection to routing-channel numbers.

    Supported values
    ----------------
    None, 'sum', 'both' -> all channels
    'ch1', 1            -> routing channel 0
    'ch2', 2            -> routing channel 1
    [0], [1], [0, 1]    -> direct routing-channel selection
    """
    if setting_value is None:
        return None

    if isinstance(setting_value, str):
        value = setting_value.strip().lower()
        if value in {"sum", "both", "all"}:
            return None
        if value in {"ch1", "channel1", "channel_1", "1"}:
            return np.array([0], dtype=np.int64)
        if value in {"ch2", "channel2", "channel_2", "2"}:
            return np.array([1], dtype=np.int64)
        raise ValueError(
            "Invalid burst_channel_mode. Use 'sum', 'ch1', or 'ch2'."
        )

    if np.isscalar(setting_value):
        if int(setting_value) == 1:
            return np.array([0], dtype=np.int64)
        if int(setting_value) == 2:
            return np.array([1], dtype=np.int64)

    allowed = np.asarray(setting_value, dtype=np.int64).ravel()
    return allowed



def _select_events(tttr, user_setting):
    tttr_mode = str(user_setting.get("tttr_mode", "T2")).upper()

    channels = np.asarray(tttr.routing_channels)
    macro_times_raw = np.asarray(tttr.macro_times)

    if tttr_mode == "PIE_T3":
        pie_gate = user_setting.get("pie_microtime_gate", None)
        if pie_gate is not None:
            raise NotImplementedError(
                "PIE-T3 microtime gating is not implemented yet. "
                "Leave pie_microtime_gate=None for now."
            )

    selected_channels = _normalize_burst_channels_setting(
        user_setting.get("burst_channel_mode", None)
    )

    allowed_channels = user_setting.get("allowed_routing_channels", None)
    if allowed_channels is not None:
        allowed_channels = np.asarray(allowed_channels, dtype=np.int64).ravel()
        if selected_channels is None:
            selected_channels = allowed_channels
        else:
            selected_channels = np.intersect1d(selected_channels, allowed_channels)

    if selected_channels is not None:
        mask = np.isin(channels, selected_channels)
        channels = channels[mask]
        macro_times_raw = macro_times_raw[mask]

    return channels, macro_times_raw



def _compute_overall_variance_of_diffs(macro_times_raw, chunk_size):
    n_events = len(macro_times_raw)
    n_diff = n_events - 1

    n_total = 0
    mean_total = 0.0
    m2_total = 0.0

    for start in range(0, n_diff, chunk_size):
        end = min(start + chunk_size, n_diff)
        diffs = np.diff(macro_times_raw[start:end + 1]).astype(np.float64)
        n_total, mean_total, m2_total = _combine_mean_variance(
            n_total, mean_total, m2_total, diffs
        )

    if n_total < 2:
        return 0.0

    return m2_total / n_total



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
    n_events = len(macro_times_raw)
    n_diff = n_events - 1

    tick_s = resolution_s
    lower_bound_ticks = 0.4e-9 / tick_s
    signal_upper_ticks = signal_upper_ms * 1e-3 / tick_s
    noise_lower_ticks = noise_lower_ms * 1e-3 / tick_s

    halo = max(2, int(setLeeFilter) + 2)

    signal_runs = BooleanRunCollector()
    noise_runs = BooleanRunCollector() if use_noise_regions else None

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

    bStart, bLength = signal_runs.finalize()

    if use_noise_regions:
        bStartN, bLengthN = noise_runs.finalize()
    else:
        bStartN = np.array([], dtype=np.int64)
        bLengthN = np.array([], dtype=np.int64)

    return bStart, bLength, bStartN, bLengthN



def get_bursts(ptufilename, user_setting=None):
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
        "burst_channel_mode": "sum",
        "pie_microtime_gate": None,
        "diff_chunk_size": 5_000_000,
        "debug_photons_n": 0,
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

    tttr = tttrlib.TTTR(ptufilename)
    channels, macro_times_raw = _select_events(tttr, user_setting)
    resolution_s = get_macro_resolution_s(tttr)

    if len(macro_times_raw) == 0:
        df1 = pd.DataFrame(columns=["channel", "photons"])
        df1.attrs["total_photons_in_file"] = 0
        df1.attrs["macro_resolution_s"] = resolution_s
        df1.attrs["macro_times_raw_full"] = np.array([], dtype=np.int64)
        df1.attrs["channels_full"] = np.array([], dtype=np.int64)
        df2 = pd.DataFrame(columns=["Burst timestart", "Burst intensity", "Burst duration"])
        return df1, df2

    if len(macro_times_raw) == 1:
        debug_n = min(debug_photons_n, len(macro_times_raw))
        df1 = pd.DataFrame({
            "channel": channels[:debug_n],
            "photons": macro_times_raw[:debug_n].astype(np.float64) * resolution_s * 1e9,
        })
        df1.attrs["total_photons_in_file"] = int(len(macro_times_raw))
        df1.attrs["macro_resolution_s"] = resolution_s
        df1.attrs["macro_times_raw_full"] = macro_times_raw
        df1.attrs["channels_full"] = channels
        df2 = pd.DataFrame(columns=["Burst timestart", "Burst intensity", "Burst duration"])
        return df1, df2

    overall_variance = _compute_overall_variance_of_diffs(macro_times_raw, diff_chunk_size)

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

    bStartLong = bStart[bLength >= minPhs]
    bLengthLong = np.asarray(bLength[bLength >= minPhs], dtype=np.int64)

    if use_noise_regions:
        bStartLongN = np.asarray(bStartN[bLengthN >= minPhsN], dtype=np.int64)
        bLengthLongN = np.asarray(bLengthN[bLengthN >= minPhsN], dtype=np.int64)
    else:
        bStartLongN = np.array([], dtype=np.int64)
        bLengthLongN = np.array([], dtype=np.int64)

    BackN = 0
    BackT_ticks = 0

    if use_noise_regions:
        for i in range(len(bStartLongN)):
            s = bStartLongN[i]
            L = bLengthLongN[i]
            if L > 0:
                first_tick = macro_times_raw[s]
                last_tick = macro_times_raw[s + L - 1]
                BackT_ticks += (last_tick - first_tick)
                BackN += L

    BackT_ns = BackT_ticks * resolution_s * 1e9
    BI = BackN / BackT_ns * 1e6 if BackT_ns != 0 else 0.0

    NI = bLengthLong.astype(np.float64)
    TBurst = np.zeros(len(bStartLong), dtype=np.float64)
    burst_start_ns = macro_times_raw[bStartLong].astype(np.float64) * resolution_s * 1e9

    for i in range(len(bStartLong)):
        s = int(bStartLong[i])
        L = int(bLengthLong[i])
        if s > 0:
            dur_ticks = macro_times_raw[s + L - 1] - macro_times_raw[s - 1]
        else:
            dur_ticks = macro_times_raw[L - 1] - macro_times_raw[0]
        TBurst[i] = dur_ticks * resolution_s * 1e3

    df2 = pd.DataFrame({
        "Burst timestart": burst_start_ns,
        "Burst intensity": NI,
        "Burst duration": TBurst,
    })

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
    df1.attrs["background_intensity"] = BI

    return df1, df2
