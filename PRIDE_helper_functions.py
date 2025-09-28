"""Utility helpers for PRIDE-related processing."""

from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import pandas as pd
import numpy as np
import xarray as xr

__all__ = [
    "load_csv",
    "load_kin",
    "mjd_sod_to_gps_datetime",
    "receiver_clock_file",
    "apply_corrections_ionosphere",
    "attach_geometric_residuals",
    "plot_carrier_residuals",
    "plot_code_residuals",
    "plot_kin_xyz",
    "plot_geometric_ranges",
    "plot_satellite_ecef_errors",
]


def load_csv(
    file_path: Union[str, Path],
    delimiter: str = ",",
    encoding: str = "utf-8",
    dtype: Optional[Union[str, Mapping[str, str]]] = None,
    usecols: Optional[Union[Sequence[str], Sequence[int]]] = None,
    na_values: Optional[Union[str, Sequence[str]]] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """Load a CSV where the first line contains column names into a DataFrame.

    Parameters
    ----------
    file_path:
        Path to the CSV file.
    delimiter:
        Field separator used in the file. Defaults to ",".
    encoding:
        Text encoding of the file. Defaults to "utf-8".
    dtype:
        Optional dtype specification forwarded to pandas.
    usecols:
        Optional subset of columns to read (names or indices).
    na_values:
        Additional strings to recognize as NA/NaN.
    nrows:
        If provided, only read the first ``nrows`` rows.

    Returns
    -------
    pandas.DataFrame
        The loaded DataFrame with columns taken from the first line of the file.
    """

    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    dataframe = pd.read_csv(
        path,
        header=0,  # first line contains column names
        sep=delimiter,
        encoding=encoding,
        dtype=dtype,
        usecols=usecols,
        na_values=na_values,
        nrows=nrows,
        low_memory=False,
    )

    # Convert an ISO8601 time column like '2025-06-20T17:50:08.99Z' to naive gps_datetime
    # Always attempt to parse from common column names
    col = None
    for candidate in ("gps_datetime", "time", "gps_time", "datetime"):
        if candidate in dataframe.columns:
            col = candidate
            break
    if col is None:
        # As a fallback, try first object column
        obj_cols = [c for c in dataframe.columns if dataframe[c].dtype == object]
        if obj_cols:
            col = obj_cols[0]

    if col is not None:
        dt = pd.to_datetime(dataframe[col], utc=True, errors="coerce")
        dataframe["gps_datetime"] = dt.dt.tz_localize(None)

    return dataframe



def load_kin(
    file_path: Union[str, Path],
    add_datetime: bool = False,
    tz_utc: bool = True,
) -> pd.DataFrame:
    """Load a PRIDE kinematic ``kin_*`` file into a DataFrame.

    The function parses the header until "END OF HEADER" and then reads the
    epoch-by-epoch kinematic block. The output columns are:

    - Mjd
    - Sod
    - X, Y, Z (meters)
    - Latitude, Longitude (deg), Height (meters)
    - Nsat_all, Nsat_G, Nsat_R, Nsat_E, Nsat_C2, Nsat_C3, Nsat_J (satellite counts)
    - PDOP

    Parameters
    ----------
    file_path:
        Path to the PRIDE ``kin_*`` file.
    add_datetime:
        If True, adds a ``datetime`` column computed from MJD and SOD (UTC by default).
    tz_utc:
        If True and ``add_datetime`` is enabled, the ``datetime`` column will be timezone-aware UTC.

    Returns
    -------
    pandas.DataFrame
        DataFrame with one row per epoch.
    """

    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"KIN file not found: {path}")

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()

    # Find the end of header marker
    header_end_index = None
    for index, line_text in enumerate(lines):
        if "END OF HEADER" in line_text:
            header_end_index = index
            break

    if header_end_index is None or header_end_index + 2 >= len(lines):
        raise ValueError("Unexpected KIN file format: 'END OF HEADER' not found or no data section present")

    # Data begins after the column header line which follows 'END OF HEADER'
    data_start_index = header_end_index + 2

    mjd_values = []
    sod_values = []
    x_values = []
    y_values = []
    z_values = []
    latitude_values = []
    longitude_values = []
    height_values = []
    nsat_all_values = []
    nsat_g_values = []
    nsat_r_values = []
    nsat_e_values = []
    nsat_c2_values = []
    nsat_c3_values = []
    nsat_j_values = []
    pdop_values = []

    for raw_line in lines[data_start_index:]:
        stripped = raw_line.strip()
        if not stripped:
            continue

        # Tokenize and drop the '*' separator that appears after SOD
        tokens = [tok for tok in stripped.split() if tok != "*"]

        # Expect 16 numeric tokens: Mjd, Sod, X, Y, Z, Lat, Lon, H, 7x Nsat, PDOP
        # Skip anything that doesn't look like a data line
        if len(tokens) < 16:
            continue

        try:
            mjd = int(float(tokens[0]))
            sod = float(tokens[1])
            x = float(tokens[2])
            y = float(tokens[3])
            z = float(tokens[4])
            lat = float(tokens[5])
            lon = float(tokens[6])
            hgt = float(tokens[7])

            # Nsat counts: 7 integers
            nsat_numbers = tokens[8:15]
            if len(nsat_numbers) != 7:
                # If the line has extra tokens, try to realign from the tail
                nsat_numbers = tokens[-8:-1]
                if len(nsat_numbers) != 7:
                    continue

            ns_all = int(float(nsat_numbers[0]))
            ns_g = int(float(nsat_numbers[1]))
            ns_r = int(float(nsat_numbers[2]))
            ns_e = int(float(nsat_numbers[3]))
            ns_c2 = int(float(nsat_numbers[4]))
            ns_c3 = int(float(nsat_numbers[5]))
            ns_j = int(float(nsat_numbers[6]))

            pdop = float(tokens[-1])
        except ValueError:
            # Non-numeric line (e.g., comment)
            continue

        mjd_values.append(mjd)
        sod_values.append(sod)
        x_values.append(x)
        y_values.append(y)
        z_values.append(z)
        latitude_values.append(lat)
        longitude_values.append(lon)
        height_values.append(hgt)
        nsat_all_values.append(ns_all)
        nsat_g_values.append(ns_g)
        nsat_r_values.append(ns_r)
        nsat_e_values.append(ns_e)
        nsat_c2_values.append(ns_c2)
        nsat_c3_values.append(ns_c3)
        nsat_j_values.append(ns_j)
        pdop_values.append(pdop)

    dataframe = pd.DataFrame(
        {
            "Mjd": mjd_values,
            "Sod": sod_values,
            "X": x_values,
            "Y": y_values,
            "Z": z_values,
            "Latitude": latitude_values,
            "Longitude": longitude_values,
            "Height": height_values,
            "Nsat_all": nsat_all_values,
            "Nsat_G": nsat_g_values,
            "Nsat_R": nsat_r_values,
            "Nsat_E": nsat_e_values,
            "Nsat_C2": nsat_c2_values,
            "Nsat_C3": nsat_c3_values,
            "Nsat_J": nsat_j_values,
            "PDOP": pdop_values,
        }
    )

    if add_datetime and not dataframe.empty:
        # MJD epoch is 1858-11-17 00:00:00 (for reference/UTC-like timeline)
        if tz_utc:
            base = pd.Timestamp("1858-11-17T00:00:00Z")
        else:
            base = pd.Timestamp("1858-11-17 00:00:00")
        dataframe["datetime"] = (
            base + pd.to_timedelta(dataframe["Mjd"], unit="D") + pd.to_timedelta(dataframe["Sod"], unit="s")
        )

        # Additionally provide GPS timescale timestamps (naive, GPS epoch based)
        dataframe["gps_datetime"] = mjd_sod_to_gps_datetime(
            dataframe["Mjd"], dataframe["Sod"]
        )

    return dataframe


def mjd_sod_to_gps_datetime(
    mjd: Union[pd.Series, Sequence[float], float, int],
    sod: Union[pd.Series, Sequence[float], float],
) -> Union[pd.Series, pd.Timestamp]:
    """Convert MJD and SOD to a pandas Timestamp on the GPS timescale.

    GPS time is continuous (no leap seconds) and starts at 1980-01-06 00:00:00.
    The returned timestamps are naive (no timezone), representing GPS time.

    Parameters
    ----------
    mjd:
        Modified Julian Day(s).
    sod:
        Seconds of day corresponding to ``mjd``.

    Returns
    -------
    pandas.Series or pandas.Timestamp
        Timestamps in GPS timescale. If inputs are array-like, returns a Series;
        if scalars, returns a single Timestamp.
    """

    gps_epoch_mjd = 44244  # 1980-01-06 00:00:00

    # Normalize inputs to numpy arrays for vectorized math, keep track of scalar case
    mjd_array = np.asarray(mjd)
    sod_array = np.asarray(sod)
    is_scalar = mjd_array.ndim == 0 and sod_array.ndim == 0

    # Convert days since GPS epoch to seconds, then add SOD
    gps_seconds = (mjd_array - gps_epoch_mjd) * 86400.0 + sod_array

    # Build timestamps relative to GPS epoch; keep them naive (GPS timescale)
    base = pd.Timestamp("1980-01-06 00:00:00")
    dt = base + pd.to_timedelta(gps_seconds, unit="s")

    if is_scalar:
        # pd.to_timedelta with scalar returns Timedelta; addition yields Timestamp
        return pd.Timestamp(dt)
    else:
        # Ensure a pandas Series with datetime64[ns] dtype
        return pd.Series(dt)


def receiver_clock_file(
    file_path: Union[str, Path],
    add_datetime: bool = False,
) -> pd.DataFrame:
    """Load a PRIDE receiver clock ``rck_*`` file into a DataFrame.

    Expected output columns:
    - Year, Mon, Day, Hour, Min, Sec (as in file, GPS timescale)
    - RCK_GPS, RCK_GLONASS, RCK_Galileo, RCK_BDS2, RCK_BDS3, RCK_QZSS (meters)

    If ``add_datetime`` is True, also adds a naive ``gps_datetime`` column
    constructed from Year/Mon/Day Hour:Min:Sec (GPS timescale, no leap seconds).
    """

    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Receiver clock file not found: {path}")

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()

    # Find end of header
    header_end_index = None
    for index, line_text in enumerate(lines):
        if "END OF HEADER" in line_text:
            header_end_index = index
            break

    if header_end_index is None or header_end_index + 2 >= len(lines):
        raise ValueError("Unexpected RCK file format: 'END OF HEADER' not found or no data section present")

    # Data starts two lines after END OF HEADER (skip the column header that begins with '*')
    data_start_index = header_end_index + 2

    year_values = []
    mon_values = []
    day_values = []
    hour_values = []
    min_values = []
    sec_values = []
    rck_gps_values = []
    rck_glo_values = []
    rck_gal_values = []
    rck_bds2_values = []
    rck_bds3_values = []
    rck_qzss_values = []

    for raw_line in lines[data_start_index:]:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("*"):
            continue

        tokens = stripped.split()
        if len(tokens) < 12:
            # Not a data line
            continue

        try:
            yr = int(float(tokens[0]))
            mo = int(float(tokens[1]))
            dy = int(float(tokens[2]))
            hr = int(float(tokens[3]))
            mi = int(float(tokens[4]))
            sc = float(tokens[5])

            # Take the last six tokens as the RCK values to be robust to extra fields
            rcks = tokens[-6:]
            r_gps = float(rcks[0])
            r_glo = float(rcks[1])
            r_gal = float(rcks[2])
            r_bds2 = float(rcks[3])
            r_bds3 = float(rcks[4])
            r_qzss = float(rcks[5])
        except ValueError:
            continue

        year_values.append(yr)
        mon_values.append(mo)
        day_values.append(dy)
        hour_values.append(hr)
        min_values.append(mi)
        sec_values.append(sc)
        rck_gps_values.append(r_gps)
        rck_glo_values.append(r_glo)
        rck_gal_values.append(r_gal)
        rck_bds2_values.append(r_bds2)
        rck_bds3_values.append(r_bds3)
        rck_qzss_values.append(r_qzss)

    dataframe = pd.DataFrame(
        {
            "Year": year_values,
            "Mon": mon_values,
            "Day": day_values,
            "Hour": hour_values,
            "Min": min_values,
            "Sec": sec_values,
            "RCK_GPS": rck_gps_values,
            "RCK_GLONASS": rck_glo_values,
            "RCK_Galileo": rck_gal_values,
            "RCK_BDS2": rck_bds2_values,
            "RCK_BDS3": rck_bds3_values,
            "RCK_QZSS": rck_qzss_values,
        }
    )

    if add_datetime and not dataframe.empty:
        # Build a naive GPS timescale datetime from Y/M/D and H:M:S
        date_str = (
            dataframe["Year"].astype(int).astype(str).str.zfill(4)
            + "-"
            + dataframe["Mon"].astype(int).astype(str).str.zfill(2)
            + "-"
            + dataframe["Day"].astype(int).astype(str).str.zfill(2)
        )
        day_start = pd.to_datetime(date_str, format="%Y-%m-%d", errors="coerce")
        seconds_of_day = (
            dataframe["Hour"].astype(float) * 3600.0
            + dataframe["Min"].astype(float) * 60.0
            + dataframe["Sec"].astype(float)
        )
        dataframe["gps_datetime"] = day_start + pd.to_timedelta(seconds_of_day, unit="s")

    return dataframe


# ----------------------------------------------------------------------------
# Ionosphere-free pseudorange from georinex Dataset (simple, no clocks)
# ----------------------------------------------------------------------------

C_MPS = 299_792_458.0


def _band_frequency_hz(system_prefix: str, band_digit: int) -> float:
    """Return carrier frequency [Hz] for a system + band number.

    Supported systems: G (GPS), J (QZSS), E (Galileo), C (BeiDou). GLONASS (R)
    is intentionally not supported here.
    """
    s = system_prefix.upper()
    b = int(band_digit)
    if s in ("G", "J"):
        if b == 1:
            return 1575.42e6
        if b == 2:
            return 1227.60e6
        if b == 5:
            return 1176.45e6
    if s == "E":
        if b == 1:
            return 1575.32e6
        if b == 5:
            return 1176.45e6  # E5a
        if b == 7:
            return 1207.14e6  # E5b
        if b == 6:
            return 1278.75e6  # E6
    if s == "C":
        if b == 1:
            return 1575.42e6  # assume B1C (BDS-3)
        if b == 2:
            return 1176.45e6  # B2a
        if b == 7:
            return 1207.14e6  # B2b
        if b == 3:
            return 1268.520e6  # B3
    raise ValueError(f"Unsupported system/band: {system_prefix}{band_digit}")


def _pick_band_var_for_sat(ds: xr.Dataset, sv: str, band_prefix: str) -> Optional[str]:
    """Pick the first observation variable matching band_prefix (e.g. 'C1')
    that contains finite values for the given satellite.
    """
    # Prioritize common codes by suffix order to improve chances
    preferred_suffixes = (
        "C","W","P","Q","X","D","S","L","I","Z","A","B"
    )
    # First pass: variables that start with the exact prefix and a preferred suffix
    vars_band = [v for v in ds.data_vars if v.startswith(band_prefix)]
    # Sort with preferred suffix priority
    def _score(name: str) -> int:
        return preferred_suffixes.index(name[2]) if len(name) >= 3 and name[2] in preferred_suffixes else len(preferred_suffixes)
    vars_band.sort(key=_score)

    for var in vars_band:
        try:
            arr = ds[var].sel(sv=sv)
        except Exception:
            continue
        if np.isfinite(arr.to_numpy()).any():
            return var
    return None


def apply_corrections_ionosphere(
    obs: xr.Dataset,
    rx_clk: Optional[pd.DataFrame] = None,
    sat_clk: Optional[pd.DataFrame] = None,
    clock_tolerance_s: float = 0.25,
    rx_clk_interpolate_missing: bool = True,
    rx_clk_interpolate_to_obs: bool = False,
) -> pd.DataFrame:
    """Compute ionosphere-free code pseudorange from a georinex Dataset.

    Assumptions:
    - Uses code observations (variables named like 'C1C', 'C2W', 'C5Q', ...)
    - Skips GLONASS ('R*') for simplicity
    - Chooses standard band pairs per constellation in this priority:
      G/J: (1,2) then (1,5);  E: (1,5) then (1,7) then (1,6);  C: (1,2) then (1,7) then (1,3)

    Returns a tidy DataFrame with columns: time, sat,
    code_if_m, carrier_if_m, f1_hz, f2_hz,
    code_obs1, code_obs2, carrier_obs1, carrier_obs2,
    and if clocks provided: rx_clk_m, sat_clk_m, code_if_clk_m, carrier_if_clk_m.
    Receiver clock values of exactly 0.0 metres are treated as missing; set
    ``rx_clk_interpolate_missing=True`` to fill gaps via time interpolation, or
    ``rx_clk_interpolate_to_obs=True`` to interpolate the clock directly at
    observation epochs before merging into the result.
    """

    if not isinstance(obs, xr.Dataset):
        raise TypeError("obs must be an xarray.Dataset (e.g., from georinex)")

    times = pd.to_datetime(obs["time"].to_pandas())
    sv_list = [str(s) for s in obs["sv"].values]
    sv_list = [s for s in sv_list if s and s[0] in ("G", "E", "C", "J")]  # skip R, S, etc.

    results: list[pd.DataFrame] = []
    for sv in sv_list:
        sys = sv[0]
        # Band pair preference per system
        if sys in ("G", "J"):
            band_pairs = [("C1", "C2"), ("C1", "C5")]
        elif sys == "E":
            band_pairs = [("C1", "C5"), ("C1", "C7"), ("C1", "C6")]
        elif sys == "C":
            band_pairs = [("C1", "C2"), ("C1", "C7"), ("C1", "C3")]
        else:
            continue

        var1 = var2 = None  # code obs variable names (e.g., C1C, C2W)
        f1 = f2 = None      # carrier frequencies (Hz)
        var1L = var2L = None  # phase obs variable names (e.g., L1C, L2W)
        for b1, b2 in band_pairs:
            v1 = _pick_band_var_for_sat(obs, sv, b1)
            v2 = _pick_band_var_for_sat(obs, sv, b2)
            if v1 and v2:
                var1, var2 = v1, v2
                try:
                    f1 = _band_frequency_hz(sys, int(b1[1]))
                    f2 = _band_frequency_hz(sys, int(b2[1]))
                except ValueError:
                    var1 = var2 = None
                    continue
                # Try to find matching carrier observations for same bands
                v1L = _pick_band_var_for_sat(obs, sv, 'L' + b1[1])
                v2L = _pick_band_var_for_sat(obs, sv, 'L' + b2[1])
                var1L, var2L = v1L, v2L
                break

        if not (var1 and var2 and f1 and f2):
            continue

        P1 = obs[var1].sel(sv=sv).to_numpy()
        P2 = obs[var2].sel(sv=sv).to_numpy()

        # Code ionosphere-free
        mask_c = np.isfinite(P1) & np.isfinite(P2)
        f1_sq = f1 * f1
        f2_sq = f2 * f2
        denom = (f1_sq - f2_sq)
        code_if = np.full_like(P1, np.nan, dtype=float)
        if mask_c.any():
            code_if[mask_c] = (f1_sq * P1[mask_c] - f2_sq * P2[mask_c]) / denom

        # Carrier ionosphere-free (if L* bands exist)
        carrier_if = np.full_like(P1, np.nan, dtype=float)
        if var1L and var2L:
            L1 = obs[var1L].sel(sv=sv).to_numpy()
            L2 = obs[var2L].sel(sv=sv).to_numpy()
            mask_l = np.isfinite(L1) & np.isfinite(L2)
            if mask_l.any():
                lam1 = C_MPS / f1
                lam2 = C_MPS / f2
                L1_m = L1 * lam1
                L2_m = L2 * lam2
                carrier_if[mask_l] = (f1_sq * L1_m[mask_l] - f2_sq * L2_m[mask_l]) / denom

        df = pd.DataFrame({
            "time": times,
            "sat": sv,
            "code_if_m": code_if,
            "carrier_if_m": carrier_if,
            "f1_hz": f1,
            "f2_hz": f2,
            "code_obs1": var1,
            "code_obs2": var2,
            "carrier_obs1": var1L,
            "carrier_obs2": var2L,
        })
        # keep epochs where at least one IF product is available
        df = df.dropna(subset=["code_if_m", "carrier_if_m"], how="all")
        if not df.empty:
            results.append(df)

    if not results:
        return pd.DataFrame(columns=[
            "time","sat","code_if_m","carrier_if_m","f1_hz","f2_hz",
            "code_obs1","code_obs2","carrier_obs1","carrier_obs2"
        ])

    out = pd.concat(results, ignore_index=True)

    # ------------------------
    # Receiver clock correction (PRIDE rck_* format only)
    # ------------------------
    if rx_clk is not None and not rx_clk.empty:
        # Expect columns: ['Year','Mon','Day','Hour','Min','Sec','RCK_GPS','RCK_GLONASS','RCK_Galileo','RCK_BDS2','RCK_BDS3','RCK_QZSS','gps_datetime']
        rck = rx_clk.copy()
        if "gps_datetime" not in rck.columns:
            raise ValueError("rx_clk must contain 'gps_datetime' column (GPS timescale)")

        # Build combined BDS column once (prefer BDS3 over BDS2 when available)
        if "RCK_BDS3" in rck.columns and "RCK_BDS2" in rck.columns:
            rck["RCK_BDS"] = rck["RCK_BDS3"].where(rck["RCK_BDS3"].notna(), rck["RCK_BDS2"])
        elif "RCK_BDS3" in rck.columns:
            rck["RCK_BDS"] = rck["RCK_BDS3"]
        elif "RCK_BDS2" in rck.columns:
            rck["RCK_BDS"] = rck["RCK_BDS2"]
        else:
            rck["RCK_BDS"] = np.nan

        rck = rck.sort_values("gps_datetime")
        rck["gps_datetime"] = pd.to_datetime(rck["gps_datetime"], errors="coerce")
        out["time"] = pd.to_datetime(out["time"], errors="coerce")
        out["rx_clk_m"] = 0.0
        out["rx_clk_dt_s"] = np.nan

        mapping = {
            "G": "RCK_GPS",
            "E": "RCK_Galileo",
            "C": "RCK_BDS",
            "J": "RCK_QZSS",
        }
        for prefix, col in mapping.items():
            left = out[out["sat"].str.startswith(prefix)].sort_values("time").copy()
            if left.empty or col not in rck.columns:
                continue
            right = (
                rck[["gps_datetime", col]]
                .rename(columns={"gps_datetime": "clk_time", col: "rx_clk_val"})
                .copy()
            )
            right["clk_time"] = pd.to_datetime(right["clk_time"], errors="coerce")
            right["rx_clk_val"] = pd.to_numeric(right["rx_clk_val"], errors="coerce")
            right.loc[right["rx_clk_val"] == 0.0, "rx_clk_val"] = np.nan
            right = right.dropna(subset=["clk_time"])
            if right.empty:
                continue
            right_sorted = right.sort_values("clk_time")
            if rx_clk_interpolate_missing and right_sorted["rx_clk_val"].notna().sum() >= 2:
                right_interp = right_sorted.set_index("clk_time")
                right_interp["rx_clk_val"] = right_interp["rx_clk_val"].interpolate(
                    method="time", limit_direction="both"
                )
                right_sorted = right_interp.reset_index()
            valid = right_sorted["rx_clk_val"].notna()
            if valid.sum() == 0:
                continue
            left = left.assign(_idx=left.index)
            if rx_clk_interpolate_to_obs:
                obs_times = pd.to_datetime(left["time"], errors="coerce")
                obs_sec = obs_times.astype("int64").to_numpy(dtype=float) / 1e9
                clk_sec = right_sorted.loc[valid, "clk_time"].astype("int64").to_numpy(dtype=float) / 1e9
                clk_vals = right_sorted.loc[valid, "rx_clk_val"].to_numpy(dtype=float)
                if clk_sec.size < 2:
                    continue
                interp_vals = np.interp(obs_sec, clk_sec, clk_vals, left=np.nan, right=np.nan)
                insert_idx = np.searchsorted(clk_sec, obs_sec)
                prev_idx = np.clip(insert_idx - 1, 0, clk_sec.size - 1)
                next_idx = np.clip(insert_idx, 0, clk_sec.size - 1)
                diff_prev = np.abs(obs_sec - clk_sec[prev_idx])
                diff_next = np.abs(obs_sec - clk_sec[next_idx])
                min_diff = np.minimum(diff_prev, diff_next)
                mask_obs = np.isfinite(interp_vals) & (min_diff <= clock_tolerance_s)
                fill_vals = np.full(obs_sec.shape, np.nan, dtype=float)
                dt_out = np.full(obs_sec.shape, np.nan, dtype=float)
                fill_vals[mask_obs] = interp_vals[mask_obs]
                dt_out[mask_obs] = 0.0
                out.loc[left["_idx"], "rx_clk_m"] = fill_vals
                out.loc[left["_idx"], "rx_clk_dt_s"] = dt_out
                continue
            merged = pd.merge_asof(
                left,
                right_sorted,
                left_on="time",
                right_on="clk_time",
                direction="nearest",
                tolerance=pd.to_timedelta(clock_tolerance_s, unit="s"),
            )
            idx = merged["_idx"].to_numpy()
            dt_s = (merged["time"] - merged["clk_time"]).abs().dt.total_seconds()
            vals = merged["rx_clk_val"].to_numpy(dtype=float)
            dt_arr = dt_s.to_numpy(dtype=float)
            fill_vals = np.full(vals.shape, np.nan, dtype=float)
            dt_out = np.full(vals.shape, np.nan, dtype=float)
            mask = np.isfinite(vals) & np.isfinite(dt_arr)
            fill_vals[mask] = vals[mask]
            dt_out[mask] = dt_arr[mask]
            out.loc[idx, "rx_clk_m"] = fill_vals
            out.loc[idx, "rx_clk_dt_s"] = dt_out

    # -------------------------
    # Satellite clock correction with interpolation per PRN
    # Accepts sat_clk columns: ['sat','gps_time','sat_clk_m'] or ['sat','gps_time','sat_clk_ns']
    # -------------------------
    if sat_clk is not None and not sat_clk.empty:
        sck = sat_clk.copy()
        if not {"sat", "gps_time"}.issubset(sck.columns):
            raise ValueError("sat_clk must contain columns: 'sat','gps_time' and one of 'sat_clk_m' or 'sat_clk_ns'")
        if "sat_clk_m" in sck.columns:
            val_col = "sat_clk_m"
        elif "sat_clk_ns" in sck.columns:
            # Convert to metres
            sck["sat_clk_m"] = C_MPS * sck["sat_clk_ns"].astype(float) * 1e-9
            val_col = "sat_clk_m"
        else:
            raise ValueError("sat_clk must contain either 'sat_clk_m' or 'sat_clk_ns'")

        sck["sat"] = sck["sat"].astype(str)
        sck["gps_time"] = pd.to_datetime(sck["gps_time"], errors="coerce")
        sck = sck.dropna(subset=["gps_time", val_col])

        out["sat"] = out["sat"].astype(str)
        out["time"] = pd.to_datetime(out["time"], errors="coerce")
        out["sat_clk_m"] = np.nan
        out["sat_clk_dt_s"] = np.nan  # not meaningful for interpolation; kept for compatibility

        # Interpolate per PRN using numeric seconds
        for prn, left_grp in out.groupby("sat", sort=False):
            right_grp = sck[sck["sat"] == prn]
            if right_grp.empty:
                continue
            # Need at least two points to interpolate
            if right_grp.shape[0] < 2:
                continue
            x = left_grp["time"].astype("int64").to_numpy(dtype="float64") / 1e9
            xp = right_grp["gps_time"].astype("int64").to_numpy(dtype="float64") / 1e9
            fp = right_grp[val_col].to_numpy(dtype="float64")
            # Ensure xp is strictly increasing for np.interp
            order = np.argsort(xp)
            xp = xp[order]
            fp = fp[order]
            interp_vals = np.interp(x, xp, fp, left=np.nan, right=np.nan)
            out.loc[left_grp.index, "sat_clk_m"] = interp_vals
    else:
        out["sat_clk_m"] = 0.0

    # Final corrected range (if clocks available). Missing clocks are treated as 0.
    if "rx_clk_m" not in out.columns:
        out["rx_clk_m"] = 0.0
    if "sat_clk_m" not in out.columns:
        out["sat_clk_m"] = 0.0
    out["code_if_clk_m"] = out["code_if_m"] - out["rx_clk_m"] + out["sat_clk_m"]
    if "carrier_if_m" in out.columns:
        out["carrier_if_clk_m"] = out["carrier_if_m"] - out["rx_clk_m"] + out["sat_clk_m"]
    else:
        out["carrier_if_clk_m"] = np.nan

    return out




def geometric_ranges(obs_sv, ecef_df, pos_df):
    """
    obs_sv: xarray DataArray or list/array of PRNs (e.g., obs.sv)
    ecef_df: DataFrame with ['Mjd','Sod','gps_datetime','sat','ECEF_x','ECEF_y','ECEF_z']
    pos_df:  DataFrame with ['Mjd','Sod','X','Y','Z','Latitude','Longitude'] (from kin)
    """
    sv_set = set((obs_sv.values if hasattr(obs_sv, "values") else obs_sv))

    # Keep only sats of interest, then align on Mjd+Sod
    sat_ecef = ecef_df.loc[ecef_df["sat"].isin(sv_set),
                           ["Mjd","Sod","gps_datetime","sat","ECEF_x","ECEF_y","ECEF_z"]]

    # Require Latitude/Longitude for elevation/azimuth
    merged = sat_ecef.merge(pos_df[["Mjd","Sod","X","Y","Z","Latitude","Longitude"]],
                            on=["Mjd","Sod"], how="inner", validate="many_to_one")

    dist = np.sqrt((merged["ECEF_x"] - merged["X"])**2 +
                   (merged["ECEF_y"] - merged["Y"])**2 +
                   (merged["ECEF_z"] - merged["Z"])**2)

    # Topocentric ENU at receiver
    dx = (merged["ECEF_x"] - merged["X"]).to_numpy(dtype=float)
    dy = (merged["ECEF_y"] - merged["Y"]).to_numpy(dtype=float)
    dz = (merged["ECEF_z"] - merged["Z"]).to_numpy(dtype=float)
    lat_rad = np.deg2rad(merged["Latitude"].to_numpy(dtype=float))
    lon_rad = np.deg2rad(merged["Longitude"].to_numpy(dtype=float))

    sin_lat = np.sin(lat_rad); cos_lat = np.cos(lat_rad)
    sin_lon = np.sin(lon_rad); cos_lon = np.cos(lon_rad)

    e_comp = -sin_lon * dx + cos_lon * dy
    n_comp = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u_comp =  cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz

    # Elevation and azimuth
    rng = dist.to_numpy(dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):
        elev_rad = np.arcsin(np.clip(u_comp / rng, -1.0, 1.0))
        azim_rad = np.arctan2(e_comp, n_comp)  # from North, east-positive
    elevation_deg = np.degrees(elev_rad)
    azimuth_deg = (np.degrees(azim_rad) + 360.0) % 360.0

    out = merged[["Mjd","Sod","gps_datetime","sat"]].copy()
    out["geometric_range_m"] = dist
    out["elevation_deg"] = elevation_deg
    out["azimuth_deg"] = azimuth_deg
    return out


def attach_geometric_residuals(
    pr_if_df: pd.DataFrame,
    rng_df: pd.DataFrame,
    tolerance_s: float = 0.25,
) -> pd.DataFrame:
    """Attach geometric range and residuals to IF results without mismatches.

    Parameters
    ----------
    pr_if_df : pd.DataFrame
        DataFrame from apply_corrections_ionosphere with columns at least
        ['time','sat'] and one or more of ['code_if_m','code_if_clk_m',
        'carrier_if_m','carrier_if_clk_m'].
    rng_df : pd.DataFrame
        Output from geometric_ranges with columns ['gps_datetime','sat',
        'geometric_range_m'].
    tolerance_s : float, default 0.25
        Maximum allowed time difference for a match, in seconds. Rows without
        a match within tolerance will get NaN geometric range and residuals.

    Returns
    -------
    pd.DataFrame
        A copy of pr_if_df with added columns: 'geom_range_m',
        'code_minus_geom_m', 'code_clk_minus_geom_m',
        'carrier_minus_geom_m', 'carrier_clk_minus_geom_m'.
    """

    if pr_if_df.empty:
        return pr_if_df.copy()

    left = pr_if_df.copy()
    left["time"] = pd.to_datetime(left["time"], errors="coerce")
    left["sat"] = left["sat"].astype(str)

    right = rng_df[["gps_datetime", "sat", "geometric_range_m", "elevation_deg", "azimuth_deg"]].copy()
    right["gps_datetime"] = pd.to_datetime(right["gps_datetime"], errors="coerce")
    right["sat"] = right["sat"].astype(str)

    # Prepare output columns
    left["geom_range_m"] = np.nan
    # Add elevation/azimuth columns if available in rng_df
    add_elev = "elevation_deg" in rng_df.columns
    add_azim = "azimuth_deg" in rng_df.columns
    if add_elev:
        left["elevation_deg"] = np.nan
    if add_azim:
        left["azimuth_deg"] = np.nan
    if "code_if_m" in left.columns:
        left["code_minus_geom_m"] = np.nan
    if "code_if_clk_m" in left.columns:
        left["code_clk_minus_geom_m"] = np.nan
    if "carrier_if_m" in left.columns:
        left["carrier_minus_geom_m"] = np.nan
    if "carrier_if_clk_m" in left.columns:
        left["carrier_clk_minus_geom_m"] = np.nan

    tol = pd.to_timedelta(tolerance_s, unit="s")

    parts: list[pd.DataFrame] = []
    for prn, grp in left.groupby("sat", sort=False):
        r = right[right["sat"] == prn]
        if r.empty:
            parts.append(grp)
            continue
        l_sorted = grp.sort_values("time").assign(_idx=grp.index)
        # Select columns to merge from rng (always range, optionally elev/az)
        r_sorted = r.sort_values("gps_datetime").rename(columns={"gps_datetime": "clk_time"})
        right_cols = ["clk_time", "geometric_range_m"]
        if add_elev and "elevation_deg" in r_sorted.columns:
            right_cols.append("elevation_deg")
        if add_azim and "azimuth_deg" in r_sorted.columns:
            right_cols.append("azimuth_deg")
        merged = pd.merge_asof(
            l_sorted,
            r_sorted[right_cols],
            left_on="time",
            right_on="clk_time",
            direction="nearest",
            tolerance=tol,
        )
        idx = merged["_idx"].to_numpy()
        geom = merged["geometric_range_m"].to_numpy()
        left.loc[idx, "geom_range_m"] = geom

        # Transfer elevation/azimuth if present
        if add_elev and "elevation_deg" in merged.columns:
            left.loc[idx, "elevation_deg"] = merged["elevation_deg"].to_numpy()
        if add_azim and "azimuth_deg" in merged.columns:
            left.loc[idx, "azimuth_deg"] = merged["azimuth_deg"].to_numpy()

        # Residuals
        if "code_if_clk_m" in left.columns:
            left.loc[idx, "code_clk_minus_geom_m"] = left.loc[idx, "code_if_clk_m"] - geom
        if "carrier_if_clk_m" in left.columns:
            left.loc[idx, "carrier_clk_minus_geom_m"] = left.loc[idx, "carrier_if_clk_m"] - geom

        parts.append(left.loc[idx])

    # Simply return the fully populated 'left' (original order preserved)
    return left


def plot_carrier_residuals(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    sat_col: str = "sat",
    value_col: str = "carrier_clk_minus_geom_m",
    title: Optional[str] = None,
    figsize: tuple[int, int] = (12, 6),
    satellites: Optional[Union[str, Sequence[str]]] = None,
):
    """Plot carrier clock-corrected IF minus geometric range for each satellite.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least columns specified by time_col, sat_col,
        and value_col.
    time_col : str
        Name of the time column (defaults to 'time').
    sat_col : str
        Name of the satellite ID column (defaults to 'sat').
    value_col : str
        Column to plot (defaults to 'carrier_clk_minus_geom_m').
    title : str | None
        Optional plot title.
    figsize : tuple[int, int]
        Matplotlib figure size.
    satellites : str | Sequence[str] | None
        Optional satellite identifiers to include in the plot. Rows where
        sat_col does not match are dropped.
    """

    import matplotlib.pyplot as plt

    if df.empty:
        raise ValueError("Input DataFrame is empty")
    if value_col not in df.columns:
        raise ValueError(f"'{value_col}' not found in DataFrame columns")
    if time_col not in df.columns or sat_col not in df.columns:
        raise ValueError("DataFrame must contain time and sat columns")

    data = df.copy()
    data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
    data = data.dropna(subset=[time_col, value_col, sat_col])

    sat_filter = None
    if satellites is not None:
        if isinstance(satellites, (str, bytes)):
            sat_filter = [satellites]
        else:
            sat_filter = list(dict.fromkeys(satellites))
        if not sat_filter:
            raise ValueError("No satellite identifiers provided in 'satellites'")
        data = data[data[sat_col].isin(sat_filter)]

    if data.empty:
        if sat_filter is not None:
            requested = ", ".join(map(str, sat_filter))
            raise ValueError(
                "No valid rows to plot after dropping NaNs for satellites: "
                f"{requested}"
            )
        raise ValueError("No valid rows to plot after dropping NaNs")

    fig, ax = plt.subplots(figsize=figsize)
    for prn, grp in data.groupby(sat_col, sort=True):
        grp = grp.sort_values(time_col)
        ax.plot(grp[time_col], grp[value_col], label=str(prn), linewidth=1.2)

    ax.set_xlabel("Time")
    ax.set_ylabel("Carrier IF (clock-corrected) - Geometric range [m]")
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    plt.tight_layout()
    return ax


def plot_code_residuals(
    df: pd.DataFrame,
    *,
    time_col: str = "time",
    sat_col: str = "sat",
    value_col: str = "code_clk_minus_geom_m",
    title: Optional[str] = None,
    figsize: tuple[int, int] = (12, 6),
    satellites: Optional[Union[str, Sequence[str]]] = None,
):
    """Plot code IF (clock-corrected) minus geometric range for each satellite.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing at least columns specified by time_col, sat_col,
        and value_col.
    time_col : str
        Name of the time column (defaults to 'time').
    sat_col : str
        Name of the satellite ID column (defaults to 'sat').
    value_col : str
        Column to plot (defaults to 'code_clk_minus_geom_m').
    title : str | None
        Optional plot title.
    figsize : tuple[int, int]
        Matplotlib figure size.
    satellites : str | Sequence[str] | None
        Optional satellite identifiers to include in the plot. Rows where
        sat_col does not match are dropped.
    """

    import matplotlib.pyplot as plt

    if df.empty:
        raise ValueError("Input DataFrame is empty")
    if value_col not in df.columns:
        raise ValueError(f"'{value_col}' not found in DataFrame columns")
    if time_col not in df.columns or sat_col not in df.columns:
        raise ValueError("DataFrame must contain time and sat columns")

    data = df.copy()
    data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
    data = data.dropna(subset=[time_col, value_col, sat_col])

    sat_filter = None
    if satellites is not None:
        if isinstance(satellites, (str, bytes)):
            sat_filter = [satellites]
        else:
            sat_filter = list(dict.fromkeys(satellites))
        if not sat_filter:
            raise ValueError("No satellite identifiers provided in 'satellites'")
        data = data[data[sat_col].isin(sat_filter)]

    if data.empty:
        if sat_filter is not None:
            requested = ", ".join(map(str, sat_filter))
            raise ValueError(
                "No valid rows to plot after dropping NaNs for satellites: "
                f"{requested}"
            )
        raise ValueError("No valid rows to plot after dropping NaNs")

    fig, ax = plt.subplots(figsize=figsize)
    for prn, grp in data.groupby(sat_col, sort=True):
        grp = grp.sort_values(time_col)
        ax.plot(grp[time_col], grp[value_col], label=str(prn), linewidth=1.2)

    ax.set_xlabel("Time")
    ax.set_ylabel("Code IF (clock-corrected) - Geometric range [m]")
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    plt.tight_layout()
    return ax


def plot_kin_xyz(
    kin_df: pd.DataFrame,
    *,
    time_col: Optional[str] = None,
    title: Optional[str] = None,
    figsize: tuple[int, int] = (12, 8),
):
    """Plot ECEF X, Y, Z from a PRIDE ``kin_*`` DataFrame in three subplots.

    The time axis is chosen in this order: ``time_col`` if provided, else
    ``gps_datetime`` if present, else ``datetime`` if present, else computed
    from ``Mjd`` + ``Sod`` (GPS timescale).
    """

    import matplotlib.pyplot as plt

    required = ["X", "Y", "Z"]
    missing = [c for c in required if c not in kin_df.columns]
    if missing:
        raise ValueError(f"kin_df missing required columns: {missing}")

    # Determine time series
    if time_col and time_col in kin_df.columns:
        t = pd.to_datetime(kin_df[time_col], errors="coerce")
    elif "gps_datetime" in kin_df.columns:
        t = pd.to_datetime(kin_df["gps_datetime"], errors="coerce")
    elif "datetime" in kin_df.columns:
        t = pd.to_datetime(kin_df["datetime"], errors="coerce")
    elif "Mjd" in kin_df.columns and "Sod" in kin_df.columns:
        t = mjd_sod_to_gps_datetime(kin_df["Mjd"], kin_df["Sod"])
    else:
        raise ValueError("No suitable time columns found (gps_datetime/datetime or Mjd+Sod)")

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=figsize)

    axes[0].plot(t, kin_df["X"], color="#1f77b4")
    axes[0].set_ylabel("X [m]")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].plot(t, kin_df["Y"], color="#ff7f0e")
    axes[1].set_ylabel("Y [m]")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    axes[2].plot(t, kin_df["Z"], color="#2ca02c")
    axes[2].set_ylabel("Z [m]")
    axes[2].set_xlabel("Time")
    axes[2].grid(True, linestyle=":", alpha=0.6)

    if title:
        fig.suptitle(title)

    fig.autofmt_xdate()
    plt.tight_layout()
    return axes


def plot_geometric_ranges(
    rng_df: pd.DataFrame,
    *,
    time_col: str = "gps_datetime",
    sat_col: str = "sat",
    value_col: str = "geometric_range_m",
    title: Optional[str] = None,
    figsize: tuple[int, int] = (12, 6),
):
    """Plot geometric ranges per satellite versus time.

    Parameters
    ----------
    rng_df : pd.DataFrame
        Output of geometric_ranges with columns including time_col, sat_col,
        and value_col.
    time_col, sat_col, value_col : str
        Column names for time, satellite id, and value to plot.
    title : str | None
        Optional title.
    figsize : tuple[int,int]
        Matplotlib figure size.
    """

    import matplotlib.pyplot as plt

    if rng_df.empty:
        raise ValueError("Input rng_df is empty")
    for col in (time_col, sat_col, value_col):
        if col not in rng_df.columns:
            raise ValueError(f"Column '{col}' not found in rng_df")

    data = rng_df.copy()
    data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
    data = data.dropna(subset=[time_col, value_col, sat_col])
    if data.empty:
        raise ValueError("No valid rows to plot after dropping NaNs")

    fig, ax = plt.subplots(figsize=figsize)
    for prn, grp in data.groupby(sat_col, sort=True):
        grp = grp.sort_values(time_col)
        ax.plot(grp[time_col], grp[value_col], label=str(prn), linewidth=1.2)

    ax.set_xlabel("Time")
    ax.set_ylabel("Geometric range [m]")
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    plt.tight_layout()
    return ax


def plot_satellite_ecef_errors(
    pos_df: pd.DataFrame,
    ecef_df: pd.DataFrame,
    *,
    pos_time_col: Optional[str] = "gps_datetime",
    ecef_time_col: str = "gps_datetime",
    sat_col: str = "sat",
    tolerance_s: float = 0.25,
    figsize: tuple[int, int] = (12, 8),
    title: Optional[str] = None,
    svs: Optional[Sequence[str]] = None,
) -> tuple[pd.DataFrame, any]:
    """Compare PRIDE satellite positions (pos_df) with propagated ECEF (ecef_df).

    - Matches by satellite and time with asof within tolerance.
    - Expects in pos_df: columns 'Sat_X_m','Sat_Y_m','Sat_Z_m' and either a
      time column or ('Mjd','Sod'). Optional 'sat' column for per-PRN data.
    - Expects in ecef_df: columns ecef_time_col, 'sat', 'ECEF_x','ECEF_y','ECEF_z'.

    Returns (errors_df, matplotlib_axes).
    errors_df columns: ['time','sat','err_x_m','err_y_m','err_z_m'].
    """

    import matplotlib.pyplot as plt

    req_pos = ["Sat_X_m", "Sat_Y_m", "Sat_Z_m"]
    missing = [c for c in req_pos if c not in pos_df.columns]
    if missing:
        raise ValueError(f"pos_df missing required columns: {missing}")

    # Determine pos time
    if pos_time_col and pos_time_col in pos_df.columns:
        pos_t = pd.to_datetime(pos_df[pos_time_col], errors="coerce")
    elif "gps_datetime" in pos_df.columns:
        pos_t = pd.to_datetime(pos_df["gps_datetime"], errors="coerce")
    elif "time" in pos_df.columns:
        pos_t = pd.to_datetime(pos_df["time"], errors="coerce")
    elif {"Mjd", "Sod"}.issubset(pos_df.columns):
        pos_t = mjd_sod_to_gps_datetime(pos_df["Mjd"], pos_df["Sod"])
    else:
        raise ValueError("pos_df must contain a time column or Mjd+Sod")

    pos = pos_df.copy()
    pos["_pos_time"] = pos_t
    if sat_col in pos.columns:
        pos[sat_col] = pos[sat_col].astype(str)

    # Check ecef_df
    req_ecef = [ecef_time_col, sat_col, "ECEF_x", "ECEF_y", "ECEF_z"]
    missing_e = [c for c in req_ecef if c not in ecef_df.columns]
    if missing_e:
        raise ValueError(f"ecef_df missing required columns: {missing_e}")

    ecef = ecef_df[[ecef_time_col, sat_col, "ECEF_x", "ECEF_y", "ECEF_z"]].copy()
    ecef[ecef_time_col] = pd.to_datetime(ecef[ecef_time_col], errors="coerce")
    ecef[sat_col] = ecef[sat_col].astype(str)

    # Optional filter by a provided list of satellites
    if svs is not None:
        sv_set = set(str(s) for s in (svs.values if hasattr(svs, "values") else svs))
        ecef = ecef[ecef[sat_col].isin(sv_set)]
        if sat_col in pos.columns:
            pos = pos[pos[sat_col].astype(str).isin(sv_set)]

    tol = pd.to_timedelta(tolerance_s, unit="s")

    err_rows: list[pd.DataFrame] = []
    for prn, left in ecef.groupby(sat_col, sort=False):
        if sat_col in pos.columns:
            right = pos[pos[sat_col] == prn]
        else:
            right = pos
        if right.empty:
            continue
        left_sorted = left.sort_values(ecef_time_col).assign(_idx=left.index)
        right_sorted = right.dropna(subset=["_pos_time"]).sort_values("_pos_time")
        merged = pd.merge_asof(
            left_sorted,
            right_sorted[["_pos_time", "Sat_X_m", "Sat_Y_m", "Sat_Z_m"]],
            left_on=ecef_time_col,
            right_on="_pos_time",
            direction="nearest",
            tolerance=tol,
        )
        # Compute errors where matched
        idx = merged["_idx"]
        ex = merged["ECEF_x"] - merged["Sat_X_m"]
        ey = merged["ECEF_y"] - merged["Sat_Y_m"]
        ez = merged["ECEF_z"] - merged["Sat_Z_m"]
        out = pd.DataFrame({
            "time": merged[ecef_time_col],
            "sat": prn,
            "err_x_m": ex,
            "err_y_m": ey,
            "err_z_m": ez,
        }, index=idx)
        err_rows.append(out.dropna(subset=["err_x_m", "err_y_m", "err_z_m"], how="any"))

    if not err_rows:
        raise ValueError("No matches within tolerance to compute errors.")

    errors_df = pd.concat(err_rows).sort_index().reset_index(drop=True)

    # Plot
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=figsize)
    for prn, grp in errors_df.groupby("sat", sort=True):
        grp = grp.sort_values("time")
        axes[0].plot(grp["time"], grp["err_x_m"], label=str(prn), linewidth=1.0)
        axes[1].plot(grp["time"], grp["err_y_m"], label=str(prn), linewidth=1.0)
        axes[2].plot(grp["time"], grp["err_z_m"], label=str(prn), linewidth=1.0)

    axes[0].set_ylabel("dX [m]")
    axes[1].set_ylabel("dY [m]")
    axes[2].set_ylabel("dZ [m]")
    axes[2].set_xlabel("Time")
    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.6)
    if title:
        fig.suptitle(title)
    axes[0].legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    plt.tight_layout()
    return errors_df, axes
