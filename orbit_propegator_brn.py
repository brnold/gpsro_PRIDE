"""Compute ECEF positions of GNSS satellites from pandas DataFrame.

This script expects a pandas DataFrame with columns:
- 'week': GPS week
- 'tow_sec': Time of week in seconds  
- 'sat': Satellite PRN (e.g., 'G10', 'G15')
- 'ECEF_x', 'ECEF_y', 'ECEF_z': ECEF coordinates in meters (optional)

It can either use existing ECEF coordinates from the DataFrame or download 
SP3 precise orbits (cached under *sp3_cache/*) and interpolate them to the 
observation epochs.

Usage:
    import pandas as pd
    from orbit_propegator_brn import load_ecef_from_dataframe
    
    df = pd.read_csv('your_data.csv')
    satellite_data = load_ecef_from_dataframe(df)
"""

# flake8: noqa: D401  # (simple so we keep doc-style short)

from __future__ import annotations

import pathlib
import datetime
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Union

import astropy.time as atime
import astropy.units as u
import numpy as np
import pandas as pd
import sp3
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from common_ro_functions import gps_week_tow_gps_dt

# =============================================================================
# Precise clock file helper
# =============================================================================

# IGS CLK files contain precise satellite clock corrections (seconds) that we
# need in **nanoseconds** for the ionosphere-free pseudorange routine.

import re



sp3.cddis.username = "benjamin.r.nold"
sp3.cddis.password = "bfy9qvt.qfr7gmp6DFH"

def parse_igs_clk_file(
    clk_paths: str | pathlib.Path | list[str | pathlib.Path],
) -> pd.DataFrame:
    """Parse one or multiple IGS ``*.CLK`` files to a tidy DataFrame.

    The returned DataFrame has the columns

    * ``sat``          — satellite identifier (e.g. ``G01``)
    * ``gps_time``     — naive ``datetime.datetime`` in UTC
    * ``sat_clk_ns``   — clock bias **nanoseconds** (float64)

    Parameters
    ----------
    clk_paths : str | pathlib.Path | list[str | pathlib.Path]
        Path(s) to the CLK file(s).  If multiple files are given their content
        is concatenated before parsing so you get a single, continuous
        DataFrame ready for :pyfunc:`pandas.merge_asof`.

    Notes
    -----
    Only lines starting with ``AS`` (satellite records) are considered.  Bias
    values are parsed from column 9 (index 8) as **seconds** and immediately
    converted to nanoseconds (\(1\,\text{ns}=1e-9\,\text{s}\)).
    """

    if not isinstance(clk_paths, (list, tuple)):
        clk_paths = [clk_paths]

    rows: list[tuple[str, datetime.datetime, float]] = []

    # Pre-compiled regex for performance (four spaces or single space between fields)
    _clk_re = re.compile(r"^AS\s+(\S+)\s+(\d{4})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+([0-9.]+)\s+([\d.eE+\-]+)")

    for path in map(pathlib.Path, clk_paths):
        if not path.is_file():
            raise FileNotFoundError(path)

        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if not line.startswith("AS"):
                    continue

                m = _clk_re.match(line)
                if not m:
                    # Fallback: space-split if regex fails (robust against format quirks)
                    parts = line.split()
                    if len(parts) < 9:
                        continue  # malformed
                    sat_id = parts[1]
                    try:
                        y, mth, d, hh, mm = map(int, parts[2:7])
                        sec = float(parts[7])
                        bias_s = float(parts[8])
                    except ValueError:
                        continue
                else:
                    sat_id = m.group(1)
                    y, mth, d, hh, mm = map(int, m.groups()[1:6])
                    sec = float(m.group(7))
                    bias_s = float(m.group(8))

                # Build naive UTC datetime -------------------------------------------------
                whole = int(sec)
                frac = sec - whole
                dt = datetime.datetime(y, mth, d, hh, mm, whole) + datetime.timedelta(seconds=frac)

                rows.append((sat_id, dt, bias_s * 1e9))  # ns

    if not rows:
        raise RuntimeError("No satellite clock records (AS lines) found in CLK file(s).")

    df = pd.DataFrame(rows, columns=["sat", "gps_time", "sat_clk_ns"])
    # Sort for efficient asof-merge later
    df.sort_values(["sat", "gps_time"], inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)
    return df


# Add to module exports if available -------------------------------------------------
try:
    __all__.append("parse_igs_clk_file")  # type: ignore[var-annotated]
except NameError:
    __all__ = ["parse_igs_clk_file"]

# =============================================================================
# Satellite clock bias from SP3
# =============================================================================


def parse_sp3_clock_file(
    sp3_paths: str | pathlib.Path | list[str | pathlib.Path],
    interpolate_to: pd.DataFrame | None = None,
    *,
    time_col: str = "gps_time",
) -> pd.DataFrame:
    """Extract satellite clock biases from one or multiple SP3 files.

    Parameters
    ----------
    sp3_paths : str | pathlib.Path | list[str | pathlib.Path]
        Path(s) to SP3 **orbit** files (type *c*, *d*, etc.).  Multiple files
        are concatenated chronologically so you get a single DataFrame ready
        for :pyfunc:`pandas.merge_asof`.

    Returns
    -------
    pd.DataFrame
        If ``interpolate_to`` is None: columns ``sat`` (str), ``gps_time`` (naive UTC),
        ``sat_clk_ns`` (nanoseconds).

        If ``interpolate_to`` is provided (must have columns ``sat`` and ``time_col``),
        the result is interpolated to those epochs and additionally contains
        ``sat_clk_m`` (metres).

    Notes
    -----
    The SP3 specification stores satellite clock bias as seconds in structured
    parsers and as microseconds on the ``P`` record in plain-text. We convert to
    nanoseconds so it matches the unit expected by
    :pyfunc:`calculate_ionosphere_free_pseudorange_with_clock`.
    """

    if not isinstance(sp3_paths, (list, tuple)):
        sp3_paths = [sp3_paths]

    rows: list[tuple[str, datetime.datetime, float]] = []

    for path in map(pathlib.Path, sp3_paths):
        if not path.is_file():
            raise FileNotFoundError(path)

        try:
            # Preferred: use sp3's own parser which handles format variants
            product = sp3.Product.from_file(str(path))

            for sat in product.satellites:
                sat_id: str = sat.id.decode() if isinstance(sat.id, (bytes, bytearray)) else str(sat.id)
                for record in sat.records:
                    if record.clock is None:
                        continue
                    # SP3 clocks are in seconds → convert to nanoseconds
                    rows.append((sat_id, record.time, record.clock * 1e9))

            continue  # done with this path
        except Exception:
            # Fallback to lightweight manual parsing (covers edge cases, no dependency on full sp3 structure)
            pass

        # Manual plain-text parse (as before) ---------------------------------
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            current_epoch: datetime.datetime | None = None

            for line in fh:
                if not line:
                    continue

                if line.startswith("*"):
                    try:
                        year = int(line[3:7])
                        month = int(line[8:10])
                        day = int(line[11:13])
                        hour = int(line[14:16])
                        minute = int(line[17:19])
                        sec = float(line[20:31])
                        whole = int(sec)
                        frac = sec - whole
                        current_epoch = datetime.datetime(
                            year, month, day, hour, minute, whole, tzinfo=datetime.timezone.utc
                        ) + datetime.timedelta(seconds=frac)
                    except ValueError:
                        current_epoch = None
                    continue

                if line.startswith("P") and current_epoch is not None:
                    prn = line[1:4].strip()
                    try:
                        clk_us = float(line[46:60])
                    except ValueError:
                        continue

                    if abs(clk_us) >= 999999.0:
                        continue

                    rows.append((prn, current_epoch, clk_us * 1_000.0))

    if not rows:
        raise RuntimeError("No satellite clock data found in SP3 file(s).")

    df = pd.DataFrame(rows, columns=["sat", "gps_time", "sat_clk_ns"])
    # Normalize to naive UTC to match other frames (e.g. pseudorange/CLK tables)
    df["gps_time"] = pd.to_datetime(df["gps_time"], utc=True).dt.tz_localize(None)
    df.sort_values(["sat", "gps_time"], inplace=True, kind="mergesort")
    df.reset_index(drop=True, inplace=True)

    # Optional interpolation to target epochs per satellite ----------------------
    if interpolate_to is None or interpolate_to.empty:
        return df

    target = interpolate_to.copy()
    if not {"sat", time_col}.issubset(target.columns):
        raise ValueError(f"interpolate_to must contain columns 'sat' and '{time_col}'")

    target["sat"] = target["sat"].astype(str)
    target[time_col] = pd.to_datetime(target[time_col], errors="coerce")
    target = target.dropna(subset=[time_col])

    results: list[pd.DataFrame] = []
    c_mps = 299_792_458.0
    for prn, sat_grp in target.groupby("sat", sort=False):
        clk_grp = df[df["sat"] == prn]
        if clk_grp.empty:
            # No clocks for this PRN → fill NaNs for required shape
            res = sat_grp[[time_col]].copy()
            res.insert(0, "sat", prn)
            res["sat_clk_ns"] = np.nan
            res["sat_clk_m"] = np.nan
            results.append(res)
            continue

        # Convert times to float seconds for interpolation
        x = sat_grp[time_col].astype("int64").to_numpy(dtype="float64") / 1e9
        xp = clk_grp["gps_time"].astype("int64").to_numpy(dtype="float64") / 1e9
        fp = clk_grp["sat_clk_ns"].to_numpy(dtype="float64")

        # Linear interpolation with NaN outside the support
        interp_ns = np.interp(x, xp, fp, left=np.nan, right=np.nan)

        res = sat_grp[[time_col]].copy()
        res.insert(0, "sat", prn)
        res.rename(columns={time_col: "gps_time"}, inplace=True)
        res["sat_clk_ns"] = interp_ns
        res["sat_clk_m"] = c_mps * interp_ns * 1e-9
        results.append(res)

    out = pd.concat(results, ignore_index=True)
    out.sort_values(["sat", "gps_time"], inplace=True, kind="mergesort")
    out.reset_index(drop=True, inplace=True)
    return out


# -- Mark helper as public ----------------------------------------------------
try:
    __all__.append("parse_sp3_clock_file")  # type: ignore[var-annotated]
except NameError:
    __all__ = ["parse_sp3_clock_file"]

# parse_igs_clk_file remains for backwards compat but recommend using SP3 -----

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def _process_satellite_worker(prn: str, times_list: list[datetime.datetime], cache_dir: pathlib.Path) -> tuple[list[int], np.ndarray]:
    """Compute ECEF for *prn* at *times_list*.

    To accelerate things we first **deduplicate** times so the expensive SP3
    interpolation is executed at most once per distinct epoch.
    The returned array is re-expanded to match the original order so the caller
    can align by index.
    """
    # Deduplicate (preserve order via pandas factorisation)
    times_series = pd.Series(times_list)
    unique_times, inverse = np.unique(times_series, return_inverse=True)
    unique_times = list(unique_times)  # convert back to python datetimes

    coords_unique = get_satellite_ecef(prn, unique_times, cache_dir)
    coords_full = coords_unique[inverse]
    return inverse.tolist(), coords_full
# -----------------------------------------------------------------------------
# Velocity helpers (from SP3 product records)
# -----------------------------------------------------------------------------


def _load_sp3_velocity_df_for_prn(prn: str, cache_dir: pathlib.Path) -> pd.DataFrame:
    """Load SP3-record velocities for a given PRN from cached SP3 files.

    Returns DataFrame with columns: gps_time (naive UTC), vx, vy, vz (m/s).
    Empty DataFrame if none found.
    """
    cache_dir = pathlib.Path(cache_dir)
    if not cache_dir.exists():
        return pd.DataFrame(columns=["gps_time", "vx", "vy", "vz"])

    rows: list[tuple[datetime.datetime, float, float, float]] = []

    # Find SP3 files in cache (non-recursive and recursive for safety)
    sp3_paths = list(cache_dir.glob("*.sp3*")) + list(cache_dir.glob("**/*.sp3*"))

    for path in sp3_paths:
        try:
            product = sp3.Product.from_file(str(path))
        except Exception:
            continue
        # Find satellite
        for sat in product.satellites:
            sat_id = sat.id.decode() if isinstance(sat.id, (bytes, bytearray)) else str(sat.id)
            if sat_id != prn:
                continue
            for rec in sat.records:
                if getattr(rec, "velocity", None) is None:
                    continue
                vx, vy, vz = rec.velocity  # m/s per provided dataclass
                t = rec.time
                # Normalize to naive UTC for merging
                t = pd.to_datetime(t, utc=True).tz_localize(None)
                rows.append((t, float(vx), float(vy), float(vz)))

    if not rows:
        return pd.DataFrame(columns=["gps_time", "vx", "vy", "vz"])

    df = pd.DataFrame(rows, columns=["gps_time", "vx", "vy", "vz"]).sort_values("gps_time", kind="mergesort")
    return df


def _process_satellite_velocity_worker(
    prn: str,
    gps_seconds_list: list[float],
    cache_dir: pathlib.Path,
) -> tuple[str, np.ndarray]:
    """Align SP3-record velocities to the requested GPS-second epochs via asof-merge.

    Returns (prn, vel_array[N,3]). If no velocity sources available, returns NaNs.
    """
    # Ensure SP3 cache contains products for these times. A prior call to get_satellite_ecef
    # usually triggers downloads; we just rely on the cache here.
    vel_df = _load_sp3_velocity_df_for_prn(prn, cache_dir)
    if vel_df.empty:
        return prn, np.full((len(gps_seconds_list), 3), np.nan)

    # Convert GPS seconds to naive UTC datetimes for alignment with SP3 times
    gps_times = atime.Time(gps_seconds_list, format="gps").utc.datetime
    left = pd.DataFrame({"gps_time": pd.to_datetime(gps_times, utc=True).tz_localize(None)})

    merged = pd.merge_asof(
        left.sort_values("gps_time"),
        vel_df.sort_values("gps_time"),
        on="gps_time",
        direction="nearest",
        tolerance=pd.Timedelta("300s"),  # up to 5 minutes
    )

    v = merged[["vx", "vy", "vz"]].to_numpy(dtype=float)
    return prn, v




def velocities_lsq(
    times_s: list[float] | np.ndarray,
    pos_m: np.ndarray,
    window: int = 5,
    poly_deg: int = 2,
) -> np.ndarray:
    """Compute velocities via local least-squares polynomial fit.

    Parameters
    ----------
    times_s : (N,) array
        Epoch times in seconds (non-uniform is okay).
    pos_m : (N, 3) array
        ECEF positions in metres.
    window : int, optional
        Number of points for local fit (must be odd, >= 3). Defaults to 5.
    poly_deg : int, optional
        Degree of fitting polynomial (1, 2, or 3 is typical). Defaults to 2.

    Returns
    -------
    np.ndarray
        ECEF velocities in m/s, shape (N, 3).
    """
    import numpy as _np

    t = _np.asarray(times_s, dtype=float)
    r = _np.asarray(pos_m, dtype=float)

    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError("pos_m must be shape (N, 3)")

    if t.ndim != 1 or t.shape[0] != r.shape[0]:
        raise ValueError("times_s must be shape (N,) and align with pos_m")

    if window % 2 == 0:
        raise ValueError("window must be an odd number")

    n = t.shape[0]
    v = _np.full_like(r, _np.nan)
    half = window // 2

    for i in range(n):
        i0 = max(0, i - half)
        i1 = min(n, i + half + 1)

        # Ensure we have enough points for the fit
        if i1 - i0 < poly_deg + 1:
            continue

        # Centre time for better numerical conditioning
        tt = t[i0:i1] - t[i]
        rr = r[i0:i1]

        # Design matrix for polynomial fit: [1, t, t^2, ...]
        A = _np.vander(tt, N=poly_deg + 1, increasing=True)

        # Solve three independent LSQ fits (one for each axis)
        try:
            coef, *_ = _np.linalg.lstsq(A, rr, rcond=None)
            # The velocity at the centre point (t=0) is the linear coefficient
            v[i] = coef[1]
        except _np.linalg.LinAlgError:
            continue  # Skip if LSQ fails

    return v


try:
    __all__.append("velocities_lsq")  # type: ignore[var-annotated]
except NameError:
    __all__ = ["velocities_lsq"]


def _gps_to_datetime_vectorized(df: pd.DataFrame, week_col: str, tow_col: str) -> pd.Series:
    """Vectorized conversion of GPS week and TOW to datetime objects."""
    gps_epoch = pd.Timestamp("1980-01-06")
    # Use .to_numpy() for faster arithmetic on the underlying array
    total_seconds = df[week_col].to_numpy() * 604800 + df[tow_col].to_numpy()
    return pd.to_timedelta(total_seconds, unit='s') + gps_epoch


def _parse_dataframe_for_times(df: pd.DataFrame) -> Dict[str, List[datetime.datetime]]:
    """Extract satellite observation times from DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'week', 'tow_sec', and 'sat'
        
    Returns
    -------
    dict
        Mapping PRN -> list of datetime objects
    """
    required_cols = ['week', 'tow_sec', 'sat']
    if not all(col in df.columns for col in required_cols):
        missing = set(required_cols) - set(df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")
    
    per_sat: Dict[str, List[datetime.datetime]] = defaultdict(list)
    
    for _, row in df.iterrows():
        try:
            gps_week = int(row['week'])
            tow = float(row['tow_sec'])
            prn = str(row['sat']).strip()
            
            # Convert to datetime object
            dt = gps_week_tow_gps_dt(gps_week, tow)
            per_sat[prn].append(dt)
        except (ValueError, KeyError):
            # Skip malformed rows
            continue
    
    return per_sat


# -----------------------------------------------------------------------------
# Public dataclass & API                                                       
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class SatelliteECEF:
    """Container for a single satellite's propagated ECEF positions."""

    times: np.ndarray  # shape (N,) of datetime objects
    ecef: np.ndarray   # shape (N, 3) float64, metres (x, y, z)


def get_satellite_ecef(
    prn: str,
    obstimes: List[Union[float, datetime.datetime]],
    cache_dir: str | pathlib.Path = "sp3_cache",
) -> np.ndarray:
    """Return ECEF XYZ [m] for *prn* at *obstimes*."""

    # astropy can handle datetime objects directly. If times are floats, we must
    # specify the format is GPS seconds.
    time_format = "gps" if obstimes and isinstance(obstimes[0], (float, np.floating)) else None

    try:
        itrs = sp3.itrs(
            id=sp3.Sp3Id(prn.encode()),
            obstime=atime.Time(obstimes, format=time_format).utc,
            download_directory=cache_dir,
        )
        # astropy ITRS -> cartesian with metre units
        return np.vstack(
            [itrs.x.to_value(u.m), itrs.y.to_value(u.m), itrs.z.to_value(u.m)]
        ).T  # shape (N, 3)
    except Exception:
        # If sp3 fails (e.g., no provider for a satellite), return NaNs
        # to allow processing of other satellites to continue.
        num_obs = len(obstimes)
        return np.full((num_obs, 3), np.nan)



# Main API functions ----------------------------------------------------------


def load_ecef_from_dataframe(
    pos: pd.DataFrame,
    sv: list[str] | np.ndarray,
    cache_dir: str | pathlib.Path = "sp3_cache",
    n_workers: int | None = None,
    use_processes: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute satellite ECEF for given PRNs at epochs from a ``kin`` DataFrame.

    Parameters
    ----------
    pos : pd.DataFrame
        DataFrame with columns ``Mjd`` and ``Sod`` (from PRIDE kin file).
    sv : list[str] | np.ndarray
        Sequence of satellite IDs (e.g. from ``obs.sv`` in georinex). Supports
        mixed constellations like 'G', 'E', 'R', 'C', 'J'.
    cache_dir : str | pathlib.Path
        Directory where SP3 products will be cached/downloaded.
    n_workers : int | None
        Parallel workers for satellite computations. Defaults to a sensible CPU-based value.
    use_processes : bool
        If True, use processes; otherwise use threads.

    Returns
    -------
    pd.DataFrame
        Tidy frame with columns: ``Mjd``, ``Sod``, ``gps_datetime``, ``sat``,
        ``ECEF_x``, ``ECEF_y``, ``ECEF_z``. ``gps_datetime`` is naive
        pandas datetime in GPS timescale (no leap seconds), derived from MJD+SOD.
    """

    if verbose:
        import time as _time
        _t0 = _time.perf_counter()
        print("Starting ECEF propagation...")
        try:
            print(f"  - epochs: {len(pos)}  (columns: {list(pos.columns)})")
        except Exception:
            pass

    # Convert MJD+SOD to GPS seconds for astropy Time(format="gps") pipeline
    mjd = pos["Mjd"].to_numpy(dtype="float64")
    sod = pos["Sod"].to_numpy(dtype="float64")
    gps_seconds_all = (mjd - 44244.0) * 86400.0 + sod  # MJD(GPS epoch)=44244
    # Build GPS timescale datetimes (naive) once and attach per-PRN below
    _gps_epoch = pd.Timestamp("1980-01-06 00:00:00")
    gps_datetime_full = _gps_epoch + pd.to_timedelta(gps_seconds_all, unit="s")

    # Deduplicate epochs for speed, map back after compute
    unique_gps_seconds, inverse = np.unique(gps_seconds_all, return_inverse=True)
    unique_list = unique_gps_seconds.tolist()
    if verbose:
        saved = 1.0 - (len(unique_gps_seconds) / max(1, len(gps_seconds_all)))
        print(f"  - deduplicated epochs: {len(unique_gps_seconds)}/{len(gps_seconds_all)}  (saved {saved*100:.1f}%)")

    # Normalise PRN list
    try:
        prns = [str(s) for s in (sv.values if hasattr(sv, "values") else sv)]
    except Exception:
        prns = [str(s) for s in sv]
    if verbose:
        print(f"  - satellites: {len(prns)}  -> {', '.join(prns[:8])}{'...' if len(prns) > 8 else ''}")

    cache = pathlib.Path(cache_dir)
    cache.mkdir(exist_ok=True)
    if verbose:
        print(f"  - SP3 cache: {cache.resolve()}")

    if n_workers is None:
        from os import cpu_count
        n_workers = min(32, (cpu_count() or 1) + 4)

    Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    if verbose:
        print(f"  - executor: {'ProcessPool' if use_processes else 'ThreadPool'}  workers={n_workers}")

    def _work(prn: str) -> tuple[str, np.ndarray]:
        coords_unique = get_satellite_ecef(prn, unique_list, cache)
        # Map back to full-length order
        return prn, coords_unique[inverse]

    results: list[pd.DataFrame] = []
    with Executor(max_workers=n_workers) as exe:
        futures = {exe.submit(_work, prn): prn for prn in prns}
        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            prn = futures[fut]
            try:
                prn, coords_full = fut.result()
            except Exception as exc:
                if verbose:
                    print(f"    - {prn}: error {exc!r}; filling NaNs")
                coords_full = np.full((len(pos), 3), np.nan)
            df_prn = pos[["Mjd", "Sod"]].copy()
            df_prn["gps_datetime"] = gps_datetime_full
            df_prn["sat"] = prn
            df_prn[["ECEF_x", "ECEF_y", "ECEF_z"]] = coords_full
            results.append(df_prn)
            done += 1
            if verbose:
                print(f"    - {prn}: done ({done}/{total})")

    out = pd.concat(results, ignore_index=True)
    if verbose:
        _t1 = _time.perf_counter()
        print(f"Finished ECEF propagation: sats={len(prns)} epochs={len(pos)}  time={_t1 - _t0:.2f}s")
    return out


# Legacy function (for backward compatibility) -------------------------------
# deprecated please this is horrible do not use me.
def _parse_pos_stat_file(path: str | pathlib.Path) -> Dict[str, List[datetime.datetime]]:
    """Read a .pos.stat file and return a mapping PRN -> list of datetimes."""
    import csv

    per_sat: Dict[str, List[datetime.datetime]] = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0] != "$SAT":
                continue
            try:
                gps_week = int(row[1])
                tow = float(row[2])
                prn = row[3].strip()
                per_sat[prn].append(gps_week_tow_gps_dt(gps_week, tow))
            except (ValueError, IndexError):
                continue
    return per_sat


def load_ecef_from_pos_stat_file(
    pos_stat_path: str | pathlib.Path,
    cache_dir: str | pathlib.Path = "sp3_cache",
    verbose: bool = False,
) -> Dict[str, SatelliteECEF]:
    """[DEPRECATED] Convert a `.pos.stat` file to ECEF arrays."""
    if verbose:
        import time
        start_time = time.perf_counter()
        print("Starting ECEF processing from .pos.stat file...")

    per_sat_dt = _parse_pos_stat_file(pos_stat_path)
    if not per_sat_dt:
        raise RuntimeError("No $SAT records found – nothing to do.")

    cache = pathlib.Path(cache_dir)
    cache.mkdir(exist_ok=True)

    if verbose:
        ecef_processing_start = time.perf_counter()
        num_sats = len(per_sat_dt)
        print(f"Processing ECEF for {num_sats} satellites...")

    results: Dict[str, SatelliteECEF] = {}
    for i, (prn, times_dt) in enumerate(per_sat_dt.items()):
        ecef = get_satellite_ecef(prn, times_dt, cache)
        results[prn] = SatelliteECEF(times=np.array(times_dt), ecef=ecef)
        if verbose and (i + 1) % 5 == 0:
            print(f"    - Processed {i + 1}/{num_sats} satellites...")

    if verbose:
        ecef_processing_end = time.perf_counter()
        print(f"  - ECEF processing took: {ecef_processing_end - ecef_processing_start:.4f} seconds")
        total_time = time.perf_counter() - start_time
        print(f"Total processing time: {total_time:.4f} seconds")

    return results


# -----------------------------------------------------------------------------
# CLI helper (kept for quick checks)                                           
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Convert $SAT records to in-memory ECEF coordinates using precise SP3 orbits.",
    )
    parser.add_argument("pos_stat", help="Path to the *.pos.stat file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print timing and progress information.")
    args = parser.parse_args()

    try:
        data = load_ecef_from_pos_stat_file(args.pos_stat, verbose=args.verbose)
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Error: {exc}")

    print("Summary of propagated satellites:")
    for prn, rec in data.items():
        # astropy can create a Time object directly from a datetime object
        first_t = atime.Time(rec.times[0]).iso
        last_t = atime.Time(rec.times[-1]).iso
        print(f"  {prn}: {len(rec.times)} epochs  ({first_t}  ->  {last_t})")