"""
GFW Small-Vessel Spatio-Temporal Site Selector (Plan A + time dimension)
========================================================================
Selects high-resolution satellite imagery purchase targets that are rich in
SMALL fishing vessels, broken down by month / week / day.

Pipeline:
  1) Resolve EEZ region id (default: Australia)
  2) Events API   -> pull fishing events (each carries time + position + vessel)
  3) Vessels API  -> look up each vessel's length
  4) Classify by length:
        core (<10m, top priority)  /  mid (10-24m)  /
        dark (length missing = "dark vessel" candidate)  /  big (>=24m, dropped)
  5) Aggregate by (grid cell x time bucket) at MONTH / WEEK / DAY granularity,
     rank by a weighted small-vessel score (so <10m dominates), and emit
     three purchase lists with AOI boxes (WKT).

NOTE on nested fields:
  GFW Events do NOT expose flat 'lat'/'lon' columns. Coordinates live inside
  the 'position' field (typically a dict like {"lat":..., "lon":...}), and the
  vessel info lives inside the 'vessel' field. This script unpacks both with a
  robust parser that handles dict / JSON-string / object forms.

Install : pip install gfw-api-python-client pandas numpy
Token   : https://globalfishingwatch.org/our-apis/tokens
Terms   : https://globalfishingwatch.org/our-apis/documentation#terms-of-use

Lines marked  # ADJUST  depend on exact SDK method/field names; tweak on first
run if the script reports a mismatch.
"""

import ast
import asyncio
import json
import math
import os

import gfwapiclient as gfw
import numpy as np
import pandas as pd


# ============================================================
# Config
# ============================================================
ACCESS_TOKEN = os.environ.get("GFW_API_TOKEN", "<PASTE_YOUR_GFW_API_ACCESS_TOKEN_HERE>")
COUNTRY_NAME = "Australia"
START_DATE = "2023-01-01"
END_DATE = "2023-01-31"

CORE_LEN = 10.0  # < 10m  -> core target
MAX_LEN = 24.0  # < 24m  -> included; >= 24m dropped
W_CORE, W_MID, W_DARK = 5.0, 1.0, 2.0  # weights so <10m dominates ranking

GRID_DEG = 0.01  # ~1 km grid
TOP_N = 50  # rows per time granularity
AOI_HALF_KM = 5.0  # half side length of purchase box (km)
MIN_CORE_PER_ROW = 1  # min number of <10m vessels per (cell x time)

OUT_MONTH = "hotspots_by_month.csv"
OUT_WEEK = "hotspots_by_week.csv"
OUT_DAY = "hotspots_by_day.csv"


# ============================================================
# Helpers
# ============================================================
def km_to_deg_lat(km):
    return km / 111.0


def km_to_deg_lon(km, lat):
    return km / (111.320 * math.cos(math.radians(lat)) + 1e-9)


def find_col(df, candidates):
    """Find a column by name (exact first, then substring)."""
    for c in candidates:
        for col in df.columns:
            if c.lower() == col.lower():
                return col
    for c in candidates:
        for col in df.columns:
            if c.lower() in col.lower():
                return col
    return None


def to_df(result):
    """Normalize an SDK response into a DataFrame."""
    if hasattr(result, "df"):
        try:
            return result.df()
        except Exception:
            pass
    if hasattr(result, "data"):
        data = result.data() if callable(result.data) else result.data
        return pd.DataFrame(data)
    return pd.DataFrame(result)


def coerce_obj(value):
    """Turn a nested field into a plain dict.
    Handles: dict already / JSON string / python-repr string / object with attrs.
    Returns {} if it cannot be parsed.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return {}
        for parser in (json.loads, ast.literal_eval):
            try:
                out = parser(s)
                if isinstance(out, dict):
                    return out
            except Exception:
                continue
        return {}
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return {}


def get_nested(d, keys):
    """Get the first present key from a dict (case-insensitive)."""
    if not isinstance(d, dict):
        return None
    lower = {k.lower(): v for k, v in d.items()}
    for k in keys:
        if k.lower() in lower:
            return lower[k.lower()]
    return None


# ============================================================
# 1) EEZ id
# ============================================================
async def lookup_eez_id(client, country):
    print(f"[1/5] Resolving EEZ id for {country} ...")
    res = await client.references.get_eez_regions()  # ADJUST
    df = to_df(res)
    name_cols = [c for c in df.columns if df[c].dtype == object]
    mask = pd.Series(False, index=df.index)
    for c in name_cols:
        mask |= df[c].astype(str).str.contains(country, case=False, na=False)
    hits = df[mask]
    if hits.empty:
        print(df.head(20).to_string())
        raise SystemExit("EEZ not found - set the id manually.")
    id_col = find_col(hits, ["id"]) or hits.columns[0]
    eez = str(hits.iloc[0][id_col])
    print(f"  Using EEZ id = {eez}")
    return eez


# ============================================================
# 2) Fishing events (time + position + vessel preserved)
# ============================================================
async def fetch_events(client, eez):
    print(f"[2/5] Fetching fishing events {START_DATE}..{END_DATE} ...")
    res = await client.events.get_all_events(  # ADJUST
        datasets=["public-global-fishing-events:latest"],  # ADJUST
        start_date=START_DATE,
        end_date=END_DATE,
        region={"dataset": "public-eez-areas", "id": eez},  # ADJUST
    )
    df = to_df(res)
    print(f"  {len(df)} events. Columns: {list(df.columns)}")
    return df


# ============================================================
# 3) Vessel length lookup (BATCH + correct nested parsing)
# ============================================================
# IMPORTANT: In the GFW Vessels API response, length is "lengthM" (camelCase,
# capital M) and it lives INSIDE the nested "registryInfo" list, e.g.:
#   registryInfo: [ { "lengthM": 53.6, "tonnageGt": 369, ... } ]
# It is NOT a top-level "length" column. That is why the old code returned all
# NaN. Below we (a) query vessels in BATCHES by id, and (b) dig lengthM /
# tonnageGt out of registryInfo (with fallbacks to other nested blocks).

BATCH_SIZE = 50  # how many vessel ids per batched request


def _dig_length_tonnage(entry):
    """Extract (lengthM, tonnageGt) from one vessel entry dict.
    Searches registryInfo first, then combinedSourcesInfo / selfReportedInfo.
    Returns (np.nan, np.nan) if not found."""
    entry = coerce_obj(entry)

    def scan(blocks):
        for b in blocks:
            d = coerce_obj(b)
            L = get_nested(d, ["lengthM", "length_m", "lengthMeters", "length"])
            T = get_nested(d, ["tonnageGt", "tonnage_gt", "tonnage", "grossTonnage"])
            if L is not None or T is not None:
                try:
                    L = float(L)
                except Exception:
                    L = np.nan
                try:
                    T = float(T)
                except Exception:
                    T = np.nan
                return L, T
        return None

    # registryInfo (best source: registry-matched dimensions)
    reg = get_nested(entry, ["registryInfo", "registry_info"])
    if isinstance(reg, list) and reg:
        got = scan(reg)
        if got:
            return got
    # sometimes length sits at the top level of the entry
    top = scan([entry])
    if top and not (pd.isna(top[0]) and pd.isna(top[1])):
        return top
    # other nested blocks as fallback
    for key in ["combinedSourcesInfo", "selfReportedInfo"]:
        blk = get_nested(entry, [key])
        if isinstance(blk, list) and blk:
            got = scan(blk)
            if got:
                return got
    return np.nan, np.nan


def _extract_entries(resp):
    """GFW vessel responses wrap records under 'entries' (a list of vessels).
    Return that list regardless of dict / object / DataFrame form."""
    obj = resp
    if hasattr(obj, "model_dump"):
        try:
            obj = obj.model_dump()
        except Exception:
            pass
    if isinstance(obj, dict):
        return obj.get("entries", obj.get("data", []))
    # DataFrame fallback: treat each row as an entry dict
    df = to_df(resp)
    if "entries" in df.columns:
        out = []
        for v in df["entries"]:
            cv = coerce_obj(v)
            out.append(cv if cv else v)
        return out
    return df.to_dict("records")


async def enrich_length(client, ev):
    print("[3/5] Looking up vessel length (batched) ...")
    vessel_col = find_col(ev, ["vessel"])
    flat_vid = find_col(ev, ["vessel_id", "vesselId", "_vid"])

    def extract_vid(row):
        if flat_vid and pd.notna(row.get(flat_vid)):
            return str(row[flat_vid])
        vobj = coerce_obj(row.get(vessel_col)) if vessel_col else {}
        vid = get_nested(vobj, ["id", "vessel_id", "vesselId"])
        return str(vid) if vid is not None else None

    ev = ev.copy()
    ev["_vid"] = ev.apply(extract_vid, axis=1)
    vids = ev["_vid"].dropna().unique().tolist()
    print(f"  {len(vids)} unique vessels to query in batches of {BATCH_SIZE} ...")

    length_map, tonnage_map = {}, {}

    for start in range(0, len(vids), BATCH_SIZE):
        batch = vids[start : start + BATCH_SIZE]
        try:
            # ADJUST: batch endpoint name. Common: get_vessels_by_ids(ids=[...])
            resp = await client.vessels.get_vessels_by_ids(
                ids=batch,
                # ADJUST: dataset for identity lookups
                datasets=["public-global-vessel-identity:latest"],
            )
            entries = _extract_entries(resp)
            for i, entry in enumerate(entries):
                vid = start + i
                d = coerce_obj(entry)
                L, T = _dig_length_tonnage(d)
                length_map[str(vid)] = L
                tonnage_map[str(vid)] = T
        except Exception as e:
            print(f"    batch {start // BATCH_SIZE} failed: {e}")
        print(f"    queried {min(start + BATCH_SIZE, len(vids))}/{len(vids)}")

    ev["length_m"] = ev["_vid"].map(lambda v: length_map.get(v, np.nan))
    ev["tonnage"] = ev["_vid"].map(lambda v: tonnage_map.get(v, np.nan))

    n_found = ev["length_m"].notna().sum()
    print(
        f"  length found for {n_found}/{len(ev)} event rows "
        f"({ev['_vid'].map(lambda v: v in length_map).sum()} of {len(vids)} vessels matched)."
    )
    if n_found == 0:
        print("  !! still all-NaN. Dump one raw vessel response to inspect field names:")
        print("     run a single get_vessels_by_ids call and print the JSON; look for 'lengthM'.")
    return ev


# ============================================================
# 4) Unpack position/time, classify, build keys
# ============================================================
def prep(ev):
    print("[4/5] Unpacking position/time and classifying ...")
    pos_col = find_col(ev, ["position"])
    time_col = find_col(ev, ["start", "event_start", "timestamp", "date", "time"])
    if pos_col is None or time_col is None:
        print("Columns:", list(ev.columns))
        raise SystemExit("Could not find 'position' or time column.")

    def get_lat(v):
        d = coerce_obj(v)
        val = get_nested(d, ["lat", "latitude", "y"])
        try:
            return float(val)
        except Exception:
            return np.nan

    def get_lon(v):
        d = coerce_obj(v)
        val = get_nested(d, ["lon", "lng", "longitude", "x"])
        try:
            return float(val)
        except Exception:
            return np.nan

    df = ev.copy()
    df["lat"] = df[pos_col].apply(get_lat)
    df["lon"] = df[pos_col].apply(get_lon)
    df["dt"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    df = df.dropna(subset=["lat", "lon", "dt"])
    print(f"  {len(df)} events with valid coordinates + time.")

    def tier(L):
        if pd.isna(L):
            return "dark"
        if L < CORE_LEN:
            return "core"
        if L < MAX_LEN:
            return "mid"
        return "big"

    df["tier"] = df["length_m"].apply(tier)
    df = df[df["tier"] != "big"]

    df["glat"] = (np.floor(df["lat"] / GRID_DEG) * GRID_DEG + GRID_DEG / 2).round(5)
    df["glon"] = (np.floor(df["lon"] / GRID_DEG) * GRID_DEG + GRID_DEG / 2).round(5)

    df["month"] = df["dt"].dt.strftime("%Y-%m")
    iso = df["dt"].dt.isocalendar()
    df["week"] = iso["year"].astype(str) + "-W" + iso["week"].astype(int).map("{:02d}".format)
    df["day"] = df["dt"].dt.strftime("%Y-%m-%d")
    return df


# ============================================================
# 5) Aggregate by (cell x time bucket), rank, build AOI
# ============================================================
def aggregate(df, time_key):
    piv = df.groupby(["glat", "glon", time_key, "tier"]).size().unstack(fill_value=0).reset_index()
    for t in ["core", "mid", "dark"]:
        if t not in piv.columns:
            piv[t] = 0

    piv["small_score"] = piv["core"] * W_CORE + piv["mid"] * W_MID + piv["dark"] * W_DARK
    denom = (piv["core"] + piv["mid"] + piv["dark"]).replace(0, np.nan)
    piv["core_ratio"] = piv["core"] / denom

    piv = piv[piv["core"] >= MIN_CORE_PER_ROW]
    piv = piv.sort_values(["small_score", "core_ratio"], ascending=False).reset_index(drop=True)

    top = piv.head(TOP_N).copy()
    dlat = km_to_deg_lat(AOI_HALF_KM)
    top["lat_min"] = top["glat"] - dlat
    top["lat_max"] = top["glat"] + dlat
    top["lon_min"] = top.apply(lambda r: r["glon"] - km_to_deg_lon(AOI_HALF_KM, r["glat"]), axis=1)
    top["lon_max"] = top.apply(lambda r: r["glon"] + km_to_deg_lon(AOI_HALF_KM, r["glat"]), axis=1)
    top["rank"] = range(1, len(top) + 1)
    top["aoi_wkt"] = top.apply(
        lambda r: (
            f"POLYGON(({r.lon_min:.5f} {r.lat_min:.5f}, {r.lon_max:.5f} {r.lat_min:.5f}, "
            f"{r.lon_max:.5f} {r.lat_max:.5f}, {r.lon_min:.5f} {r.lat_max:.5f}, "
            f"{r.lon_min:.5f} {r.lat_min:.5f}))"
        ),
        axis=1,
    )

    cols = [
        "rank",
        time_key,
        "glat",
        "glon",
        "core",
        "mid",
        "dark",
        "small_score",
        "core_ratio",
        "lat_min",
        "lat_max",
        "lon_min",
        "lon_max",
        "aoi_wkt",
    ]
    return top[[c for c in cols if c in top.columns]]


async def main():
    if ACCESS_TOKEN.startswith("<PASTE"):
        raise SystemExit("Set your GFW token first (top of file or GFW_API_TOKEN env var).")
    client = gfw.Client(access_token=ACCESS_TOKEN)

    eez = await lookup_eez_id(client, COUNTRY_NAME)

    raw_event_csv = f"gfw_raw_events_{eez}.csv"
    if not os.path.isfile(raw_event_csv):
        ev = await fetch_events(client, eez)
        ev.to_csv(raw_event_csv, index=False)
    else:
        ev = pd.read_csv(raw_event_csv)

    ev = await enrich_length(client, ev)
    ev.to_csv("gfw_events_with_length.csv", index=False)
    df = prep(ev)

    print("[5/5] Aggregating at month / week / day ...")
    aggregate(df, "month").to_csv(OUT_MONTH, index=False)
    aggregate(df, "week").to_csv(OUT_WEEK, index=False)
    aggregate(df, "day").to_csv(OUT_DAY, index=False)

    print("\n==================== DONE ====================")
    print(f"month -> {OUT_MONTH}\nweek  -> {OUT_WEEK}\nday   -> {OUT_DAY}")
    print(
        "\nEach row = one purchase site x one time bucket. "
        "core=<10m count, dark=length-missing (dark-vessel candidate)."
    )
    print("Take aoi_wkt + the time bucket to an imagery archive and pick the lowest-cloud scene in that window.")
    print("\nTop 10 by month:")
    print(aggregate(df, "month").head(10).to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())
