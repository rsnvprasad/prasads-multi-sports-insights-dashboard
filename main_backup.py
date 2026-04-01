# v9: Steps tab styling + sharper globe + native-font story card
import io
import math
import datetime as dt
current_year = dt.date.today().year

def _resolve_year_slice(all_df, year_selection: str, year_col: str = "Year"):
    """Return (label, df_slice). If year_selection is 'All', slice == all_df."""
    if year_selection == "All":
        return "All Years", all_df
    try:
        y = int(year_selection)
    except Exception:
        return "All Years", all_df
    if all_df is None or all_df.empty or (year_col not in all_df.columns):
        return str(y), all_df
    return str(y), all_df[all_df[year_col] == y].copy()

import json
import base64
from pathlib import Path

import pandas as pd
import numpy as np

import streamlit as st
import plotly.express as px
import streamlit.components.v1 as components
import requests
from textwrap import dedent
from PIL import Image

import time

def _html(s: str) -> str:
    """Dedent + strip + remove ALL leading whitespace per line (prevents Markdown code blocks)."""
    s = dedent(s).strip("\n")
    return "\n".join(line.lstrip() for line in s.splitlines())



# =========================
# Small utils
# =========================
def _robust_parse_datetime(val: str) -> pd.Timestamp:
    if pd.isna(val):
        return pd.NaT
    s = str(val).strip()

    try:
        return pd.to_datetime(s, errors="raise", infer_datetime_format=True, utc=False)
    except Exception:
        pass

    for fmt in (
        "%b %d, %Y, %I:%M:%S %p",
        "%b %d, %Y %I:%M:%S %p",
        "%Y-%m-%d %H:%M:%S",
        "%d-%m-%Y %H:%M",
    ):
        try:
            return pd.to_datetime(s, format=fmt, errors="raise", utc=False)
        except Exception:
            continue

    return pd.to_datetime(s, errors="coerce", dayfirst=True, utc=False)


def _to_numeric_clean(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace("\u2009", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _img_to_data_uri(img_path: Path) -> str:
    """
    Returns a data URI for an image file. Auto-detects mime by extension.
    Supports .jpg/.jpeg and .png (common for globe textures).
    """
    ext = img_path.suffix.lower()
    if ext in [".png"]:
        mime = "image/png"
    else:
        mime = "image/jpeg"
    b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"



# =========================
# Caching wrappers
# =========================
@st.cache_data(show_spinner=False)
def load_csv_cached(file) -> pd.DataFrame:
    df = pd.read_csv(file, dtype=str, low_memory=False, encoding="utf-8")
    return df


@st.cache_data(show_spinner=False)
def parse_activities_cached(df: pd.DataFrame, v: int = 3) -> pd.DataFrame:
    return parse_activities(df)


@st.cache_data(show_spinner=False)
def _steps_summary_cache_key(path: Path) -> float:
    # use high precision so cache invalidates even if file is rewritten within the same second
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def load_steps_summary(path_str: str, cache_key: float) -> dict | None:
    """
    Load the Garmin steps summary JSON from a given path.
    cache_key is used only to invalidate cache when the file changes.
    """
    _ = cache_key  # cache invalidation key (intentionally unused)

    path = Path(path_str)
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None



# =========================
# Strava API helpers
# =========================
@st.cache_data(show_spinner=False, ttl=900)
def get_strava_access_token() -> str:
    cfg = st.secrets["strava"]
    resp = requests.post(
        "https://www.strava.com/oauth/token",
        data={
            "client_id": cfg["client_id"],
            "client_secret": cfg["client_secret"],
            "grant_type": "refresh_token",
            "refresh_token": cfg["refresh_token"],
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


@st.cache_data(show_spinner=True, ttl=60)
def fetch_strava_activities_json(
    before: int | None = None,
    after: int | None = None,
    max_pages: int = 10,
    per_page: int = 200,
    cache_buster: int | None = None,  # ✅ Added
) -> list[dict]:

    # cache_buster is intentionally unused.
    # It forces Streamlit to treat each refresh as a new call.
    _ = cache_buster

    access_token = get_strava_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}

    all_acts: list[dict] = []
    page = 1
    while page <= max_pages:
        params = {"page": page, "per_page": per_page}
        if before is not None:
            params["before"] = before
        if after is not None:
            params["after"] = after

        r = requests.get(
            "https://www.strava.com/api/v3/athlete/activities",
            headers=headers,
            params=params,
            timeout=15,
        )
        r.raise_for_status()
        acts = r.json()
        if not acts:
            break

        all_acts.extend(acts)
        if len(acts) < per_page:
            break
        page += 1

    return all_acts


def strava_json_to_dataframe(activities: list[dict]) -> pd.DataFrame:
    if not activities:
        return pd.DataFrame(
            columns=[
                "Activity ID",
                "Activity Date",
                "Activity Name",
                "Activity Type",
                "Distance",
                "Moving Time",
                "Elapsed Time",
                "Average Speed",
                "Max Speed",
                "Elevation Gain",
                "Calories",
                "Commute",
                "Carbon Saved",
            ]
        )

    rows = []
    for a in activities:
        aid = a.get("id")
        name = a.get("name")
        a_type = a.get("sport_type") or a.get("type")

        dist_km = (a.get("distance") or 0.0) / 1000.0
        mov_s = a.get("moving_time") or 0
        elap_s = a.get("elapsed_time") or 0

        avg_kmh = (a.get("average_speed") or 0.0) * 3.6
        max_kmh = (a.get("max_speed") or 0.0) * 3.6

        elev_m = a.get("total_elevation_gain") or 0.0

        calories = a.get("calories")
        if calories is None and a.get("kilojoules") is not None:
            calories = a["kilojoules"] / 4.184

        commute = bool(a.get("commute", False))
        start_date = a.get("start_date_local") or a.get("start_date")

        carbon_saved = dist_km * 0.21 if commute else 0.0

        rows.append(
            {
                "Activity ID": aid,
                "Activity Date": start_date,
                "Activity Name": name,
                "Activity Type": a_type,
                "Distance": dist_km,
                "Moving Time": mov_s,
                "Elapsed Time": elap_s,
                "Average Speed": avg_kmh,
                "Max Speed": max_kmh,
                "Elevation Gain": elev_m,
                "Calories": calories,
                "Commute": commute,
                "Carbon Saved": carbon_saved,
            }
        )

    return pd.DataFrame(rows)

# =========================
# Strava local cache (Parquet)
# =========================
STRAVA_PARQUET_PATH = Path("App_Data") / "strava_activities.parquet"
STRAVA_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)

def _load_strava_parquet() -> pd.DataFrame | None:
    """Load cached Strava activities from parquet, if available."""
    try:
        if STRAVA_PARQUET_PATH.exists():
            return pd.read_parquet(STRAVA_PARQUET_PATH)
    except Exception:
        return None
    return None

def _save_strava_parquet(df: pd.DataFrame) -> None:
    """Save Strava activities dataframe to parquet (best-effort)."""
    try:
        if df is not None and not df.empty:
            df.to_parquet(STRAVA_PARQUET_PATH, index=False)
    except Exception:
        # If parquet engine missing (pyarrow), we silently skip
        pass

# =========================
# Cache freshness helpers
# =========================
CACHE_MAX_AGE_HOURS = 6

def _parquet_cache_age_hours() -> float | None:
    """
    Returns cache age in hours if parquet exists, else None.
    """
    try:
        if STRAVA_PARQUET_PATH.exists():
            mtime = STRAVA_PARQUET_PATH.stat().st_mtime  # seconds
            age_sec = (time.time() - mtime)
            return age_sec / 3600.0
    except Exception:
        return None
    return None

def _parquet_is_stale(max_age_hours: float = CACHE_MAX_AGE_HOURS) -> bool:
    age = _parquet_cache_age_hours()
    if age is None:
        # no cache file (or error) => treat as stale so we fetch once
        return True
    return age > max_age_hours

# =========================
# App Config
# =========================
st.set_page_config(page_title="Strava Insights – Multi-Sport", layout="wide")

st.markdown(
    """
<style>

/* ===== Section Divider (reliable) ===== */
.soft-divider{
  height: 1px;
  width: 100%;
  background: rgba(255,255,255,0.14);
  margin: 18px 0 14px;
  border-radius: 1px;
}

@media (prefers-color-scheme: light){
  .soft-divider{
    background: rgba(49,51,63,0.18);
  }
}

.quote-box{
  margin-top: 12px;
  padding: 10px 12px;
  border-radius: 12px;
  border-left: 4px solid rgba(245, 158, 11, 0.85);
  background: rgba(245, 158, 11, 0.08);

  /* ⭐ Premium finishing touch */
  box-shadow: 0 8px 22px rgba(0,0,0,0.25);
}

/* ===== Number highlight (global) ===== */
.num-highlight{
  font-weight: 800;
  color: #f59e0b;
  background: rgba(245, 158, 11, 0.12);
  padding: 2px 6px;
  border-radius: 6px;
}

.quote-text{
  font-size: 1.0rem;
  font-weight: 800;
  line-height: 1.35;
}
.quote-by{
  margin-top: 6px;
  font-size: 0.85rem;
  opacity: 0.75;
}
@media (prefers-color-scheme: dark) {
  .quote-box{
    background: rgba(245, 158, 11, 0.10);
    border-left-color: rgba(245, 158, 11, 0.90);
  }
}


/* Steps tab: KPI tiles (Apple Health-ish) */
.steps-kpi-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin-top: 12px;
}

.steps-kpi-card {
  border: 1px solid rgba(49, 51, 63, 0.12);
  border-radius: 14px;
  padding: 12px 14px;
  background: rgba(49, 51, 63, 0.04);
}

.steps-kpi-card .label {
  font-size: 0.82rem;
  color: #6b7280;
  margin-bottom: 4px;
}

.steps-kpi-card .value {
  font-size: 1.55rem;
  font-weight: 800;
  line-height: 1.05;
}

.steps-kpi-card .unit {
  font-size: 0.85rem;
  color: #6b7280;
  margin-left: 6px;
  font-weight: 600;
}

.steps-kpi-card .sub {
  font-size: 0.78rem;
  color: #6b7280;
  margin-top: 6px;
}

/* KPI Section Frame (for KPI rows in all tabs) */
/* KPI Section Frame (clean, no box) */
.kpi-frame{
  border: none !important;
  background: transparent !important;
  border-radius: 0 !important;
  padding: 0 !important;
  margin: 12px 0 16px 0;
}

/* ===== KPI Hover Premium Effect ===== */
.kpi-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px rgba(255,255,255,0.08);
  transition: all 0.2s ease;
}

[data-testid="stMetricLabel"] > div {
  font-size: 0.83rem;
  color: #6b7280;
  margin-bottom: -6px;
}
[data-testid="stMetricValue"] {
  margin-top: -10px !important;
}
section.main > div:first-child { padding-top: 0.4rem; }
hr { margin-top: 0.6rem !important; margin-bottom: 0.6rem !important; }
/* Steps tab: story card styling (match Streamlit look & feel) */
.steps-story {
  border: 1px solid rgba(49, 51, 63, 0.12);
  border-radius: 14px;
  padding: 14px 16px;
  background: rgba(49, 51, 63, 0.04);
}
.steps-story .kicker {
  font-size: 0.85rem;
  color: #6b7280;
  margin-bottom: 0.25rem;
}
.steps-story .headline {
  font-size: 1.65rem;
  font-weight: 800;
  line-height: 1.15;
  margin: 0.15rem 0 0.75rem 0;
}
.steps-story .body {
  font-size: 0.95rem;
  line-height: 1.65;
}
.steps-tip {
  margin-top: 0.9rem;
  padding: 10px 12px;
  border-radius: 12px;
  background: rgba(49, 51, 63, 0.04);
  border: 1px solid rgba(49, 51, 63, 0.08);
}
.steps-tip .t1 {
  font-size: 0.8rem;
  color: #6b7280;
  margin-bottom: 2px;
}
.steps-tip .t2 {
  font-size: 0.85rem;
}

/* Progress ring */
.steps-ring-card {
  border: 1px solid rgba(49, 51, 63, 0.12);
  border-radius: 14px;
  padding: 12px 14px;
  background: rgba(49, 51, 63, 0.04);
  margin-top: 12px;
}

.steps-ring-wrap {
  display: flex;
  align-items: center;
  gap: 14px;
}

.steps-ring {
  width: 86px;
  height: 86px;
  border-radius: 50%;
  position: relative;
  display: grid;
  place-items: center;
}

.steps-ring::before {
  content: "";
  width: 68px;
  height: 68px;
  border-radius: 50%;
  background: rgba(255,255,255,0.85);
  border: 1px solid rgba(49, 51, 63, 0.08);
  z-index: 1;
}

.steps-ring-text {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  font-weight: 800;
  font-size: 1.05rem;
  z-index: 2;
}

.steps-ring-sub {
  font-size: 0.82rem;
  color: #6b7280;
  margin-top: 2px;
}

/* --- Dark mode safety: avoid hard-coded light backgrounds/text --- */
@media (prefers-color-scheme: dark) {
  .steps-kpi-card,
  .steps-story,
  .steps-ring-card {
    background: rgba(255,255,255,0.03) !important;
    border-color: rgba(255,255,255,0.12) !important;
  }

  .steps-kpi-card .label,
  .steps-kpi-card .unit,
  .steps-kpi-card .sub,
  .steps-story .kicker,
  [data-testid="stMetricLabel"] > div,
  .steps-ring-sub,
  .steps-tip .t1 {
    color: rgba(255,255,255,0.70) !important;
  }

  .steps-tip {
    background: rgba(255,255,255,0.06) !important;
    border-color: rgba(255,255,255,0.10) !important;
  }


  .steps-ring::before {
    background: rgba(0,0,0,0.55) !important;
    border-color: rgba(255,255,255,0.12) !important;
  }


  /* ===== Number Highlight Style ===== */
  .num-highlight {
      font-weight: 800;
      color: #f59e0b;
      background: rgba(245, 158, 11, 0.12);
      padding: 2px 6px;
      border-radius: 6px;
  }
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Prasad's multi-sports (🚴‍♂️,🏃,🚶,🏊) stats insights dashboard")
st.caption("Data sources: Strava (activities) + Garmin (steps Summary)")


# =========================
# Helper functions
# =========================
SPORT_ALIASES = {
    "Cycling": [
        "ride",
        "virtual ride",
        "gravel ride",
        "mountain bike ride",
        "e-bike ride",
        "velomobile",
        "handcycle",
        "indoor cycling",
    ],
    "Running": ["run", "trail run"],
    "Walking": ["walk", "hike", "trek"],
    "Swimming": ["swim"],
    "Workout": ["workout", "yoga", "weight training", "crossfit"],
    "Other": [],
}

def soft_divider():
    st.markdown('<div class="soft-divider"></div>', unsafe_allow_html=True)

def compute_activity_day_stats(data: pd.DataFrame):
    """
    Returns:
    - total_active_days: number of unique days with at least one activity
    - longest_streak: longest consecutive-day activity streak
    """

    if data.empty:
        return 0, 0

    # Convert Activity Date to pure date (drop time)
    days = pd.to_datetime(data["Activity Date"], errors="coerce").dt.date.dropna()

    unique_days = sorted(set(days))
    total_active_days = len(unique_days)

    # Compute longest consecutive streak
    longest_streak = 1
    current_streak = 1

    for i in range(1, len(unique_days)):
        if (unique_days[i] - unique_days[i - 1]).days == 1:
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
        else:
            current_streak = 1

    return total_active_days, longest_streak



def canonical_sport(activity_type: str) -> str:
    if not isinstance(activity_type, str):
        return "Other"
    t = activity_type.strip().lower()
    for sport, aliases in SPORT_ALIASES.items():
        for alias in aliases:
            if alias in t:
                return sport
    if "bike" in t:
        return "Cycling"
    if "run" in t:
        return "Running"
    if "walk" in t or "hike" in t or "trek" in t:
        return "Walking"
    if "swim" in t:
        return "Swimming"
    if "yoga" in t or "workout" in t or "training" in t:
        return "Workout"
    return "Other"


def parse_activities(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    expected = [
        "Activity ID",
        "Activity Date",
        "Activity Name",
        "Activity Type",
        "Distance",
        "Moving Time",
        "Elapsed Time",
        "Average Speed",
        "Max Speed",
        "Elevation Gain",
        "Calories",
        "Commute",
        "Carbon Saved",
    ]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan

    df["Activity Date"] = df["Activity Date"].apply(_robust_parse_datetime)

    for c in ["Distance", "Average Speed", "Max Speed", "Elevation Gain", "Calories", "Carbon Saved"]:
        df[c] = _to_numeric_clean(df[c])

    for c in ["Moving Time", "Elapsed Time"]:
        df[c] = _to_numeric_clean(df[c])

    alt_distance_col = None
    for c in df.columns:
        if c.lower().startswith("distance") and c != "Distance":
            alt_distance_col = c
            break
    if alt_distance_col is not None:
        alt_num = _to_numeric_clean(df[alt_distance_col])
        if alt_num.dropna().median() > 1000:
            alt_num = alt_num / 1000.0
        df["Distance"] = df["Distance"].fillna(alt_num)

    if df["Commute"].dtype != bool:
        df["Commute"] = df["Commute"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])

    df["Sport"] = df["Activity Type"].astype(str).apply(canonical_sport)

    df["Year"] = df["Activity Date"].dt.year
    df["Month"] = df["Activity Date"].dt.to_period("M").astype(str)
    df["MonthPeriod"] = df["Activity Date"].dt.to_period("M")
    df["Week"] = df["Activity Date"].dt.to_period("W").astype(str)

    return df

def build_cycling_lifestyle_story(cyc_all: pd.DataFrame, cyc_view: pd.DataFrame) -> dict:
    """
    Returns a dict with:
      - headline: str
      - body_html: str (HTML-ready body lines)
      - footer: str (small note)
    Story is evergreen: works even if last ride was months ago (weather/winter).
    """

    if cyc_all is None or cyc_all.empty:
        return {
            "headline": "Cycling story is waiting…",
            "body_html": "No cycling activities found in the currently loaded Strava range.",
            "footer": "Tip: click ‘Fetch latest from Strava’ or verify Strava API credentials.",
        }

    # --- Overall (within fetched Strava range) ---
    total_rides = int(len(cyc_all))
    total_km = float(cyc_all["Distance"].sum() or 0.0)
    # 🌍🌕 Globe-style comparisons for cycling
    earth_circum_km = 40075.0
    earth_rounds = total_km / earth_circum_km if total_km > 0 else 0.0

    moon_distance_km = 384400.0
    moon_pct = (total_km / moon_distance_km * 100.0) if total_km > 0 else 0.0

    total_hours = float((cyc_all["Moving Time"].sum() or 0.0) / 3600.0)
    total_cal = float(cyc_all["Calories"].sum() or 0.0)

    comm = cyc_all[cyc_all["Commute"] == True]
    nonc = cyc_all[cyc_all["Commute"] == False]

    comm_km = float(comm["Distance"].sum() or 0.0)
    nonc_km = float(nonc["Distance"].sum() or 0.0)
    comm_rides = int(len(comm))
    nonc_rides = int(len(nonc))

    comm_share = (comm_km / total_km * 100.0) if total_km > 0 else 0.0

    # Carbon Saved is already computed in my Strava transform (for commute rides).
    carbon_kg = float(cyc_all["Carbon Saved"].sum() or 0.0)
    # 🌳 Trees equivalent impact (1 tree ≈ 22 kg CO₂ absorbed/year)
    trees_equiv = carbon_kg / 22 if carbon_kg > 0 else 0
    # Cleaner display: avoid showing "planting 0 trees"
    trees_line = "" if trees_equiv <= 0 else f"(≈ planting <b>{trees_equiv:,.0f} trees</b>) 🌳"


    # 🏆 Ride achievements: Half-century & Century rides
    half_century = cyc_all[cyc_all["Distance"] >= 50]
    century = cyc_all[cyc_all["Distance"] >= 100]

    half_century_count = int(len(half_century))
    century_count = int(len(century))

    best_ride = float(cyc_all["Distance"].max() or 0.0)
    
    # 📅 Milestones: Most active year + Top 3 longest rides
    cyc_all["Year"] = pd.to_datetime(cyc_all["Activity Date"], errors="coerce").dt.year

    # Most active year by distance
    yearly_km = cyc_all.groupby("Year")["Distance"].sum().sort_values(ascending=False)
    top_year = int(yearly_km.index[0]) if not yearly_km.empty else None
    top_year_km = float(yearly_km.iloc[0]) if not yearly_km.empty else 0.0

    # Top 3 longest rides
    top3_rides = cyc_all["Distance"].sort_values(ascending=False).head(3).tolist()



    # Last ride recency (timezone-safe: force UTC for both sides)
    last_dt = pd.to_datetime(cyc_all["Activity Date"], errors="coerce", utc=True).max()
    today = pd.Timestamp.now(tz="UTC")

    if pd.notna(last_dt):
        days_since = int((today.normalize() - last_dt.normalize()).days)
        months_since = int(round(days_since / 30.0))
    else:
        days_since = None
        months_since = None


    # --- Current view (after filters) ---
    view_km = float(cyc_view["Distance"].sum() or 0.0) if (cyc_view is not None and not cyc_view.empty) else 0.0
    view_rides = int(len(cyc_view)) if (cyc_view is not None and not cyc_view.empty) else 0

    # Choose headline based on commute emphasis + inactivity
    if months_since is not None and months_since >= 2:
        headline = "A season of pause — my cycling story still stands strong"
        recency_line = f"• Last recorded ride: <b>{last_dt.date()}</b> (~{months_since} months ago). Weather seasons change, but not the cyclist mindset, my bike always looks at me to go for a ride. Ha ha :-)."
    else:
        if comm_share >= 50:
            headline = "I didn’t just ride — I chose a cleaner way to move"
        else:
            headline = "Cycling as a lifestyle — strength, freedom, and wellbeing"
        recency_line = f"• Last recorded ride: <b>{last_dt.date()}</b>." if pd.notna(last_dt) else "Last recorded ride: <b>—</b>."

    badge_line = ""
    if comm_share >= 60:
        badge_line = "🚴 <b>Commute Champion of self:</b> Most of my rides powered daily life, not just workouts.<br/>"

    ##
    # Helper: highlight numbers in story text
    def nh(s: str) -> str:
        return f"<span class='num-highlight'>{s}</span>"

    total_km_s   = f"{total_km:,.0f} km"
    total_rides_s= f"{total_rides:,}"
    earth_rounds_s = f"{earth_rounds:,.2f}"
    moon_pct_s   = f"{moon_pct:,.1f}%"

    comm_km_s    = f"{comm_km:,.0f} km"
    comm_rides_s = f"{comm_rides:,}"
    nonc_km_s    = f"{nonc_km:,.0f} km"
    nonc_rides_s = f"{nonc_rides:,}"

    comm_share_s = f"{comm_share:,.0f}%"
    carbon_s     = f"{carbon_kg:,.1f} kg CO₂"

    half_century_s = f"{half_century_count:,}"
    century_s      = f"{century_count:,}"
    best_ride_s    = f"{best_ride:,.0f} km"
    top_year_km_s  = f"{top_year_km:,.0f} km"
    total_hours_s  = f"{total_hours:,.0f} hrs"

    top3_s = ", ".join([f"{x:.0f} km" for x in top3_rides])
    ##

    body_html = (
        badge_line +
        f"• Overall mileage: {nh(total_km_s)} across {nh(total_rides_s)} rides<br/>"
        f"🌍 <b>Earth rounds:</b> {nh(earth_rounds_s)} times around Earth<br/>"
        f"🌕 <b>Earth → Moon journey:</b> {nh(moon_pct_s)} completed<br/>"

        f"• Commute: {nh(comm_km_s)} ({nh(comm_rides_s)} rides)<br/>"
        f"• Non-commute: {nh(nonc_km_s)} ({nh(nonc_rides_s)} rides)<br/>"

        f"• Commute share is {nh(comm_share_s)} of my cycling distance. "
        f"Since, I believe that, where there is a will, there is a way. Super proud of my commutes<br/>"

        f"• Estimated carbon saved is {nh(carbon_s)} {trees_line}. "
        f"When we cannot create our nature, then we can only conserve it. Hence, doing my bit to conserve our mother nature and preserving for our next generations.<br/><br/>"

        f"🏆 <b>Half-century rides (50+ km):</b> {nh(half_century_s)}<br/>"
        f"🏅 <b>Century rides (100+ km):</b> {nh(century_s)}<br/>"
        f"🚀 <b>Best ride:</b> {nh(best_ride_s)}<br/>"
        f"🏅 <b>Top 3 rides:</b> {nh(top3_s)}<br/>"
        f"📅 <b>Most active year:</b> {top_year} ({nh(top_year_km_s)})<br/>"
        f"⏱️ <b>Total time on bike saddle:</b> {nh(total_hours_s)}<br/><br/>"
        f"✨ That’s thousands of hours invested into health, discipline, and freedom.<br/><br/>"
        f"{recency_line}<br/>"
    )


    comeback_line = ""
    if months_since is not None and months_since >= 4:
        comeback_line = (
            "<br/>💫 <b>Comeback Season:</b> My bike & the road are waiting for the next memorable rides… "
            "my next chapter will be even stronger."
        )

    # Append comeback line into body
    body_html += comeback_line

    footer = "Note: ‘Carbon saved’ is an estimate derived from commute distance and a simple per-km factor."

    return {"headline": headline, "body_html": body_html, "footer": footer}

from contextlib import contextmanager

@contextmanager
def kpi_frame():
    st.markdown("<div class='kpi-frame'>", unsafe_allow_html=True)
    yield
    st.markdown("</div>", unsafe_allow_html=True)

def kpi_card(label, value, help_text=None):
    # Plain metric (no box around each KPI)
    st.metric(label, value, help=help_text)


def generic_time_series(
    df: pd.DataFrame,
    sport_name: str,
    key_prefix: str | None = None, *,
    show_monthly: bool = True,
    show_distance: bool = True,
    show_elevation: bool = True,
    granularity: str = "month",   # "month" or "year"
):
    if df is None or df.empty:
        st.warning(f"No data to plot for {sport_name}.")
        return

    df = df.copy()
    base_key = (key_prefix or sport_name).lower().replace(" ", "_")

    # Ensure datetime
    if "Activity Date" in df.columns:
        df["Activity Date"] = pd.to_datetime(df["Activity Date"], errors="coerce")

    # -------------------------
    # YEARLY aggregation
    # -------------------------
    if granularity.lower() == "year":
        if "Year" not in df.columns:
            df["Year"] = df["Activity Date"].dt.year

        yearly = (
            df.groupby("Year")
              .agg(
                  Rides=("Activity ID", "count"),
                  Distance_km=("Distance", "sum"),
                  Elevation_m=("Elevation Gain", "sum"),
              )
              .reset_index()
              .sort_values("Year")
        )

        if show_monthly:
            fig1 = px.bar(
                yearly, x="Year", y="Rides",
                title=f"{sport_name} – Yearly activities count",
                labels={"Rides": "Count", "Year": "Year"},
            )
            fig1.update_xaxes(title_text="Year", type="category")
            st.plotly_chart(fig1, use_container_width=True, key=f"{base_key}_yearly_count")

        if show_distance:
            fig2 = px.line(
                yearly, x="Year", y="Distance_km",
                title=f"{sport_name} – Distance over time (km) (Yearly)",
                labels={"Distance_km": "Distance (km)", "Year": "Year"},
            )
            fig2.update_xaxes(title_text="Year", type="category")
            st.plotly_chart(fig2, use_container_width=True, key=f"{base_key}_yearly_distance")

        if show_elevation:
            fig3 = px.line(
                yearly, x="Year", y="Elevation_m",
                title=f"{sport_name} – Elevation Gain over time (m) (Yearly)",
                labels={"Elevation_m": "Elevation (m)", "Year": "Year"},
            )
            fig3.update_xaxes(title_text="Year", type="category")
            st.plotly_chart(fig3, use_container_width=True, key=f"{base_key}_yearly_elevation")

        return

    # -------------------------
    # MONTHLY aggregation (fill missing months)
    # -------------------------
    if "MonthPeriod" not in df.columns:
        df["MonthPeriod"] = pd.to_datetime(df["Activity Date"], errors="coerce").dt.to_period("M")

    monthly = (
        df.groupby("MonthPeriod")
          .agg(
              Rides=("Activity ID", "count"),
              Distance_km=("Distance", "sum"),
              Elevation_m=("Elevation Gain", "sum"),
          )
          .reset_index()
          .sort_values("MonthPeriod")
    )

    if monthly.empty:
        st.info(f"No monthly data available for {sport_name}.")
        return

    # Fill missing months
    full_periods = pd.period_range(monthly["MonthPeriod"].min(), monthly["MonthPeriod"].max(), freq="M")
    monthly = (
        monthly.set_index("MonthPeriod")
              .reindex(full_periods, fill_value=0)
              .rename_axis("MonthPeriod")
              .reset_index()
    )

    # Use datetime for Plotly date axis
    monthly["MonthDate"] = monthly["MonthPeriod"].dt.to_timestamp()

    # ---- Monthly activities count ----
    if show_monthly:
        fig1 = px.bar(
            monthly, x="MonthDate", y="Rides",
            title=f"{sport_name} – Monthly activities count",
            labels={"Rides": "Count", "MonthDate": "Month"},
        )
        fig1.update_xaxes(
            title_text="Month",
            type="date",
            dtick="M2",
            tickformat="%b\n%Y",
        )
        st.plotly_chart(fig1, use_container_width=True, key=f"{base_key}_monthly_count")

    # ---- Distance over time ----
    if show_distance:
        fig2 = px.line(
            monthly, x="MonthDate", y="Distance_km",
            title=f"{sport_name} – Distance over time (km)",
            labels={"Distance_km": "Distance (km)", "MonthDate": "Month"},
        )
        fig2.update_xaxes(title_text="Month", type="date", dtick="M2", tickformat="%b\n%Y")
        st.plotly_chart(fig2, use_container_width=True, key=f"{base_key}_distance_over_time")

    # ---- Elevation over time ----
    if show_elevation:
        fig3 = px.line(
            monthly, x="MonthDate", y="Elevation_m",
            title=f"{sport_name} – Elevation Gain covered over time (m)",
            labels={"Elevation_m": "Elevation (m)", "MonthDate": "Month"},
        )
        fig3.update_xaxes(title_text="Month", type="date", dtick="M2", tickformat="%b\n%Y")
        st.plotly_chart(fig3, use_container_width=True, key=f"{base_key}_elevation_over_time")

        return

   




def distance_distribution(df: pd.DataFrame, sport_name: str):
    if df.empty:
        return

    base_key = sport_name.lower().replace(" ", "_")

    # ✅ Clean distance: numeric + remove negatives
    dfx = df.copy()
    dfx["Distance"] = pd.to_numeric(dfx["Distance"], errors="coerce")
    dfx = dfx[dfx["Distance"].notna()]
    dfx["Distance"] = dfx["Distance"].clip(lower=0)

    if dfx.empty:
        st.info(f"No valid distance data for {sport_name}.")
        return

    max_x = float(dfx["Distance"].max())

    fig = px.histogram(
        dfx,
        x="Distance",
        nbins=40,
        title=f"{sport_name} – Distance Distribution over number of activities (km)",
        labels={"Distance": "Distance (km)", "count": "Number of activities"},
    )

    # ✅ Make axis nice + prevent negative range
    fig.update_xaxes(
        range=[0, max_x],
        tickmode="linear",
        dtick=5,
    )

    st.plotly_chart(fig, use_container_width=True, key=f"{base_key}_distance_hist")


def commute_split(df: pd.DataFrame, sport_name: str):
    if df.empty:
        return
    base_key = sport_name.lower().replace(" ", "_")

    agg = (
        df.groupby("Commute")
          .agg(Count=("Activity ID", "count"), Distance_km=("Distance", "sum"))
          .reset_index()
    )
    agg["Type"] = np.where(agg["Commute"], "Commute", "Non-Commute")

    fig = px.bar(
        agg,
        x="Type",
        y="Count",
        title=f"{sport_name} – Activties between Commute vs Non-Commute (Count)",
        labels={"Count": "Number of rides", "Type": "Type of ride"},
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{base_key}_commute_count")

    fig2 = px.bar(
        agg,
        x="Type",
        y="Distance_km",
        title=f"{sport_name} – Distance covered in Commute vs Non-Commute (km)",
        labels={"Distance_km": "Distance (Km)", "Type": "Type of ride"},
    )
    st.plotly_chart(fig2, use_container_width=True, key=f"{base_key}_commute_distance")




def download_button(df: pd.DataFrame, label: str, filename: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, data=csv, file_name=filename, mime="text/csv")


def render_cycling_globe_plotly(total_km: float):
    import plotly.graph_objects as go

    earth_circum_km = 40075.0
    earth_rounds = (total_km / earth_circum_km) if total_km > 0 else 0.0
    lap_pct = int(round((earth_rounds % 1.0) * 100)) if total_km > 0 else 0

    # Progress arc around the equator
    n = 120
    end_lon = (lap_pct / 100.0) * 360.0
    lons = [i * (end_lon / (n - 1)) for i in range(n)]
    lats = [0.0] * n

    fig = go.Figure()

    # Arc line
    fig.add_trace(
        go.Scattergeo(
            lon=lons,
            lat=lats,
            mode="lines",
            line=dict(width=6),
        )
    )

    # Bike marker at arc end
    fig.add_trace(
        go.Scattergeo(
            lon=[end_lon],
            lat=[0],
            mode="markers+text",
            text=["🚴"],
            textposition="top center",
            marker=dict(size=12),
        )
    )

    fig.update_geos(
        projection_type="orthographic",
        showcountries=True,
        showcoastlines=True,
        showland=True,
        showocean=True,
    )

    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=50, b=0),
        title=f"🌍 Cycling Earth Lap: {lap_pct}% of current round",
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, key="cycling_globe_plotly")


# =========================
# Globe (offline assets) – PATH A
# =========================
def _find_vendor_dir() -> Path:
    """
    Finds App_Data/vendor reliably even if you run streamlit from different working dirs.
    """
    here = Path(__file__).resolve().parent  # App_Data
    candidates = [
        here / "vendor",  # App_Data/vendor (most common)
        here.parent / "App_Data" / "vendor",  # WIP/App_Data/vendor (if file moved)
        Path.cwd() / "App_Data" / "vendor",  # run from WIP
        Path.cwd() / "vendor",  # run from App_Data
    ]

    # Also try climbing parents (in case you run from elsewhere)
    cur = Path.cwd().resolve()
    for _ in range(5):
        candidates.append(cur / "App_Data" / "vendor")
        candidates.append(cur / "vendor")
        cur = cur.parent

    for p in candidates:
        if p.exists() and p.is_dir():
            return p.resolve()

    # fallback (so debug shows something meaningful)
    return (here / "vendor").resolve()


def render_globe(height: int = 520, altitude: float = 1.4):
    """
    Render ONLY the globe (HTML component). The story text is rendered using native
    Streamlit so fonts match the rest of the app.
    """
    vendor = _find_vendor_dir()

    three_js_path = vendor / "three.min.js"
    globe_js_path = vendor / "globe.gl.min.js"
    earth_img_path = vendor / "earth.jpg"

    missing = [p.name for p in (three_js_path, globe_js_path, earth_img_path) if not p.exists()]
    if missing:
        st.warning(f"Missing local globe assets: {', '.join(missing)}. Expected in: {vendor}")
        st.caption(f"DEBUG: vendor resolved to: {vendor}")
        return

    three_js = _read_text(three_js_path)
    globe_js = _read_text(globe_js_path)
    earth_uri = _img_to_data_uri(earth_img_path)

    html = f"""
<div style="border-radius:14px; border:1px solid rgba(49,51,63,0.12); padding:10px; background:rgba(49,51,63,0.04);">
  <div id="globeWrap" style="width:100%; height:{height}px; position:relative;">
    <div id="globeViz" style="width:100%; height:100%;"></div>
  </div>
</div>

<script>
{three_js}
</script>

<script>
{globe_js}
</script>

<script>
(function() {{
  const container = document.getElementById('globeViz');
  if (!container) return;

  const globe = Globe()(container)
    .globeImageUrl("{earth_uri}")
    .backgroundColor("rgba(0,0,0,0)")
    .showAtmosphere(true)
    .atmosphereColor("lightskyblue")
    .atmosphereAltitude(0.15);

  // Improve crispness on HiDPI screens
  try {{
    const r = globe.renderer();
    const dpr = Math.min((window.devicePixelRatio || 1), 2);
    r.setPixelRatio(dpr);
  }} catch (e) {{}}

  // initial viewpoint
  globe.pointOfView({{ lat: 20, lng: 0, altitude: {altitude:.2f} }});

  // a small “marker” point as a visual anchor
  const points = [{{ lat: 12, lng: 77, size: 0.8, color: "orange" }}];
  globe.pointsData(points)
       .pointLat(d => d.lat)
       .pointLng(d => d.lng)
       .pointColor(d => d.color)
       .pointAltitude(0.02)
       .pointRadius(d => d.size);

  // Auto-rotate slowly
  const controls = globe.controls();
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.25;
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;

  // Resize handling
  function resize() {{
    const w = container.clientWidth;
    const h = container.clientHeight;
    globe.width([w]);
    globe.height([h]);
    try {{
      globe.renderer().setSize(w, h, false);
    }} catch (e) {{}}
  }}
  resize();
  new ResizeObserver(resize).observe(container);

  // Double-click reset
  container.addEventListener("dblclick", () => {{
    globe.pointOfView({{ lat: 20, lng: 0, altitude: {altitude:.2f} }}, 800);
  }});
}})();
</script>
"""
    components.html(html, height=height + 30, scrolling=False)

def render_earth_moon_journey(
    *,
    moon_pct: float,
    total_km: float,
    height: int = 520,
):
    """
    Cycling visual: Earth → Moon journey progress (no external images needed).
    Shows Earth, Moon, a dotted travel path, a glowing progress segment,
    a checkpoint marker at the progress end, and a 🚴 locked at the same end point.
    """

    # -------------------------
    # Inputs / derived values
    # -------------------------
    try:
        moon_pct = float(moon_pct)
    except Exception:
        moon_pct = 0.0
    moon_pct = max(0.0, min(100.0, moon_pct))

    t = moon_pct / 100.0  # 0..1

    # Single source of truth for the route curve
    ROUTE_D = "M 170 310 C 360 170, 640 170, 830 260"

    # Progress dash lengths (tuned visuals)
    PROG_ON = t * 1600.0
    PROG_OFF = 2000.0

    # Bike offset (so it sits slightly above the path end marker)
    BIKE_DX = 0     # +ve moves right
    BIKE_DY = -30   # -ve moves up

    html = f"""
<div style="border-radius:18px; overflow:hidden; border:1px solid rgba(49,51,63,0.15); background:#070A12;">
  <div style="position:relative; height:{height}px; width:100%;">

    <!-- Starfield background -->
    <div style="
      position:absolute; inset:0;
      background:
        radial-gradient(circle at 15% 25%, rgba(255,255,255,0.12) 0 1px, transparent 2px),
        radial-gradient(circle at 60% 35%, rgba(255,255,255,0.10) 0 1px, transparent 2px),
        radial-gradient(circle at 80% 70%, rgba(255,255,255,0.08) 0 1px, transparent 2px),
        radial-gradient(circle at 30% 80%, rgba(255,255,255,0.09) 0 1px, transparent 2px),
        radial-gradient(circle at 45% 55%, rgba(255,255,255,0.07) 0 1px, transparent 2px);
      opacity:0.9;
    "></div>

    <!-- Scene -->
    <svg viewBox="0 0 1000 520" width="100%" height="100%"
         preserveAspectRatio="xMidYMid meet"
         style="position:absolute; inset:0;">

      <!-- ====== defs ====== -->
      <defs>
        <radialGradient id="earthGrad" cx="35%" cy="30%" r="70%">
          <stop offset="0%"   stop-color="rgba(120,200,255,0.95)"/>
          <stop offset="45%"  stop-color="rgba(40,120,220,0.95)"/>
          <stop offset="100%" stop-color="rgba(10,30,90,0.95)"/>
        </radialGradient>

        <radialGradient id="moonGrad" cx="35%" cy="30%" r="70%">
          <stop offset="0%"   stop-color="rgba(255,255,255,0.95)"/>
          <stop offset="55%"  stop-color="rgba(200,200,200,0.95)"/>
          <stop offset="100%" stop-color="rgba(120,120,120,0.95)"/>
        </radialGradient>

        <filter id="softGlow">
          <feGaussianBlur stdDeviation="6" result="blur"/>
          <feMerge>
            <feMergeNode in="blur"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>

        <!-- Route path definition -->
        <path id="route" d="{ROUTE_D}" />
      </defs>

      <!-- ====== route (dotted base) ====== -->
      <use href="#route"
           fill="none"
           stroke="rgba(255,255,255,0.20)"
           stroke-width="3"
           stroke-dasharray="6 10" />

      <!-- ====== progress segment (orange glow) ====== -->
      <use href="#route"
           fill="none"
           stroke="rgba(255,180,70,0.85)"
           stroke-width="6"
           stroke-linecap="round"
           stroke-dasharray="{PROG_ON:.1f} {PROG_OFF:.1f}" />

      <!-- ====== bike locked at end of progress (same endpoint) ====== -->
      <g>
        <text x="0" y="0"
              font-size="44"
              text-anchor="middle"
              dominant-baseline="central"
              transform="translate({BIKE_DX},{BIKE_DY}) scale(-1,1)">
          
        </text>

        <animateMotion dur="0.01s" fill="freeze"
          path="{ROUTE_D}"
          keyPoints="{t:.4f};{t:.4f}"
          keyTimes="0;1"
          calcMode="linear" />
      </g>

      <!-- ====== planets ====== -->
      <!-- Earth -->
      <circle cx="150" cy="330" r="95" fill="url(#earthGrad)" filter="url(#softGlow)"/>
      <circle cx="150" cy="330" r="98" fill="none" stroke="rgba(120,200,255,0.20)" stroke-width="10"/>

      <!-- Moon -->
      <circle cx="860" cy="250" r="55" fill="url(#moonGrad)" filter="url(#softGlow)"/>
      <circle cx="860" cy="250" r="58" fill="none" stroke="rgba(255,255,255,0.15)" stroke-width="8"/>

      <!-- ====== labels ====== -->
      <text x="70" y="455" fill="rgba(255,255,255,0.75)" font-size="22" font-family="sans-serif">Earth</text>
      <text x="830" y="345" fill="rgba(255,255,255,0.75)" font-size="22" font-family="sans-serif">Moon</text>

    </svg>

    <!-- Bottom info strip -->
    <div style="
      position:absolute; left:0; right:0; bottom:0;
      padding:12px 14px;
      background:linear-gradient(180deg, rgba(7,10,18,0.00), rgba(7,10,18,0.75));
      color:rgba(255,255,255,0.88);
      font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
      display:flex; justify-content:space-between; gap:12px; align-items:flex-end;
    ">
      <div style="font-weight:800; font-size:1.05rem;">🌍 → 🌕 Journey Progress</div>
      <div style="text-align:right; font-size:0.95rem;">
        <div><b>{moon_pct:.1f}%</b> completed</div>
        <div style="opacity:0.80;">Total cycling: <b>{total_km:,.0f} km</b></div>
      </div>
    </div>

  </div>
</div>
"""
    components.html(html, height=height + 10, scrolling=False)



# =========================
# Sidebar – Data Source (Auto-refresh if cache stale)
# =========================
st.sidebar.header("📥 Data source")
st.sidebar.caption("Data is integrated via Strava API + Garmin dump data for Steps.")

refresh_clicked = st.sidebar.button("🔄 Fetch latest from Strava")

# Cache status (for user visibility)
cache_exists = STRAVA_PARQUET_PATH.exists()
cache_age = _parquet_cache_age_hours() if cache_exists else None
auto_refresh_needed = _parquet_is_stale(CACHE_MAX_AGE_HOURS)

if not cache_exists:
    st.sidebar.info("📁 Cache: not found (will fetch from Strava)")
else:
    st.sidebar.caption(f"🕒 Cache age: {cache_age:.1f} hours" if cache_age is not None else "🕒 Cache age: unknown")

# Treat auto-refresh same as refresh (but softer: no full cache clear)
refresh_mode = refresh_clicked or auto_refresh_needed

# If user manually refreshes, clear Streamlit cache too
if refresh_clicked:
    st.cache_data.clear()
    st.session_state.pop("raw_activities_df", None)

# If auto-refresh needed, drop session raw so we fetch fresh once
if auto_refresh_needed and ("raw_activities_df" in st.session_state):
    st.session_state.pop("raw_activities_df", None)
    st.sidebar.info("Auto-refreshing from Strava (cache older than 6 hours)…")

# 1) Try parquet first (fast + works even if API fails), unless refresh_mode requested
if ("raw_activities_df" not in st.session_state) and (not refresh_mode):
    cached_df = _load_strava_parquet()
    if cached_df is not None and not cached_df.empty:
        st.session_state["raw_activities_df"] = cached_df

# 2) If refresh_mode OR no cached df in session, fetch from API and overwrite parquet
if ("raw_activities_df" not in st.session_state) or refresh_mode:
    try:
        with st.spinner("Fetching activities from Strava…"):
            acts_json = fetch_strava_activities_json(
                after=None,
                max_pages=50,
                cache_buster=int(time.time()),  # forces truly fresh API fetch
            )
            fresh_df = strava_json_to_dataframe(acts_json)

        _save_strava_parquet(fresh_df)
        st.session_state["raw_activities_df"] = fresh_df

    except Exception as e:
        fallback_df = _load_strava_parquet()
        if fallback_df is not None and not fallback_df.empty:
            st.warning("Strava API fetch failed. Loaded cached parquet data instead.")
            st.session_state["raw_activities_df"] = fallback_df
        else:
            st.error(f"Strava API fetch failed and no parquet cache found.\n\nError: {e}")
            st.stop()

raw = st.session_state.get("raw_activities_df")
if raw is None or raw.empty:
    st.warning("No activities returned from Strava. Check API credentials or try refresh.")
    st.stop()

data = parse_activities_cached(raw)

# =========================
# Diagnostics (helps catch missing activities / caching issues)
# =========================
# with st.sidebar.expander("🧪 Diagnostics", expanded=False):
#     try:
#         raw_max = pd.to_datetime(raw["Activity Date"], errors="coerce").max()
#         data_max = pd.to_datetime(data["Activity Date"], errors="coerce").max()

#         st.write("**Raw (from Strava/parquet) rows:**", len(raw))
#         st.write("**Parsed (after processing) rows:**", len(data))
#         st.write("**Raw max Activity Date:**", raw_max)
#         st.write("**Parsed max Activity Date:**", data_max)

#         st.markdown("**Raw counts by Activity Type**")
#         if "Activity Type" in raw.columns:
#             st.dataframe(raw["Activity Type"].value_counts().rename("count"))

#         st.markdown("**Parsed counts by Sport**")
#         if "Sport" in data.columns:
#             st.dataframe(data["Sport"].value_counts().rename("count"))

#     except Exception as e:
#         st.warning(f"Diagnostics error: {e}")

total_active_days, longest_streak = compute_activity_day_stats(data)
# --- Convert days into approximate years ---
active_years = total_active_days / 365.0
streak_years = longest_streak / 365.0



# =========================
# Filters (Keep ONLY YEAR)
# =========================
st.sidebar.header("🔎 Filter (for Sport Tabs only)")

# ✅ Keep Year filter (Current year by default)
from datetime import datetime

years = sorted(data["Year"].dropna().unique().tolist())
year_options = ["All"] + years

current_year = datetime.now().year
default_index = year_options.index(current_year) if current_year in years else 0

year_selection = st.sidebar.selectbox(
    "Year",
    year_options,
    index=default_index
)

# ✅ No sports filter (tabs already separate them)
df_all_years = data.copy()   # for charts / trends (no year filter)
df = df_all_years.copy()     # for KPIs / quick views (year filter applies)

if year_selection != "All":
    df = df[df["Year"] == year_selection].copy()

# Header range (informational only)
hdr_start = pd.to_datetime(df["Activity Date"]).min().date() if len(df) else None
hdr_end = pd.to_datetime(df["Activity Date"]).max().date() if len(df) else None

st.success(f"Loaded **{len(df):,}** activities from **{len(data):,}** total (Year filter applied to KPIs).")

with st.container():
    st.markdown("##### Current Selection")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Year:** {year_selection}")
    with c2:
        st.write(f"**Date:** {hdr_start} → {hdr_end}" if hdr_start and hdr_end else "**Date:** —")


# -------------------------
# Small helpers (global)
# -------------------------
def _dur_str(sec: float) -> str:
    """Format seconds as 'Hh Mm' (safe for NaN/None)."""
    try:
        sec = float(sec)
    except Exception:
        return "—"
    if not np.isfinite(sec) or sec <= 0:
        return "—"
    sec = int(round(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    return f"{h}h {m}m" if h > 0 else f"{m}m"



# =========================
# Tabs
# =========================
tab_about, tab_steps, tab_cyc, tab_run, tab_walk, tab_yoga, tab_swim, tab_all = st.tabs(
    [
        "👋 About",
        "👣 Steps",
        "🚴 Cycling",
        "🏃 Running",
        "🚶 Walking",
        "➕ Yoga/Strength training",
        "🏊 Swimming",
        "📊 All Sports",
    ]
)

with tab_about:
    st.subheader("👋 About me & this dashboard")

    # --- About layout: Photo + Story ---
    col1, col2 = st.columns([0.9, 2.6], gap="large")

    with col1:
        st.markdown(
            """
            <style>
            .profile-pic img {
                border-radius: 50%;
                border: 3px solid rgba(255,255,255,0.25);
                box-shadow: 0 8px 20px rgba(0,0,0,0.35);
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown('<div class="profile-pic">', unsafe_allow_html=True)
        # --- Profile Photo (from vendor folder) ---
        vendor_dir = _find_vendor_dir()
        photo_path = vendor_dir / "my_photo.jpg"

        if photo_path.exists():
            st.image(str(photo_path), use_container_width=True)

        else:
            st.info("📷 Please place `my_photo.jpg` inside App_Data/vendor/")

        # =========================
        # ✅ NEW: Latest Day Activity Proof (freshness indicator)
        # =========================
        st.divider()
        st.subheader("📅 Latest activity(-ies) in my day to day journey")

        # Activity Date is already parsed; still keep robust conversion
        _dt = pd.to_datetime(data["Activity Date"], errors="coerce")
        last_ts = _dt.max()

        if pd.isna(last_ts):
            st.info("No valid activity dates found yet in the dataset.")
        else:
            last_day = last_ts.date()
            day_df = data[_dt.dt.date == last_day].copy()

            # 🔥 Headline: "On Feb 21, I logged 3 activities across 3 sports: ..."
            day_df["Sport"] = day_df["Sport"].fillna("Other").astype(str)

            n_acts = int(len(day_df))
            n_sports = int(day_df["Sport"].nunique())
            sports_list = day_df["Sport"].value_counts().index.tolist()

            # nice emoji mapping
            sport_emoji = {
                "Cycling": "🚴",
                "Running": "🏃",
                "Walking": "🚶",
                "Workout": "🧘",
                "Swimming": "🏊",
                "Other": "⭐",
            }

            sport_str = ", ".join([f"{sport_emoji.get(s, '⭐')} {s}" for s in sports_list])

            st.markdown(
                f"🔥 On **{last_day:%b %d, %Y}**, I logged **{n_acts}** activities across **{n_sports}** sports: {sport_str}"
            )

            

            # Clean numeric fields (safe)
            day_df["Distance"] = pd.to_numeric(day_df["Distance"], errors="coerce").fillna(0.0)
            day_df["Moving Time"] = pd.to_numeric(day_df["Moving Time"], errors="coerce").fillna(0.0)

            # Summary by sport
            summary = (
                day_df.groupby("Sport")
                .agg(
                    Activities=("Activity ID", "count"),
                    Distance_km=("Distance", "sum"),
                    Time_s=("Moving Time", "sum"),
                )
                .reset_index()
                .sort_values(["Activities", "Distance_km"], ascending=False)
            )

            summary["Distance"] = summary["Distance_km"].apply(lambda x: f"{x:,.1f} km" if x > 0 else "—")
            summary["Time"] = summary["Time_s"].apply(_dur_str)

            #st.markdown("**What I did today:**")
            st.dataframe(
                summary[["Sport", "Activities", "Distance", "Time"]],
                use_container_width=True,
                hide_index=True,
            )

            st.success(f"✅ Latest activity date in this app: **{last_day}**")
            
            # with st.expander("🔎 See activity names (that day)", expanded=False):
            #     tmp = day_df[["Activity Name", "Sport", "Distance", "Moving Time"]].copy()
            #     tmp["Distance (km)"] = tmp["Distance"].round(1)
            #     tmp["Time"] = tmp["Moving Time"].apply(_dur_str)

            #     st.dataframe(
            #         tmp[["Activity Name", "Sport", "Distance (km)", "Time"]],
            #         use_container_width=True,
            #         hide_index=True,
            #     )

            # Optional: show cache freshness (uses your existing cache_age variables)
            # try:
            #     if cache_age is not None:
            #         st.caption(
            #             f"🕒 Cache age at load time: {cache_age:.1f} hours "
            #             f"(auto-refresh triggers if > {CACHE_MAX_AGE_HOURS} hours)."
            #         )
            # except Exception:
            #     pass

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
        f"""

Hi there, I’m **Prasad Reddi from India** — an engineer by profession, a cyclist at heart, and a sports practitioner by choice.

Years ago I realized my **health was slipping** — especially when I saw my HbA1c values trending in the wrong direction. I loved my food, and I didn’t want to give up the flavors of life, but I knew something had to change. I didn’t want strict diets or intense routines that burn out quickly. I wanted a sustainable way to stay healthy and joyful.

So my journey began with **walking**. At first it felt simple — and not always intense — but consistency was the key, and I kept showing up. Then yoga entered the routine — peaceful, grounding, but still not quite hitting the intensity I needed.

One day I decided to ride a bicycle — **just 5 km**. That small ride felt *big* at the time. But with **discipline and consistency**, those 5 km slowly became 10 km, then 15–20 km. Life is busy, and finding dedicated time for sports felt nearly impossible due to work — so I made my commute my training. I started **cycling to work** instead of riding a motorcycle. That small decision turned into a love affair with cycling.

I learned how to cope with real-world obstacles — traffic, weather, sweat, carrying bags, and equipping the cycle with the right lights and gear — all by myself. There was no perfect setup, no ideal day — just *showing up* every single time.

To complement cycling, I took up **running** too. Initially I feared running — especially with flat feet. The first runs were awkward, uncomfortable, and unfamiliar. But slowly I learned my body mechanics, respected my limits, and progressed from short 2 km jogs to **half marathons**.

Even on days when I couldn’t go out, I walked at home, stretched, or did yoga. Nowadays, sports isn’t just a hobby — it’s an **addiction of positivity**. In the last few years, I have stayed deeply committed to movement.

🌿 Beyond health and lifestyle, sports gave me something even deeper.

Through cycling and running, I discovered an unlimited friendship with nature — the early morning silence, open roads, fresh air, and the joy of being fully present.

Along the way, I also met people from diverse cultures and backgrounds. Sports became a bridge — helping me connect, learn, share stories, and build friendships beyond language or nationality.

Most importantly, this journey transformed me into a more self-sustainable person — confident, independent, and adaptable. Whether I’m riding alone through challenges or blending into new environments, sports taught me how to stay grounded, resilient, and open to life.

What started as a health decision became a life philosophy:  
**move forward, connect deeply, and grow stronger — inside and out.**

📌 According to my Strava history, I have logged activities on  
📌 According to my Strava history, I have been active for <span class="num-highlight"><span class="num-highlight">{total_active_days:,}</span> days (~<span class="num-highlight">{active_years:.1f}</span> years of movement).

with a longest streak of <span class="num-highlight">{longest_streak:,}</span> consecutive active days (~<span class="num-highlight">{streak_years:.1f}</span> years without a break) — be it cycling, running, walking, yoga, or anyting. That discipline and commitment didn’t just change my body — it transformed my resilience, my confidence, and my mindset. What once scared me now feels like second nature — just go and get it.

---

### 🔗 My Fitness Profiles

If you’d like to explore my journey beyond this dashboard, here are my official activity profiles:

- 🚴 **Strava:** [Prasad on Strava](https://www.strava.com/athletes/22213198)  
- ⌚ **Garmin Connect:** [Prasad on Garmin](https://connect.garmin.com/app/profile/23d031b8-5afc-4de0-96af-5c7210aca7f5)

---

---

### 🧭 How to Use This Dashboard

This dashboard is built to show the **life & story behind the numbers** — not just charts and figures. It pulls data from **Strava activities** (dynamic) and **Garmin steps** (static), and organises them into insights that celebrate consistency, progress, and personal growth.

- **Explore each tab in order** — Steps, Cycling, Running, Walking, Yoga/Strength, Swimming, and All Sports.
- Use the **Year filter** to see how each year contributed to your journey.
- **Important:** The graphs show **overall (all-time) trends**, while the Year filter updates only the **KPI numbers**.

Enjoy the journey — numbers tell *what*, stories tell *why*, and together they show *how far you’ve come*.

        """,
        unsafe_allow_html=True
    )
    st.info("Tip: Start with Steps → Cycling → Running → Walking → Yoga → Swimming → All Sports ✅")


with tab_steps:
    st.subheader("👣 Steps (data from Garmin) – Story on a Globe")

    # -------------------------------
    # Repo-safe Steps JSON locator
    # -------------------------------
    from pathlib import Path
    import datetime as dt

    def _find_repo_root(start: Path) -> Path | None:
        cur = start.resolve()
        for _ in range(10):  # climb up to 10 levels
            if (cur / ".git").exists():
                return cur
            cur = cur.parent
        return None

    import os   # ← make sure this exists at top

    def _find_steps_json() -> Path:

        # ---------------------------------
        # 0) ENVIRONMENT VARIABLE OVERRIDE
        # ---------------------------------
        override = os.environ.get("STEPS_JSON_PATH", "").strip()
        if override:
            p0 = Path(override)
            if p0.exists():
                return p0

        # ---------------------------------
        # Normal repo detection
        # ---------------------------------
        here = Path(__file__).resolve().parent
        repo = _find_repo_root(here)

        # 1) If repo root found, prefer repo/data
        if repo:
            p = repo / "data" / "demo_garmin_steps_summary.json"
            if p.exists():
                return p

        # 2) fallback: next to script (your earlier logic)
        p1 = here / "data" / "demo_garmin_steps_summary.json"
        if p1.exists():
            return p1

        p2 = here / "App_Data" / "demo_garmin_steps_summary.json"
        if p2.exists():
            return p2

        return (repo / "data" / "demo_garmin_steps_summary.json") if repo else p1

    steps_json_path = _find_steps_json()

    cache_key = _steps_summary_cache_key(steps_json_path)
    steps_summary = load_steps_summary(str(steps_json_path), cache_key)

    if steps_json_path.exists():
        st.caption(
            f"📌 Steps file: {steps_json_path.name} • "
            f"Last updated: "
            f"{dt.datetime.fromtimestamp(steps_json_path.stat().st_mtime).strftime('%d %b %Y, %I:%M %p')}"
        )


    # st.write("DEBUG total_days:", steps_summary.get("total_days") if steps_summary else None)
    # st.write("DEBUG range:", steps_summary.get("range_start"), steps_summary.get("range_end"))

    # st.caption(f"DEBUG steps json: {json_path}")
    # if json_path.exists():
    #     st.caption(f"DEBUG last modified: {dt.datetime.fromtimestamp(json_path.stat().st_mtime)}")

    # st.caption(f"DEBUG steps json path: {json_path}")
    # st.caption(f"DEBUG exists: {json_path.exists()}")

    if not steps_summary:
        st.error(f"Steps summary file not found/readable at: {steps_json_path}")
    else:
        total_steps = int(steps_summary.get("total_steps", 0))
        dist_km = float(steps_summary.get("distance_from_steps_km_estimated", 0.0))
        # --- Daily Average Steps (auto-computed) ---
        # last_updated_str = steps_summary.get("last_updated", None)
        # end_date = pd.to_datetime(last_updated_str) if last_updated_str else pd.Timestamp.today()

        # # Use earliest activity date as tracking start (available in app data)
        # start_date = pd.to_datetime(data["Activity Date"].min())

        # # Inclusive day count (avoid divide-by-zero)
        # total_days = max(1, (end_date.date() - start_date.date()).days + 1)

        # daily_avg_steps = int(round(total_steps / total_days))

        earth_rounds = steps_summary.get("times_around_earth", "—")
        earth_diam = steps_summary.get("earth_diameters_equivalent", "—")

        # --- Garmin-range Daily Average Steps (robust) ---
        g_days = int(steps_summary.get("total_days", 0) or 0)

        g_start_raw = (steps_summary.get("range_start") or "").strip()
        g_end_raw = (steps_summary.get("range_end") or steps_summary.get("last_updated") or "").strip()

        if g_days <= 0:
            daily_avg_steps = 0
            avg_sub = "Garmin range not available"
        else:
            daily_avg_steps = int(round(total_steps / g_days))
            avg_sub = f"Avg since {g_start_raw} → {g_end_raw} ({g_days:,} days)"



        # Map distance to a gentle altitude range (0.8 .. 2.6)
        altitude = 1.4
        try:
            altitude = min(2.6, max(0.8, 0.8 + (dist_km / 40075.0) * 1.8))
        except Exception:
            altitude = 1.4

        # --- 2-column layout: Globe (left) + Steps story (right) ---
        earth_circum_km = 40075.0
        lap_frac = (max(0.0, dist_km) / earth_circum_km) % 1.0
        lap_pct = int(round(lap_frac * 100))

        col_globe, col_story = st.columns([2.2, 1.0], gap="large")

        with col_globe:
            render_globe(height=520, altitude=altitude)

            # ✅ Photo Courtesy Line (NASA Credit)
            st.caption(
                "🌍 Globe texture courtesy: NASA Scientific Visualization Studio (SVS) — "
                "https://svs.gsfc.nasa.gov/2915"
            )

            st.markdown(
                _html(
                    """
                    <div class="steps-tip" style="margin-top:12px;">
                    <div class="t1">Tip</div>
                    <div class="t2">Drag to rotate • scroll to zoom • double-click to reset</div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

            st.markdown(
                _html(
                    """
                    <div class="quote-box">
                    <div class="quote-text">👣 “A journey of a thousand miles begins with a single step.” — Lao Tzu</div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )


        with col_story:
            st.markdown(
                _html(
                    f"""
                    <div class="steps-story">
                    <div class="kicker">👣 My walking & steps journey</div>
                    <div class="headline">{dist_km:,.0f} km by steps — {earth_rounds} Earth rounds 🌍</div>
                    <div class="body">
                        • <b>Total steps:</b> {total_steps:,}<br/>
                        • <b>Estimated distance:</b> {dist_km:,.1f} km<br/>
                        • <b>Earth diameters equivalent:</b> {earth_diam}<br/>
                        • <b>Daily average:</b> {daily_avg_steps:,} steps<br/>
                        <br/>
                        <span style="color:#6b7280;">Range: {g_start_raw} → {g_end_raw} ({g_days:,} days)</span>
                    </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )


            # 2) Progress Ring BELOW Story card
            st.markdown(
                _html(
                    f"""
            <div class="steps-ring-card">
            <div class="steps-ring-wrap">
                <div style="position:relative;">
                <div class="steps-ring" style="background: conic-gradient(#f59e0b 0% {lap_pct}%, rgba(49, 51, 63, 0.10) {lap_pct}% 100%);"></div>
                <div class="steps-ring-text"
                    style="position:absolute; inset:0; display:grid; place-items:center;">
                    {lap_pct}%
                </div>
                </div>

                <div>
                <div style="font-weight:800; font-size:1.05rem;">Walking Progress</div>
                <div class="steps-ring-sub">
                    I have completed <b>{lap_pct}%</b> of one Earth round
                </div>
                </div>
            </div>
            </div>
            """
                ),
                unsafe_allow_html=True,
            )

            # 3) KPI Tiles (2 per row) — Apple Health style
            st.markdown(
                _html(
                    f"""
            <div class="steps-kpi-grid">

            <div class="steps-kpi-card">
                <div class="label">Daily Average Steps</div>
                <div class="value">{daily_avg_steps:,}</div>
                <div class="sub">{avg_sub}</div>
            </div>


            <div class="steps-kpi-card">
                <div class="label">Current 10K steps/day Streak</div>
                <div class="value">{steps_summary.get("current_10k_streak_days", 0)}
                <span class="unit">days</span>
                </div>
                <div class="sub">Active streak right now</div>
            </div>

            <div class="steps-kpi-card">
                <div class="label">Best 10K steps/day Streak</div>
                <div class="value">{steps_summary.get("max_10k_streak_days", 0)}
                <span class="unit">days</span>
                </div>
                <div class="sub">my all-time record</div>
            </div>

            <div class="steps-kpi-card">
                <div class="label">Best Day with highest Steps</div>
                <div class="value">{int(steps_summary.get("best_day_steps", 0)):,}</div>
                <div class="sub">Highest single-day total</div>
            </div>

            </div>
            """
                ),
                unsafe_allow_html=True,
            )




        # KPI row below (keep it light)
        # c1, c2, c3, c4 = st.columns(4)
        # c1.metric("Total Steps", f"{int(steps_summary.get('total_steps', 0)):,}")
        # c2.metric("Current 10K Streak", f"{steps_summary.get('current_10k_streak_days', 0)} days")
        # c3.metric("Best 10K Streak", f"{steps_summary.get('max_10k_streak_days', 0)} days")
        # c4.metric("Best Day Steps", f"{int(steps_summary.get('best_day_steps', 0)):,}")

        st.divider()


with tab_all:
    
    soft_divider()   # ✅ ADD THIS LINE
    st.subheader("📊 All Sports – Overview (All Years)")

    # ✅ Always use full dataset
    all_df = df_all_years.copy()

    # =====================================================
    # 🏆 EXECUTIVE INSIGHTS (NEW SECTION)
    # =====================================================
    if not all_df.empty:

        # ---- Lifetime distance ----
        lifetime_km = all_df["Distance"].sum()

        # ---- Best Year ----
        yearly = (
            all_df.groupby("Year")["Distance"]
            .sum()
            .sort_values(ascending=False)
        )

        best_year = int(yearly.index[0])
        best_year_km = yearly.iloc[0]

        # ---- Most Active Month ----
        monthly = (
            all_df.groupby("MonthPeriod")["Distance"]
            .sum()
            .sort_values(ascending=False)
        )

        best_month = (
            monthly.index[0]
            .to_timestamp()
            .strftime("%b-%Y")
        )
        best_month_km = monthly.iloc[0]

        # ---- Journey Since badge ----
        first_year = int(all_df["Year"].min())
        last_year = int(all_df["Year"].max())
        years_active = (last_year - first_year + 1)

        st.markdown(
            f"🧭 **Active Since {first_year}**  •  **{years_active} years** of continuous movement and still counting...."
        )

        # ---- Display cards ----
        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("🏆 Best Year", f"{best_year}")
            st.caption(f"{best_year_km:,.0f} km covered")

        with c2:
            st.metric("🔥 Most Active Month", best_month)
            st.caption(f"{best_month_km:,.0f} km in that month")

        with c3:
            st.metric("🚀 Lifetime Distance", f"{lifetime_km:,.0f} km")
            st.caption("Across all recorded activities")
        

    st.divider()

    # =====================================================
    # 🔥 CONSISTENCY INSIGHTS
    # =====================================================

    years_available = all_df["Year"].nunique()
    avg_km_per_year = lifetime_km / years_available if years_available > 0 else 0

    c4, c5, c6 = st.columns(3)

    with c4:
        st.metric(
            "🔥 Active Days",
            f"{total_active_days:,}"
        )
        st.caption(f"~{active_years:.1f} years of movement")

    with c5:
        st.metric(
            "🧭 Longest Streak",
            f"{longest_streak:,} days"
        )
        st.caption(f"~{streak_years:.1f} years without break")

    with c6:
        st.metric(
            "📈 Avg km / Year",
            f"{avg_km_per_year:,.0f} km"
        )
        st.caption("Across entire activity history")

    st.divider()

    st.info("Note: All Sports tab always shows **All Years** (Year filter applies only to sport tabs KPIs).")

    generic_time_series(all_df, "All Sports", key_prefix="all_sports_all_years")
    distance_distribution(all_df, "All Sports")

    st.divider()
    st.caption("Export the all-years dataset used in this tab:")
    download_button(all_df, "⬇️ Download CSV (All Years)", "strava_all_years.csv")


with tab_cyc:
    st.subheader("🚴 Cycling – Lifestyle, Commute & Impact")

    # All cycling activities in loaded Strava range (ignores current filters)
    cyc_all = data[data["Sport"] == "Cycling"].copy()

    # Cycling activities under current sidebar filters
    cyc = df[df["Sport"] == "Cycling"].copy()

    if cyc_all.empty:
        st.info("No cycling data found in the loaded Strava range.")
    else:
        story = build_cycling_lifestyle_story(cyc_all, cyc)
        # --- Total cycling distance (for globe progress) ---
        total_km = float(cyc_all["Distance"].sum() or 0.0)

        # =========================
        # JIRA: Overall + Selected-year cycling stats (default: current year)
        # =========================
        year_label, cyc_selected_year = _resolve_year_slice(cyc_all, year_selection, year_col="Year")
        cyc_overall = cyc_all  # overall ignores sidebar filters

        def _cyc_stats(dfx: pd.DataFrame) -> dict:
            rides = int(len(dfx))
            km = float(dfx["Distance"].sum() or 0.0)
            hours = float((dfx["Moving Time"].sum() or 0.0) / 3600.0)
            commute_rides = int((dfx["Commute"] == True).sum())
            commute_km = float(dfx.loc[dfx["Commute"] == True, "Distance"].sum() or 0.0)
            commute_share = (commute_km / km * 100.0) if km > 0 else 0.0
            return dict(rides=rides, km=km, hours=hours, commute_rides=commute_rides, commute_share=commute_share)

        ov = _cyc_stats(cyc_overall)
        sy = _cyc_stats(cyc_selected_year)

        # =========================
        # Cycling KPI sections (WITH outer frame + inner KPI cards)
        # =========================

        # --- Overall Cycling (All Years) ---
        try:
            with st.container(border=False):
                st.markdown("### 🌐 Overall Cycling (All Years) statistics - Both commutes and non-commutes")
                o1, o2, o3, o4 = st.columns(4)
                with o1: kpi_card("Rides", f"{ov['rides']:,}")
                with o2: kpi_card("Distance", f"{ov['km']:,.0f} km")
                with o3: kpi_card("Time", f"{ov['hours']:,.0f} hrs")
                with o4: kpi_card("Commutes", f"{ov['commute_rides']:,}")
        except TypeError:
            # Fallback (older Streamlit without border=True)
            st.markdown("### 🌐 Overall Cycling (All Years) statistics - Both commutes and non-commutes")
            o1, o2, o3, o4 = st.columns(4)
            with o1: kpi_card("Rides", f"{ov['rides']:,}")
            with o2: kpi_card("Distance", f"{ov['km']:,.0f} km")
            with o3: kpi_card("Time", f"{ov['hours']:,.0f} hrs")
            with o4: kpi_card("Commutes", f"{ov['commute_rides']:,}")

        soft_divider()
        
        # --- Cycling in selected year ---
        try:
            with st.container(border=False):
                st.markdown(f"### 📅 Cycling in {year_label}")
                y1, y2, y3, y4 = st.columns(4)
                with y1: kpi_card("Rides", f"{sy['rides']:,}")
                with y2: kpi_card("Distance", f"{sy['km']:,.0f} km")
                with y3: kpi_card("Time", f"{sy['hours']:,.0f} hrs")
                with y4: kpi_card("Commute share", f"{sy['commute_share']:.0f}%")
        except TypeError:
            st.markdown(f"### 📅 Cycling in {year_label}")
            y1, y2, y3, y4 = st.columns(4)
            with y1: kpi_card("Rides", f"{sy['rides']:,}")
            with y2: kpi_card("Distance", f"{sy['km']:,.0f} km")
            with y3: kpi_card("Time", f"{sy['hours']:,.0f} hrs")
            with y4: kpi_card("Commute share", f"{sy['commute_share']:.0f}%")


        # --- NEW: Cycling globe (different from Steps globe) ---
        #render_cycling_globe_plotly(total_km)

        
        # --- 2-column layout: Plotly Cycling Globe (left) + Story (right) ---
        #total_km = float(cyc_all["Distance"].sum() or 0.0)

        # --- Story-first layout: Story (left, primary) + Visual (right, supporting) ---
        col_story, col_viz = st.columns([1.55, 1.15], gap="large")

        with col_story:
            # KPIs to strengthen the story column (so text dominates)
            rides = int(len(cyc_all))
            avg_ride = (total_km / rides) if rides else 0.0

            try:
                commute_km = float(cyc_all.loc[cyc_all["Commute"] == True, "Distance"].sum() or 0.0)
            except Exception:
                commute_km = 0.0

            commute_share = (commute_km / total_km * 100.0) if total_km > 0 else 0.0

            # k1, k2, k3 = st.columns(3)
            # with k1:
            #     kpi_card("Total rides (commutes + non commutes)", f"{rides:,}")
            # with k2:
            #     kpi_card("Avg ride", f"{avg_ride:,.1f} km")
            # with k3:
            #     kpi_card("Commute share", f"{commute_share:,.0f}%")

            st.markdown(
                _html(
                    f"""
                    <div class="steps-story" style="margin-top:10px;">
                    <div class="kicker">🚴 My cycling journey from Earth → Moon 🌍➡️🌕🚀😄</div>
                    <div class="headline">{story["headline"]}</div>
                    <div class="body">{story["body_html"]}</div>

                    <!-- 🌿 Beyond the Numbers -->
                    <div class="steps-tip" style="margin-top:16px;">
                        <div class="t1">🌿 Beyond the Numbers 🌿</div>

                        <div class="t2">
                        Cycling has given me far more than kilometers and ride counts.<br/>
                        It has shaped my character & qualities in both my personal and professional journey:<br/>
                        • <b>Resilience</b> to keep moving forward even on hard days<br/>
                        • <b>Patience</b> to trust long journeys and slow progress<br/>
                        • <b>Discipline</b> to show up consistently<br/>
                        • <b>A never-give-up mindset</b> when the road gets tough<br/>
                        • <b>Consistency</b> that translates into every part of life and work<br/><br/>

                        For me, cycling is not just a sport —<br/>
                        It is a lifestyle of growth, strength, and freedom that embedded
                        <b>resilience</b>, <b>patience</b>, and a <b>never-give-up mindset</b> in my life.<br/><br/>

                        🌟 To anyone reading this: Whether you cycle or not, I hope you find your own
                        <b>lifestyle</b> that builds these qualities in <b>YOU</b>, because they are the true fuel
                        for any long journey in both life and work. 🌟
                        </div>

                        <div class="quote-box">
                        <div class="quote-text">✨ “It always seems impossible until it’s done.” — Nelson Mandela</div>
                        </div>
                    </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )



        with col_viz:
            moon_distance_km = 384400.0
            moon_pct = (total_km / moon_distance_km * 100.0) if total_km > 0 else 0.0

            render_earth_moon_journey(
                moon_pct=moon_pct,
                total_km=total_km,
                height=380,   # ✅ reduced so story dominates
            )


        # =========================
        # 📖 My Commute Blog (Expandable)
        # =========================
        BLOG_URL = "https://cyclingmonks.com/cycle-commuting-pune-prasad-reddi/"

        with st.expander("📖 Read my commute blog by Cycling Monks – Cycling as a lifestyle"):
            st.write(
                "This is my personal story about cycle commuting, health, mindset, and sustainability. Though its a little old, but gives some more information :-)"
            )

            st.link_button(
                "Open the full blog in a new tab ↗️",
                BLOG_URL,
            )

        st.markdown("#### Cycling – Trends & Distributions (Overall / All Years)")

        # ✅ Freeze charts to overall cycling (ignore year filter)
        cyc_charts = cyc_all.copy()   # or: df_all_years[df_all_years["Sport"]=="Cycling"].copy()

        if cyc_charts.empty:
            st.info("No cycling data available for overall charts.")
        else:
            generic_time_series(cyc_charts, "Cycling")
            distance_distribution(cyc_charts, "Cycling")
            commute_split(cyc_charts, "Cycling")




with tab_run:
    # --- Overall (All Years) + Selected Year (Running) ---
        run_all = data[data["Sport"] == "Running"].copy()

        year_label, run_year = _resolve_year_slice(run_all, year_selection, year_col="Year")


        def _min_per_km_str(min_per_km: float) -> str:
            if not (min_per_km and min_per_km > 0):
                return "—"
            m = int(min_per_km)
            s = int(round((min_per_km - m) * 60))
            if s == 60:
                m += 1
                s = 0
            return f"{m}:{s:02d} /km"

        def _running_kpis(x: pd.DataFrame) -> dict:
            if x is None or x.empty:
                return {
                    "runs": 0, "km": 0.0, "sec": 0.0, "avg_pace": "—", "longest": 0.0,
                    "k5": 0, "k10": 0, "k15": 0, "hm": 0
                }

            dist_km = pd.to_numeric(x["Distance"], errors="coerce")
            sec = pd.to_numeric(x["Moving Time"], errors="coerce")

            runs = int(len(x))
            km = float(dist_km.sum() or 0.0)
            sec_sum = float(sec.sum() or 0.0)
            longest = float(dist_km.max() or 0.0)

            # --- Average pace with your realistic filter ---
            PACE_MIN = 4.5  # min/km
            PACE_MAX = 8.0  # min/km
            pace_min_per_km = (sec / 60.0) / dist_km
            valid_pace = (dist_km > 0) & (sec > 0) & pace_min_per_km.between(PACE_MIN, PACE_MAX)

            if valid_pace.any():
                avg_pace = float(((sec[valid_pace].sum() / 60.0) / dist_km[valid_pace].sum()))
                avg_pace_str = _min_per_km_str(avg_pace)
            else:
                avg_pace_str = "—"

            # --- Distance bucket counts (your rules) ---
            k5  = int(((dist_km > 0) & (dist_km < 9.99)).sum())
            k10 = int(((dist_km >= 10) & (dist_km < 20)).sum())
            k15 = int(((dist_km > 15) & (dist_km < 20)).sum())
            hm  = int(((dist_km > 21) & (dist_km < 42)).sum())

            return {
                "runs": runs, "km": km, "sec": sec_sum, "avg_pace": avg_pace_str, "longest": longest,
                "k5": k5, "k10": k10, "k15": k15, "hm": hm
            }

        # ---- UI: Overall ----
        st.markdown("#### 🌍 Overall Running (All Years) — Quick & Overall View")
        A = _running_kpis(run_all)

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: kpi_card("Total Runs", f"{A['runs']:,}" if A["runs"] else "0")
        with c2: kpi_card("Total Distance", f"{A['km']:,.1f} km" if A["km"] else "0 km")
        with c3: kpi_card("Total Time", _dur_str(A["sec"]) if A["sec"] else "—")
        with c4: kpi_card("Average Pace", A["avg_pace"])
        with c5: kpi_card("Longest Run", f"{A['longest']:.1f} km" if A["longest"] else "0 km")

        c6, c7, c8, c9 = st.columns(4)
        with c6: kpi_card("No of 5K Runs", f"{A['k5']:,}")
        with c7: kpi_card("No of 10K Runs", f"{A['k10']:,}")
        with c8: kpi_card("No of 15K Runs", f"{A['k15']:,}")
        with c9: kpi_card("No of Half Marathons", f"{A['hm']:,}")

        soft_divider()

        # ---- UI: Selected Year ----
        st.markdown(f"#### 🗓️ Running in {year_label}")
        Y = _running_kpis(run_year)

        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: kpi_card("Total Runs", f"{Y['runs']:,}" if Y["runs"] else "0")
        with c2: kpi_card("Total Distance", f"{Y['km']:,.1f} km" if Y["km"] else "0 km")
        with c3: kpi_card("Total Time", _dur_str(Y["sec"]) if Y["sec"] else "—")
        with c4: kpi_card("Average Pace", Y["avg_pace"])
        with c5: kpi_card("Longest Run", f"{Y['longest']:.1f} km" if Y["longest"] else "0 km")

        c6, c7, c8, c9 = st.columns(4)
        with c6: kpi_card("No of 5K Runs", f"{Y['k5']:,}")
        with c7: kpi_card("No of 10K Runs", f"{Y['k10']:,}")
        with c8: kpi_card("No of 15K Runs", f"{Y['k15']:,}")
        with c9: kpi_card("No of Half Marathons", f"{Y['hm']:,}")

        
        # ---- keep your existing section after this ----
        st.subheader("Running – Quick View (current filters)")
        run = df[df["Sport"] == "Running"].copy()


        # =========================
        # 🏃 Running story + quote card (premium)
        # =========================
        st.markdown(
            dedent(
                f"""
                <div class="steps-story" style="margin-top:14px;">
                <div class="kicker">🏃‍♂️ My running journey — discipline in motion</div>

                <div class="headline">
                    From breathless beginnings to Half-Marathons — one patient step at a time
                </div>

                <div class="body">

                <p><b>Why I run:</b> Running is my cross-strength sport for cycling — it sharpens endurance,
                strengthens the body, and trains the mind to stay calm when the legs complain.</p>

                <p><b>How it started:</b> My first few runs were genuinely hard. I felt out of rhythm,
                and I learned quickly that pushing ego is the fastest path to injury.</p>

                <p>
                So I listened to my body and built slowly:
                <b>2K → 3K → 5K</b>… then <b>10K</b>, and eventually <b>Half-Marathons</b>.
                This progressive increase kept me <b>injury-free</b> and helped me grow with confidence.
                </p>

                <p>
                Today, running reminds me: progress isn’t only about speed — it’s about
                <b>discipline</b>, <b>patience</b>, and showing up even on the “not-feeling-it” days.
                </p>

                <p>
                What do I need to run: a pair of shoes, time & any place around me. No fancy gear, no gym required — just the willingness to step outside and move.
                </p>

                </div>



                <div class="quote-box" style="margin-top:12px;">
                    <div class="quote-text">✨ “The miracle isn’t that I finished. The miracle is that I had the courage to start.” — John Bingham </div>
                </div>
                </div>
                """
            ).strip(),
            unsafe_allow_html=True,
        )


        run_stats  = df[df["Sport"] == "Running"].copy()             # ✅ year-filtered for stats (already used)
        run_charts = df_all_years[df_all_years["Sport"] == "Running"].copy()  # ✅ all-years for graphs

        st.markdown("#### Running – Trends & Distributions (Overall)")
        generic_time_series(run_charts, "Running")
        distance_distribution(run_charts, "Running")




with tab_walk:
    # --- Overall (All Years) + Selected Year (Walking) ---
    walk_all = data[data["Sport"] == "Walking"].copy()

    year_label, walk_year = _resolve_year_slice(walk_all, year_selection, year_col="Year")

    def _walk_min_per_km_str(min_per_km: float) -> str:
        if not (min_per_km and min_per_km > 0):
            return "—"
        m = int(min_per_km)
        s = int(round((min_per_km - m) * 60))
        if s == 60:
            m += 1
            s = 0
        return f"{m}:{s:02d} /km"

    def _walking_kpis(x: pd.DataFrame) -> dict:
        if x is None or x.empty:
            return {
                "walks": 0, "km": 0.0, "sec": 0.0, "avg_pace": "—", "longest": 0.0,
                "lt10": 0, "gt10": 0
            }

        dist_km = pd.to_numeric(x["Distance"], errors="coerce")
        sec = pd.to_numeric(x["Moving Time"], errors="coerce")

        walks = int(len(x))
        km = float(dist_km.sum() or 0.0)
        sec_sum = float(sec.sum() or 0.0)
        longest = float(dist_km.max() or 0.0)

        # --- Walking-friendly pace filter ---
        WALK_PACE_MIN = 8.0   # min/km
        WALK_PACE_MAX = 18.0  # min/km
        pace_min_per_km = (sec / 60.0) / dist_km

        valid = (
            dist_km.notna()
            & sec.notna()
            & (dist_km >= 0.5)
            & (pace_min_per_km >= WALK_PACE_MIN)
            & (pace_min_per_km <= WALK_PACE_MAX)
        )

        valid_sec = float(sec[valid].sum() or 0.0)
        valid_km = float(dist_km[valid].sum() or 0.0)
        if valid_km > 0:
            avg_pace = (valid_sec / 60.0) / valid_km
            avg_pace_str = _walk_min_per_km_str(avg_pace)
        else:
            avg_pace_str = "—"

        # --- Buckets (match what you show today) ---
        lt10 = int(((dist_km > 0) & (dist_km < 10)).sum())
        gt10 = int(((dist_km >= 10)).sum())

        return {
            "walks": walks, "km": km, "sec": sec_sum, "avg_pace": avg_pace_str, "longest": longest,
            "lt10": lt10, "gt10": gt10
        }

    # ---- UI: Overall ----
    st.markdown("#### 🌍 Overall Walking (All Years) — Quick & Overall View")
    A = _walking_kpis(walk_all)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: kpi_card("Total Walks", f"{A['walks']:,}" if A["walks"] else "0")
    with c2: kpi_card("Total Distance", f"{A['km']:,.1f} km" if A["km"] else "0 km")
    with c3: kpi_card("Total Time", _dur_str(A["sec"]) if A["sec"] else "—")
    with c4: kpi_card("Average Pace", A["avg_pace"])
    with c5: kpi_card("Longest Walk", f"{A['longest']:.1f} km" if A["longest"] else "0 km")

    c6, c7, c8, c9 = st.columns(4)
    with c6: kpi_card("Walks < 10K", f"{A['lt10']:,}")
    with c7: kpi_card("Walks > 10K", f"{A['gt10']:,}")
    with c8: st.write("")  # spacer
    with c9: st.write("")  # spacer

    soft_divider()

    # ---- UI: Selected Year ----
    st.markdown(f"#### 🗓️ Walking in {year_label}")
    Y = _walking_kpis(walk_year)

    d1, d2, d3, d4, d5 = st.columns(5)
    with d1: kpi_card("Total Walks", f"{Y['walks']:,}" if Y["walks"] else "0")
    with d2: kpi_card("Total Distance", f"{Y['km']:,.1f} km" if Y["km"] else "0 km")
    with d3: kpi_card("Total Time", _dur_str(Y["sec"]) if Y["sec"] else "—")
    with d4: kpi_card("Average Pace", Y["avg_pace"])
    with d5: kpi_card("Longest Walk", f"{Y['longest']:.1f} km" if Y["longest"] else "0 km")

    d6, d7, d8, d9 = st.columns(4)
    with d6: kpi_card("Walks < 10K", f"{Y['lt10']:,}")
    with d7: kpi_card("Walks > 10K", f"{Y['gt10']:,}")
    with d8: st.write("")  # spacer
    with d9: st.write("")  # spacer

    # ---- Quick View: current filters ----

    walk_stats = df[df["Sport"] == "Walking"].copy()
    walk_charts = df_all_years[df_all_years["Sport"] == "Walking"].copy()  # ✅ charts always overall

    if walk_stats.empty:
        st.info("No Walking activities in current filters.")
    else:
        # ---------- Premium story + quote card ----------
        st.markdown(
            _html(
                """
                <div class="steps-story" style="margin-top:14px;">
                <div class="kicker">🚶 My walking journey — quiet progress, big impact</div>

                <div class="headline">
                    Walking is my “reset button” — recovery for the body, clarity for the mind
                </div>

                <div class="body">
                    <p><b>Why I walk:</b> On rest days and busy days, walking keeps me consistent without stress —
                    it supports recovery from cycling/running, improves focus, and protects long-term health.</p>

                    <p><b>What it taught me:</b> Progress doesn’t always need intensity. Sometimes it’s
                    just showing up — one calm step at a time — and letting habits do the heavy lifting.</p>

                    <p><b>My mindset:</b> If I can’t train hard today, I still move. Consistency wins.</p>
                </div>

                <div class="quote-box" style="margin-top:12px;">
                    <div class="quote-text">✨ “Small steps every day add up to big results.”</div>
                </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
    
    soft_divider()   # ✅ ADD THIS LINE
    st.subheader("Walking – Quick View (current filters)")
    # ---------- Charts (overall only) ----------
    if walk_charts.empty:
        st.info("No Walking data available for overall charts.")
    else:
        # If your generic_time_series already prints titles above charts (recommended),
        # keep it simple like this:
        generic_time_series(walk_charts, "Walking", key_prefix="walking_overall")
        distance_distribution(walk_charts, "Walking")




with tab_swim:
    soft_divider()   # ✅ ADD THIS LINE
    st.subheader("🏊 Swimming – Quick View")

    # --- Overall (All Years) + Selected Year (Swimming) ---
    swim_all = data[data["Sport"] == "Swimming"].copy()
    year_label, swim_year = _resolve_year_slice(swim_all, year_selection, year_col="Year")

    def _swim_kpis(x: pd.DataFrame) -> dict:
        if x is None or x.empty:
            return {"swims": 0, "km": 0.0, "sec": 0.0, "avg_km": 0.0}
        km = float(pd.to_numeric(x["Distance"], errors="coerce").sum() or 0.0)
        sec = float(pd.to_numeric(x["Moving Time"], errors="coerce").sum() or 0.0)
        swims = int(len(x))
        avg_km = (km / swims) if swims > 0 else 0.0
        return {"swims": swims, "km": km, "sec": sec, "avg_km": avg_km}

    # ---- UI: Overall ----
    st.markdown("#### 🌍 Overall Swimming (All Years) — Quick & Overall View")
    A = _swim_kpis(swim_all)

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Swims", f"{A['swims']:,}")
    with c2: kpi_card("Total Distance (km)", f"{A['km']:,.2f}")
    with c3: kpi_card("Total Time", _dur_str(A["sec"]) if A["sec"] else "—")
    with c4: kpi_card("Avg Distance / Swim", f"{A['avg_km']:,.2f} km" if A["avg_km"] else "—")

    soft_divider()
    
    # ---- UI: Selected Year ----
    st.markdown(f"#### 🗓️ Swimming in {year_label}")
    Y = _swim_kpis(swim_year)

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Swims", f"{Y['swims']:,}")
    with c2: kpi_card("Total Distance (km)", f"{Y['km']:,.2f}")
    with c3: kpi_card("Total Time", _dur_str(Y["sec"]) if Y["sec"] else "—")
    with c4: kpi_card("Avg Distance / Swim", f"{Y['avg_km']:,.2f} km" if Y["avg_km"] else "—")

    
    # ✅ Freeze charts to overall swimming (ignore year filter)
    swim = df_all_years[df_all_years["Sport"] == "Swimming"].copy()


    # --- Story + Quote (always show) ---
    st.markdown(
        dedent(
            """
            <div class="steps-story" style="margin-top:14px;">
              <div class="kicker">🏊 My swimming journey — learning phase (wishlist → reality)</div>
              <div class="headline">Swimming is on my “conquer next” list — one calm stroke at a time</div>
              <div class="body">
                <b>Where I am now:</b> I learned swimming for a short time and started getting comfortable in the water.<br/><br/>
                <b>Why it matters to me:</b> Swimming is the perfect low-impact skill — it builds endurance, breath control,
                and full-body strength, while being kind to the joints.<br/><br/>
                <b>What’s next:</b> I’ll keep showing up, improve technique step-by-step, and turn this wishlist into a confident habit.
                Just like every other journey — consistency will do the heavy lifting.
              </div>

              <div class="quote-box" style="margin-top:12px;">
                <div class="quote-text">✨ “The expert in anything was once a beginner.” — Helen Hayes</div>
              </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    # --- Charts only if data exists (current filters) ---
    if swim.empty:
        st.info("No Swimming activities in current filters (yet). The story stays — the data will catch up 🙂")
    else:
        st.markdown("#### Swimming – Trends & Distributions (Overall)")
        generic_time_series(swim, "Swimming", show_monthly=True, show_distance=True, show_elevation=False)



with tab_yoga:
    soft_divider()   # ✅ ADD THIS LINE
    st.subheader("➕ Yoga/Strength training – Quick View")

    # --- Overall (All Years) + Selected Year (Workout/Yoga/Strength) ---
    oth_all = data[data["Sport"] == "Workout"].copy()
    year_label, oth_year = _resolve_year_slice(oth_all, year_selection, year_col="Year")

    def _workout_kpis(x: pd.DataFrame) -> dict:
        if x is None or x.empty:
            return {"sessions": 0, "sec": 0.0, "avg_sec": 0.0}
        sessions = int(len(x))
        sec = float(pd.to_numeric(x["Moving Time"], errors="coerce").sum() or 0.0)
        avg_sec = (sec / sessions) if sessions > 0 else 0.0
        return {"sessions": sessions, "sec": sec, "avg_sec": avg_sec}

    st.markdown("#### 🌍 Overall Yoga/Strength training (All Years) — Quick & Overall View")
    A = _workout_kpis(oth_all)

    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Sessions", f"{A['sessions']:,}")
    with c2: kpi_card("Total Time", _dur_str(A["sec"]) if A["sec"] else "—")
    with c3: kpi_card("Avg Time / Session", _dur_str(A["avg_sec"]) if A["avg_sec"] else "—")

    soft_divider()

    st.markdown(f"#### 🗓️ Yoga/Strength training in {year_label}")
    Y = _workout_kpis(oth_year)

    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Sessions", f"{Y['sessions']:,}")
    with c2: kpi_card("Total Time", _dur_str(Y["sec"]) if Y["sec"] else "—")
    with c3: kpi_card("Avg Time / Session", _dur_str(Y["avg_sec"]) if Y["avg_sec"] else "—")

    # --- Current filters for charts only ---
    oth = df[df["Sport"] == "Workout"].copy()



    # --- Story + Quote (always show) ---
    st.markdown(
        _html(
            """
            <div class="steps-story" style="margin-top:14px;">
            <div class="kicker">🧘‍♂️ Yoga & Strength — the quiet work behind the strong days</div>
            <div class="headline">Flexibility, breathing, and durability — so the body stays ready</div>
            <div class="body">
                <b>Why I do this:</b> To keep muscles active and flexible, protect joints, and support recovery from cycling/running.<br/><br/>
                <b>Breath training:</b> I’m building better breathing control with <b>Pranayama</b> at least
                <b>4 days/week</b> for about <b>30 minutes</b> each session.<br/><br/>
                <b>Stability & posture:</b> I practice sitting in <b>Vajrasana</b> for around <b>20 minutes</b>
                to improve calm focus, posture, and consistency.<br/><br/>

                <b>What I also include (simple but powerful):</b>
                <ul style="margin: 6px 0 0 18px;">
                <li>Mobility / stretching (hips, hamstrings, calves)</li>
                <li>Core + glutes activation (injury prevention for running & cycling)</li>
                <li>Light strength work (bodyweight / resistance bands)</li>
                </ul>

                This is my “maintenance mode” — small practice, big impact over years.
            </div>

            <div class="quote-box" style="margin-top:12px;">
                <div class="quote-text">✨ “When the breath is steady, the mind is steady.”</div>
            </div>
            </div>
            """
        ),
        unsafe_allow_html=True,
    )


    # --- Charts only if data exists ---
    # --- Charts only if data exists ---
    oth_charts = df_all_years[df_all_years["Sport"] == "Workout"].copy()  # ✅ freeze graphs

    if oth_charts.empty:
        st.info("No Yoga/Strength/Workout activities in overall data (yet).")
    else:
        st.markdown("#### Yoga/Strength training – Trends & Distributions (Overall)")
        generic_time_series(
            oth_charts,
            "Yoga/Strength training",
            show_monthly=True,
            show_distance=False,
            show_elevation=False
        )


st.divider()
st.caption("Built with ❤️ in Streamlit. This project is intentionally modular to grow feature-by-feature.")
