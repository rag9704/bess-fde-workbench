import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_jepx_hourly(jepx_csv: Path, freq: str = "1h") -> pd.DataFrame:
    """Parse JEPX spot_summary_YYYY.csv (30-min slots 1..48) into Tokyo area prices at `freq` ("1h" or "30min")."""
    df = pd.read_csv(jepx_csv, encoding="cp932")
    df = df.rename(
        columns={
            "受渡日": "date",
            "時刻コード": "slot",
            "エリアプライス東京(円/kWh)": "tokyo_price_yen_kwh",
        }
    )
    slot_start = pd.to_datetime(df["date"]) + pd.to_timedelta((df["slot"].astype(int) - 1) * 30, unit="min")
    df["timestamp"] = slot_start.dt.floor(freq)

    return df.groupby("timestamp", as_index=False)["tokyo_price_yen_kwh"].mean().round(3)


def load_tepco_area_hourly(
    area_csv: Path,
    site_peak_load_kw: float = 1000.0,
    site_solar_cap_kw: float = 600.0,
    freq: str = "1h",
) -> pd.DataFrame:
    """Parse TEPCO eria_jukyu CSV (utf-8-sig, MW) and scale normalized curves to a BTM site."""
    df = pd.read_csv(area_csv, encoding="utf-8-sig", skiprows=1)
    df["timestamp"] = pd.to_datetime(df["DATE"].str.strip() + " " + df["TIME"].str.strip())

    df["demand_mw"] = pd.to_numeric(df["エリア需要"], errors="coerce")
    df["solar_mw"] = pd.to_numeric(df["太陽光発電実績"], errors="coerce").fillna(0.0)

    hourly = df.set_index("timestamp")[["demand_mw", "solar_mw"]].resample(freq).mean().dropna().reset_index()

    hourly["load_kw"] = (hourly["demand_mw"] / hourly["demand_mw"].max() * site_peak_load_kw).round(1)
    hourly["solar_kw"] = (hourly["solar_mw"] / max(hourly["solar_mw"].max(), 1e-6) * site_solar_cap_kw).round(1)

    return hourly[["timestamp", "load_kw", "solar_kw"]]


def build_dataset(
    area_csv: Path,
    jepx_csv: Path,
    out_csv: Path,
    wheeling_fee_yen_kwh: float = 8.5,
    start_date: str | None = None,
    days: int | None = None,
    freq: str = "1h",
) -> pd.DataFrame:
    tepco = load_tepco_area_hourly(area_csv, freq=freq)
    jepx = load_jepx_hourly(jepx_csv, freq=freq)

    merged = pd.merge(tepco, jepx, on="timestamp", how="inner").sort_values("timestamp")

    if start_date:
        start_ts = pd.to_datetime(start_date)
        merged = merged[merged["timestamp"] >= start_ts]
        if days is not None:
            merged = merged[merged["timestamp"] < start_ts + pd.Timedelta(days=days)]
    elif days is not None and not merged.empty:
        first_ts = merged["timestamp"].min()
        merged = merged[merged["timestamp"] < first_ts + pd.Timedelta(days=days)]

    merged["import_price_yen_kwh"] = (merged["tokyo_price_yen_kwh"] + wheeling_fee_yen_kwh).round(2)
    merged["export_price_yen_kwh"] = np.minimum(merged["tokyo_price_yen_kwh"] * 0.7, 10.0).round(2)

    cols = [
        "timestamp",
        "load_kw",
        "solar_kw",
        "tokyo_price_yen_kwh",
        "import_price_yen_kwh",
        "export_price_yen_kwh",
    ]
    out = merged[cols].copy()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Wrote {len(out)} rows ({out['timestamp'].min()} -> {out['timestamp'].max()}) to {out_csv}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build site BESS time-series CSV from TEPCO & JEPX data.")
    parser.add_argument("--area-csv", type=Path, default=Path("data/raw/eria_jukyu_202604_03.csv"))
    parser.add_argument("--jepx-csv", type=Path, default=Path("data/raw/spot_summary_2026.csv"))
    parser.add_argument("--out-csv", type=Path, default=Path("data/april_2026_hourly.csv"))
    parser.add_argument("--wheeling-fee", type=float, default=8.5)
    parser.add_argument("--start-date", type=str, default=None, help="Optional YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=None, help="Optional number of days (e.g. 7 for a week)")
    parser.add_argument("--freq", choices=("1h", "30min"), default="1h", help="Time step (raw data is 30-min)")
    args = parser.parse_args()

    build_dataset(
        area_csv=args.area_csv,
        jepx_csv=args.jepx_csv,
        out_csv=args.out_csv,
        wheeling_fee_yen_kwh=args.wheeling_fee,
        start_date=args.start_date,
        days=args.days,
        freq=args.freq,
    )


if __name__ == "__main__":
    main()
