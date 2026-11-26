"""Figures for the README, the X posts and the email to JIJ (English and Japanese versions).

    python docs/figures/make_figures.py            # all figures, EN + JA, into docs/figures/
    python docs/figures/make_figures.py --refresh  # also re-measure the decode timings

1. policies   - October 2026: share of the hindsight-optimal saving per policy (forecasting/results/policies.csv)
2. peak_day   - the hottest day: price, grid import with / without battery, battery charge (fresh solve)
3. decode     - ommx-highs-adapter decode time vs number of variables (Jij-Inc/ommx#1431)

Colours follow the dataviz reference palette (validated: categorical slots 1-3, light surface).
"""

import argparse
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent
DATA = OUT / "data"

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1, 2
NEUTRAL_BAR, NEUTRAL_DARK = "#c3c2b7", "#52514e"

plt.rcParams.update(
    {
        "font.family": ["Hiragino Sans", "DejaVu Sans"],
        "font.size": 10.5,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "axes.titlelocation": "left",
    }
)

T = {
    "en": {
        "p_title": "Hourly re-planning on forecasts captures 79% of the best possible saving",
        "p_sub": "Tokyo solar + battery site, October 2026, real JEPX prices and TEPCO demand. "
        "Saving vs no battery, as a share of the hindsight optimum.",
        "p_rows": [
            "Hand-written rule",
            "Re-planning + yesterday as forecast",
            "Re-planning + gradient boosting",
            "Re-planning + perfect forecast",
            "Hindsight optimum",
            "…without published JEPX prices",
            "…without real-time peak guard",
        ],
        "p_ablation": "What each piece is worth (gradient boosting, one piece removed)",
        "p_x": "Share of the best possible saving",
        "d_title": "On the hottest day, the battery holds grid import flat and shaves the evening peak",
        "d_price": "Import price (¥/kWh)",
        "d_grid": "Grid import (kW)",
        "d_soc": "Battery charge (kWh)",
        "d_without": "without battery",
        "d_with": "with battery",
        "d_peak": "peak",
        "d_floor": "10% reserve",
        "d_hour": "Hour of day",
        "c_title": "ommx-highs-adapter: reading the solution back is O(n²)",
        "c_sub": "decode_to_state copies the whole solution vector once per variable (Jij-Inc/ommx#1431)",
        "c_cur": "current decode_to_state",
        "c_fix": "read col_value once (fix)",
        "c_model": "our JijModeling model, 11.5k variables:\ndecode {d:.2f} s vs MILP solve {s:.2f} s",
        "c_x": "Decision variables",
        "c_y": "Decode time (log scale)",
        "c_x4": "×4 per doubling",
        "c_x2": "×2 per doubling",
        "source": "Source: TEPCO area demand / PV, JEPX spot prices. BESS FDE Workbench (JijModeling + OMMX + HiGHS).",
    },
    "ja": {
        "p_title": "予測＋毎時再計画で、理論上の最大削減の79%を達成",
        "p_sub": "東京の太陽光＋蓄電池サイト、2026年10月、JEPX価格・TEPCO需給の実データ。\n"
        "蓄電池なしに対する削減額を、後知恵の最適解に対する割合で表示。",
        "p_rows": [
            "ルール運用",
            "毎時再計画＋前日値を予測に",
            "毎時再計画＋勾配ブースティング",
            "毎時再計画＋完全予測",
            "後知恵の最適解",
            "…JEPX公表価格を使わない場合",
            "…ピーク制御なしの場合",
        ],
        "p_ablation": "各要素の価値（勾配ブースティングから1つずつ外した場合）",
        "p_x": "理論上の最大削減に対する割合",
        "d_title": "最も暑い日：受電を平らに保ち、夕方のピークを削る",
        "d_price": "買電単価（円/kWh）",
        "d_grid": "系統からの受電（kW）",
        "d_soc": "蓄電池残量（kWh）",
        "d_without": "蓄電池なし",
        "d_with": "蓄電池あり",
        "d_peak": "ピーク",
        "d_floor": "下限10%",
        "d_hour": "時刻",
        "c_title": "ommx-highs-adapter：解の読み出しが O(n²)",
        "c_sub": "decode_to_state が変数ごとに解ベクトル全体をコピーしている（Jij-Inc/ommx#1431）",
        "c_cur": "現在の decode_to_state",
        "c_fix": "修正案：col_value を\n一度だけ取得",
        "c_model": "本プロジェクトの JijModeling モデル（約1.15万変数）：\n読み出し {d:.2f} 秒 ＞ MILP求解 {s:.2f} 秒",
        "c_x": "決定変数の数",
        "c_y": "読み出し時間（対数）",
        "c_x4": "倍にすると約4倍",
        "c_x2": "倍にすると約2倍",
        "source": "出典：TEPCO エリア需給・太陽光、JEPX スポット価格。BESS FDE Workbench（JijModeling + OMMX + HiGHS）",
    },
}


def footer(fig, text: str) -> None:
    fig.text(0.012, 0.012, text, fontsize=8, color=MUTED, ha="left", va="bottom")


# --------------------------------------------------------------------------- 1. policies


def fig_policies(lang: str) -> Path:
    t = T[lang]
    df = pd.read_csv(ROOT / "forecasting" / "results" / "policies.csv").set_index("policy")
    keys = [
        "Rule-based (charge cheap hours, discharge expensive)",
        "MPC + persistence",
        "MPC + gradient boosting",
        "MPC + perfect forecast",
        "Hindsight optimum (one solve, actual data)",
        "MPC + gradient boosting, JEPX prices forecast too",
        "MPC + gradient boosting, no peak guard",
    ]
    share = [float(df.loc[k, "share_of_hindsight_saving_pct"]) for k in keys]
    saving = [int(df.loc[k, "saving_vs_no_battery_yen"]) for k in keys]
    colors = [NEUTRAL_BAR, NEUTRAL_BAR, BLUE, NEUTRAL_BAR, NEUTRAL_DARK, "#e1e0d9", "#e1e0d9"]
    y = np.array([0, 1, 2, 3, 4, 5.9, 6.9])  # gap before the ablation group

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    fig.subplots_adjust(left=0.31, right=0.97, top=0.79, bottom=0.17)
    ax.barh(y, share, height=0.62, color=colors, edgecolor=SURFACE, linewidth=2)
    for yi, s, yen, c in zip(y, share, saving, colors, strict=True):
        weight = "bold" if c == BLUE else "normal"
        ax.text(s + 1.2, yi, f"{s:.0f}%   ¥{yen:,.0f}", va="center", fontsize=9, color=INK, weight=weight)
    ax.set_yticks(y, t["p_rows"])
    for lbl, c in zip(ax.get_yticklabels(), colors, strict=True):
        lbl.set_color(INK if c == BLUE else INK2)
        lbl.set_fontweight("bold" if c == BLUE else "normal")
    ax.tick_params(axis="y", length=0, labelsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 128)
    ax.set_xticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(t["p_x"], fontsize=9)
    ax.text(0, 5.25, t["p_ablation"], fontsize=8.5, color=MUTED, transform=ax.get_yaxis_transform(), ha="left")
    fig.text(0.012, 0.95, t["p_title"], fontsize=13.5, weight="bold", color=INK, ha="left", va="top")
    fig.text(0.012, 0.885, t["p_sub"], fontsize=8.5, color=INK2, ha="left", va="top", wrap=True)
    ax.xaxis.set_label_coords(0.4, -0.13)
    footer(fig, t["source"])
    out = OUT / f"policies_{lang}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- 2. peak day


def peak_day_data() -> dict:
    from core_model.formulation import BessParams
    from core_model.solver_runner import SolverOptions, solve_dispatch
    from service.pipeline import load_data

    df = load_data(ROOT / "data" / "japan_2026_hourly.csv")
    net = (df["load_kw"] - df["solar_kw"]).to_numpy()
    day_idx = int(np.argmax(net)) // 24  # day with the highest net demand of the year
    start = (day_idx - 1) * 24  # solve the day before, the day and the day after; plot the middle day
    window = df.iloc[start : start + 72].reset_index(drop=True)
    r, _ = solve_dispatch(window, BessParams(), options=SolverOptions(mip_rel_gap=1e-9))
    sl = slice(24, 48)
    return {
        "date": pd.Timestamp(window["timestamp"].iloc[24]),
        "price": window["import_price_yen_kwh"].to_numpy()[sl],
        "without": np.maximum(net[start : start + 72][sl], 0),
        "with": r.p_imp_kw[sl],
        "soc": r.soc_kwh[24:49],
        "floor": BessParams().default_soc_min_frac * BessParams().capacity_kwh,
    }


def fig_peak_day(lang: str, d: dict) -> Path:
    t = T[lang]
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.4), dpi=200, sharex=True, gridspec_kw={"height_ratios": [1, 1.6, 1.1]})
    fig.subplots_adjust(left=0.08, right=0.84, top=0.86, bottom=0.1, hspace=0.32)

    def ext(a):  # repeat the last hour so a step line reaches 24:00
        return np.append(a, a[-1])

    hours = np.arange(25)

    ax = axes[0]
    ax.step(hours, ext(d["price"]), where="post", color=INK2, linewidth=2)
    ax.set_title(t["d_price"], fontsize=9, color=INK2, loc="left", pad=4)
    ax.yaxis.grid(True)

    ax = axes[1]
    ax.step(hours, ext(d["without"]), where="post", color=NEUTRAL_BAR, linewidth=2)
    ax.step(hours, ext(d["with"]), where="post", color=BLUE, linewidth=2)
    p0, p1 = float(d["without"].max()), float(d["with"].max())
    for val, color in ((p0, MUTED), (p1, BLUE)):
        ax.axhline(val, color=color, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(24.3, p0, f"{t['d_without']}\n{t['d_peak']} {p0:,.0f} kW", color=INK2, fontsize=8.5, va="center")
    ax.text(24.3, p1, f"{t['d_with']}\n{t['d_peak']} {p1:,.0f} kW", color=INK, fontsize=8.5, va="center", weight="bold")
    ax.set_title(t["d_grid"], fontsize=9, color=INK2, loc="left", pad=4)
    ax.set_ylim(0, p0 * 1.12)
    ax.yaxis.grid(True)

    ax = axes[2]
    ax.plot(np.arange(25), d["soc"], color=ORANGE, linewidth=2, marker="o", markersize=3.5)
    ax.axhline(d["floor"], color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(24.3, d["floor"], t["d_floor"], color=INK2, fontsize=8.5, va="center")
    ax.set_title(t["d_soc"], fontsize=9, color=INK2, loc="left", pad=4)
    ax.set_ylim(0, 2100)
    ax.yaxis.grid(True)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3), [f"{h}:00" for h in range(0, 25, 3)])
    ax.set_xlabel(t["d_hour"], fontsize=9)

    cut = 100 * (1 - p1 / p0)
    title = t["d_title"]
    date = d["date"].strftime("%Y-%m-%d (%a)") if lang == "en" else d["date"].strftime("%Y年%-m月%-d日")
    fig.text(0.012, 0.965, title, fontsize=13.5, weight="bold", color=INK, ha="left", va="top")
    sub = (
        f"{date}: peak grid import {p0:,.0f} → {p1:,.0f} kW (−{cut:.0f}%). 2,000 kWh / 500 kW battery, 600 kW PV."
        if lang == "en"
        else f"{date}：受電ピーク {p0:,.0f} → {p1:,.0f} kW（−{cut:.0f}%）。蓄電池 2,000 kWh / 500 kW、太陽光 600 kW。"
    )
    fig.text(0.012, 0.915, sub, fontsize=9, color=INK2, ha="left", va="top")
    footer(fig, T[lang]["source"])
    out = OUT / f"peak_day_{lang}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- 3. decode


def measure_decode(sizes=(1_000, 2_000, 4_000, 8_000, 16_000)) -> pd.DataFrame:
    from ommx.v1 import DecisionVariable, Instance, State
    from ommx_highs_adapter import OMMXHighsAdapter

    rows = []
    for n in sizes:
        xs = [DecisionVariable.continuous(i, lower=0, upper=1) for i in range(n)]
        inst = Instance.from_components(
            decision_variables=xs, objective=sum(x for x in xs), constraints=[], sense=Instance.MAXIMIZE
        )
        adapter = OMMXHighsAdapter(inst)
        model = adapter.solver_input
        model.setOptionValue("output_flag", False)
        model.run()
        t0 = time.perf_counter()
        state = adapter.decode_to_state(model)
        t_cur = time.perf_counter() - t0
        t_fix = float("inf")
        for _ in range(7):  # best of 7: the fix takes milliseconds, so one GC pause would dominate
            t0 = time.perf_counter()
            col = model.getSolution().col_value
            fixed = State(entries={v.id: col[i] for i, v in enumerate(adapter.instance.used_decision_variables)})
            t_fix = min(t_fix, time.perf_counter() - t0)
        assert fixed.entries == state.entries
        rows.append({"n": n, "current_sec": t_cur, "fixed_sec": t_fix})
        print(f"  decode n={n:>6}: {t_cur:.3f}s vs {t_fix:.4f}s")
    return pd.DataFrame(rows)


# Measured with reports/ommx-highs-decode/repro_bess_model.py (April 2026, 30-min month).
MODEL_POINT = {"n": 11_523, "decode_sec": 1.607, "solve_sec": 1.228}


def fig_decode(lang: str, df: pd.DataFrame) -> Path:
    t = T[lang]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
    fig.subplots_adjust(left=0.11, right=0.8, top=0.8, bottom=0.16)
    ax.loglog(df["n"], df["current_sec"], color=BLUE, marker="o", markersize=6)
    ax.loglog(df["n"], df["fixed_sec"], color=ORANGE, marker="o", markersize=6)
    ax.text(
        df["n"].iloc[-1] * 1.12,
        df["current_sec"].iloc[-1],
        f"{t['c_cur']}\n{t['c_x4']}",
        color=INK,
        fontsize=8.5,
        va="center",
    )
    ax.text(
        df["n"].iloc[-1] * 1.12,
        df["fixed_sec"].iloc[-1],
        f"{t['c_fix']}\n{t['c_x2']}",
        color=INK,
        fontsize=8.5,
        va="center",
    )
    m = MODEL_POINT
    ax.plot(
        [m["n"]],
        [m["decode_sec"]],
        marker="D",
        markersize=8,
        color=BLUE,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        linestyle="none",
    )
    ax.annotate(
        t["c_model"].format(d=m["decode_sec"], s=m["solve_sec"]),
        xy=(m["n"], m["decode_sec"]),
        xytext=(1_050, 0.6),
        fontsize=8.5,
        color=INK2,
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 1},
    )
    ax.set_xlabel(t["c_x"], fontsize=9)
    ax.set_ylabel(t["c_y"], fontsize=9)
    ax.set_xticks([1_000, 2_000, 4_000, 8_000, 16_000], ["1k", "2k", "4k", "8k", "16k"])
    ax.minorticks_off()
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1, 10], ["1 ms", "10 ms", "0.1 s", "1 s", "10 s"])
    ax.set_ylim(2e-4, 10)
    ax.grid(True, which="major")
    ax.set_axisbelow(True)
    fig.text(0.012, 0.95, t["c_title"], fontsize=13.5, weight="bold", color=INK, ha="left", va="top")
    fig.text(0.012, 0.885, t["c_sub"], fontsize=9, color=INK2, ha="left", va="top")
    footer(fig, "ommx 2.9.5, ommx-highs-adapter 2.9.5, highspy 1.16.0, Python 3.14, macOS arm64")
    out = OUT / f"decode_{lang}.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="Re-measure decode timings")
    args = ap.parse_args(argv)
    DATA.mkdir(exist_ok=True)

    timing_csv = DATA / "decode_timing.csv"
    if args.refresh or not timing_csv.exists():
        measure_decode().to_csv(timing_csv, index=False)
    timing = pd.read_csv(timing_csv)
    day = peak_day_data()

    for lang in ("en", "ja"):
        for path in (fig_policies(lang), fig_peak_day(lang, day), fig_decode(lang, timing)):
            print(f"Wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
