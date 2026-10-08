"""霧日数(棒)に、「湿度と、日最低気温−海面水温」の条件を満たす日数(黒線)を重ねて表示する。
fog_plot.py と同じフォルダに置いて実行します (fog_plot.py の読み込み処理を使います)。

    python fog_days.py                       # 既定の条件で全官署
    python fog_days.py --stations 八戸 宮古     # 官署を選ぶ
    python fog_days.py --rh-stat mean --rh-thr 80 --dt-lo -1 --dt-hi 1   # 先生の条件を再現 (|日最低気温-SST|<=1)
    python fog_days.py --save out/fog_days.png

注意: 相対湿度の記録は2013年ごろに変わっていて(最大湿度98%以上の日の割合が約14%→約49%に急増)、
      それ以前は湿度の条件を使った日数が他の年と比べられません。このため黒線と相関は --from-year (既定2014) 以降だけです。

条件: 「日の相対湿度(--rh-stat: max/mean/min) が --rh-thr 以上」かつ
      「日の気温(--t-stat: min/mean/max) − 海面水温 が --dt-lo 以上 --dt-hi 以下」の日 (6〜8月) の年ごとの日数。
      既定: 日最大湿度98%以上、かつ 日最低気温−海面水温 −3℃以上 (上限なし)。
霧日数: 気象庁の月別値「霧日数」CSV (--kiri、既定は同じフォルダの tohoku.csv) の6〜8月の合計。
        均質番号(観測環境の変化)が変わった年に▲を付け、最新の区間を濃い緑にします。
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fog_plot as fp


def load_kiri(path):
    """月別の霧日数CSV(cp932/utf-8) -> 官署, 年, 6〜8月合計, 均質番号"""
    rows = fp._read_rows(path)
    names = rows[2]
    out = []
    for st in dict.fromkeys(n for n in names[1:] if n):
        c = next(i for i in range(1, len(names)) if names[i] == st)
        for r in rows[5:]:
            if not r or "/" not in r[0] or len(r) <= c + 2:
                continue
            y, m = map(int, r[0].split("/"))
            val = float(r[c]) if r[c] != "" and r[c + 1] == "8" else np.nan
            out.append((st, y, m, val, int(r[c + 2]) if r[c + 2].isdigit() else np.nan))
    d = pd.DataFrame(out, columns=["station", "yr", "mon", "kiri", "homog"])
    d = d[d["mon"].isin([6, 7, 8])]
    return d.groupby(["station", "yr"]).agg(
        kiri=("kiri", lambda s: s.sum() if len(s) == 3 and s.notna().all() else np.nan),
        homog=("homog", "max")).reset_index()


def daily_table(sst, coast):
    """官署ごと・日ごとの 湿度(最大/平均/最小)、気温(最低/平均/最高)、海面水温との差。1日20時刻以上そろった日だけ。"""
    c = coast[coast["station"].isin(fp.STATION_SST)].copy()
    c["day"] = (c["time"] - pd.Timedelta(hours=1)).dt.normalize()
    g = c.groupby(["station", "day"])
    d = pd.DataFrame({"n_ta": g.Ta.count(), "n_rh": g.RH.count(),
                      "rh_max": g.RH.max(), "rh_mean": g.RH.mean(), "rh_min": g.RH.min(),
                      "t_min": g.Ta.min(), "t_mean": g.Ta.mean(), "t_max": g.Ta.max()}).reset_index()
    d["area"] = d["station"].map(fp.STATION_SST)
    long = sst.stack().rename("SST").rename_axis(["day", "area"]).reset_index()
    d = d.merge(long, on=["day", "area"], how="left")
    d = d[(d.n_ta >= 20) & (d.n_rh >= 20) & d.SST.notna() & d.day.dt.month.isin([6, 7, 8])].copy()
    d["yr"] = d["day"].dt.year
    for s in ("min", "mean", "max"):
        d["dt_" + s] = d["t_" + s] - d["SST"]
    return d


def cond_days(d, rh_stat, rh_thr, t_stat, lo, hi):
    m = (d["rh_" + rh_stat] >= rh_thr) & (d["dt_" + t_stat] >= lo) & (d["dt_" + t_stat] <= hi)
    return d[m].groupby(["station", "yr"]).size()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=fp.HERE, help="海面水温 *.txt のあるフォルダ")
    p.add_argument("--coast-dir", default=fp.COAST_DIR)
    p.add_argument("--kiri", default=os.path.join(fp.HERE, "tohoku.csv"), help="霧日数の月別値CSV")
    p.add_argument("--stations", nargs="*", help="官署名 (省略で全官署)")
    p.add_argument("--rh-stat", choices=["max", "mean", "min"], default="max")
    p.add_argument("--rh-thr", type=float, default=98)
    p.add_argument("--t-stat", choices=["min", "mean", "max"], default="min")
    p.add_argument("--dt-lo", type=float, default=-3)
    p.add_argument("--dt-hi", type=float, default=np.inf)
    p.add_argument("--from-year", type=int, default=2014, help="黒線を引く・相関を取る最初の年 (湿度の記録が変わった後)")
    p.add_argument("--no-ref", action="store_true", help="先生の条件(灰色の破線)を重ねない")
    p.add_argument("--save", help="PNGの保存先 (省略で画面表示のみ)")
    a = p.parse_args()

    fp.setup_font()
    kiri = load_kiri(a.kiri)
    coast = fp.load_coast(a.coast_dir)
    if coast is None:
        raise SystemExit("沿岸官署のCSVが見つかりません: " + a.coast_dir)
    d = daily_table(fp.summer(fp.load(a.dir)), coast)
    mine = cond_days(d, a.rh_stat, a.rh_thr, a.t_stat, a.dt_lo, a.dt_hi)
    ref = cond_days(d, "mean", 80, "min", -1, 1)  # 先生の条件
    valid = d.groupby(["station", "yr"]).size()
    valid = valid[(valid >= 80) & (valid.index.get_level_values(1) >= a.from_year)]  # 92日のうち80日以上そろった年だけ
    label = (f"日{a.rh_stat}湿度≧{a.rh_thr:g}% かつ {a.dt_lo:g}℃≦日{a.t_stat}気温−SST"
             + (f"≦{a.dt_hi:g}℃" if np.isfinite(a.dt_hi) else ""))
    stations = a.stations or [s for s in fp.STATION_SST if s in set(kiri["station"])]

    fig, axs = plt.subplots(len(stations), 1, figsize=(14, 2.9 * len(stations)), squeeze=False)
    for ax, st in zip(axs.ravel(), stations):
        k = kiri[kiri.station == st].dropna(subset=["kiri"])
        last_h = k.homog.max()
        cur = k.homog == last_h
        ax.bar(k.yr[~cur], k.kiri[~cur], color="#b9e6b9", edgecolor="#777", lw=.5, label="霧日数(以前の区間)")
        ax.bar(k.yr[cur], k.kiri[cur], color="#2ecc40", edgecolor="k", lw=.6, label="霧日数(最新の区間)")
        for y in k.yr[k.homog.diff().fillna(0) != 0]:
            ax.plot(y, 0, "^", color="darkorange", mec="k", ms=10, clip_on=False, zorder=5)
        years = sorted(valid[valid.index.get_level_values(0) == st].index.get_level_values(1))
        if not a.no_ref:
            r = ref.reindex(pd.MultiIndex.from_product([[st], years]), fill_value=0)
            ax.plot(years, r.values, "--", color="gray", marker="o", ms=3, lw=1, label="先生の条件 (日平均湿度≧80% かつ |日最低気温−SST|≦1℃)")
        m = mine.reindex(pd.MultiIndex.from_product([[st], years]), fill_value=0)
        ax.plot(years, m.values, "-", color="k", marker="D", ms=4, lw=1.5, label=label)
        both = pd.DataFrame({"k": k.set_index("yr").kiri, "c": pd.Series(m.values, index=years)}).dropna()
        both = both[both.index.isin(k.yr[cur])]
        rr = both.k.corr(both.c) if len(both) > 4 and both.c.std() > 0 else np.nan
        ax.set_title(f"{st}   {both.index.min()}〜{both.index.max()}年の相関 r = {rr:.2f} (n={len(both)}年)" if np.isfinite(rr) else st,
                     loc="left", fontsize=11)
        ax.set_ylabel("日数")
        ax.set_xlim(1930, 2027)
        ax.grid(alpha=.3)
    axs.ravel()[0].legend(loc="upper left", fontsize=8, ncol=2)
    axs.ravel()[-1].set_xlabel("年")
    fig.tight_layout()
    if a.save:
        os.makedirs(os.path.dirname(os.path.abspath(a.save)), exist_ok=True)
        fig.savefig(a.save, dpi=110)
        print("保存:", a.save)
    plt.show()


if __name__ == "__main__":
    main()
