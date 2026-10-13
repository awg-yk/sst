"""霧日数(棒)に、「海面水温 − 露点温度」が条件を満たす時刻がある日の日数(黒線)を重ねて表示する。官署ごとに1段、縦軸は全官署で同じ範囲。
海面水温 − 露点温度 が小さい(海面で冷やされた空気が飽和に近づく)時刻がある日が多い年ほど、霧日数も多いはず、という見方を確かめる。
fog_plot.py と fog_days.py と fog_plot_td_sst.py と同じフォルダに置いて実行します。
露点温度は「東北地方沿岸気象官署時別値露点温度」フォルダ、霧日数は tohoku.csv (気象庁の月別値) を使います。

    python fog_days_td_cond.py                        # 既定の条件 (海面水温 − 露点温度 ≦ 0℃ の時刻が1時間以上ある日) で全官署
    python fog_days_td_cond.py --gap-max 1            # 条件を変える (海面水温 − 露点温度 ≦ 1℃)
    python fog_days_td_cond.py --gap-min -3 --gap-max 2 --min-hours 2   # 範囲と時間数を指定
    python fog_days_td_cond.py --sat-max 2            # さらに 気温 − 露点温度 ≦ 2℃ (空気が飽和に近い) を加える
    python fog_days_td_cond.py --sat-max 2 --dt-min -2 --dt-max 1     # さらに 気温 − 海面水温 が −2〜1℃ を加える
    python fog_days_td_cond.py --stations 八戸 宮古
    python fog_days_td_cond.py --save out/fog_days_td_cond.png

黒線: 6〜8月で、毎時の観測が次の条件をすべて満たす時刻が --min-hours 時間以上ある日の、年ごとの日数。
      ・--gap-min ≦ 海面水温 − 露点温度 ≦ --gap-max  (既定: 上限0℃、下限なし)
      ・気温 − 露点温度 ≦ --sat-max  (既定: 使わない。小さいほど空気が飽和に近い)
      ・--dt-min ≦ 気温 − 海面水温 ≦ --dt-max  (既定: 使わない)
      (露点温度の観測時刻ごとに、その日の海面水温を引く。海面水温が露点温度以下なら 差 ≦ 0 で、空気が海面で露点まで冷やされる状態)
棒: 気象庁の霧日数 (6〜8月の合計)。均質番号 (観測環境) が変わった年に▲を付け、最新の区間を濃い緑にします。
タイトルの r: 霧日数と黒線の相関 (最新の区間 / そのうち --corr-from 年以降)。正なら「条件の日が多い年ほど霧日数が多い」。
注意: 相対湿度の記録は2013年ごろに変わっていて (点線)、露点温度はその影響を受けます。1989年以前 (灰色の帯) は3時間おきの観測で、日数が少なめに出ます。
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fog_days as fd
import fog_plot as fp
import fog_plot_td_sst as td


def cond_days(sst, coast, dew, lo, hi, min_hours, sat_max=np.inf, dt_min=-np.inf, dt_max=np.inf):
    """(官署, 年) ごとの、「海面水温 − 露点温度」が lo〜hi で、気温 − 露点温度 ≦ sat_max、気温 − 海面水温が dt_min〜dt_max の時刻が
    min_hours 時間以上ある日の日数 (Series)。
    あわせて、露点温度と海面水温がそろった日が60日以上ある (官署, 年) の目印も返す。"""
    d = fp.fog_dataset(sst, coast)[["time", "station", "Ta", "SST"]].merge(dew, on=["time", "station"], how="left")
    d = d.dropna(subset=["SST", "Td"] + (["Ta"] if np.isfinite(sat_max) or np.isfinite(dt_min) or np.isfinite(dt_max) else []))
    d["day"] = (d["time"] - pd.Timedelta(hours=1)).dt.normalize()   # 0時は前日24時
    d = d[d["day"].dt.month.isin([6, 7, 8])]
    g = d["SST"] - d["Td"]
    hit = (g >= lo) & (g <= hi)
    if np.isfinite(sat_max):
        hit &= (d["Ta"] - d["Td"]) <= sat_max
    if np.isfinite(dt_min) or np.isfinite(dt_max):
        hit &= ((d["Ta"] - d["SST"]) >= dt_min) & ((d["Ta"] - d["SST"]) <= dt_max)
    d["hit"] = hit
    n = d.groupby(["station", "day"])["hit"].sum()
    ok = (n >= min_hours).reset_index(name="ok")
    ok["yr"] = ok["day"].dt.year
    days = ok[ok["ok"]].groupby(["station", "yr"]).size()
    have = d.groupby(["station", d["day"].dt.year])["day"].nunique()
    return days, have[have >= 60]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=fp.SST_DIR, help="海面水温 *.txt のあるフォルダ")
    p.add_argument("--coast-dir", default=fp.COAST_DIR)
    p.add_argument("--dew-dir", default=td.DEW_DIR, help="露点温度の時別値CSVのフォルダ")
    p.add_argument("--kiri", default=os.path.join(fp.HERE, "tohoku.csv"), help="霧日数の月別値CSV")
    p.add_argument("--stations", nargs="*", help="官署名 (省略で全官署)")
    p.add_argument("--gap-max", type=float, default=0, help="海面水温 − 露点温度 がこれ以下 (℃)")
    p.add_argument("--gap-min", type=float, default=-np.inf, help="海面水温 − 露点温度 がこれ以上 (℃)")
    p.add_argument("--sat-max", type=float, default=np.inf, help="気温 − 露点温度 がこれ以下 (℃)。省略で使わない")
    p.add_argument("--dt-min", type=float, default=-np.inf, help="気温 − 海面水温 がこれ以上 (℃)。省略で使わない")
    p.add_argument("--dt-max", type=float, default=np.inf, help="気温 − 海面水温 がこれ以下 (℃)。省略で使わない")
    p.add_argument("--min-hours", type=int, default=1, help="1日にこの時間数以上 条件を満たせば、その日を数える")
    p.add_argument("--corr-from", type=int, default=2014, help="もう1つの相関を取る最初の年 (湿度の記録が変わった後)")
    p.add_argument("--save", help="PNGの保存先 (省略で画面表示のみ)")
    a = p.parse_args()

    fp.setup_font()
    kiri = fd.load_kiri(a.kiri)
    coast = fp.load_coast(a.coast_dir)
    if coast is None:
        raise SystemExit("沿岸官署のCSVが見つかりません: " + a.coast_dir)
    days, have = cond_days(fp.summer(fp.load(a.dir)), coast, td.load_dew(a.dew_dir), a.gap_min, a.gap_max, a.min_hours,
                           a.sat_max, a.dt_min, a.dt_max)
    stations = [s for s in (a.stations or fp.STATION_SST) if s in set(kiri["station"])]
    parts = [(f"{a.gap_min:g}℃ ≦ " if np.isfinite(a.gap_min) else "") + f"海面水温−露点温度 ≦ {a.gap_max:g}℃"]
    if np.isfinite(a.sat_max):
        parts.append(f"気温−露点温度 ≦ {a.sat_max:g}℃")
    if np.isfinite(a.dt_min) or np.isfinite(a.dt_max):
        parts.append((f"{a.dt_min:g}℃ ≦ " if np.isfinite(a.dt_min) else "") + "気温−海面水温" + (f" ≦ {a.dt_max:g}℃" if np.isfinite(a.dt_max) else ""))
    label = " かつ ".join(parts) + f" の時刻が{a.min_hours}時間以上ある日の日数"
    ymax = max(kiri[kiri.station.isin(stations)].kiri.max(), days[days.index.get_level_values(0).isin(stations)].max()) * 1.05

    fig, axs = plt.subplots(len(stations), 1, figsize=(14, 2.9 * len(stations)), squeeze=False)
    print(f"条件: {label}")
    print("官署別: 霧日数と黒線の相関  最新の区間 / 最新の区間のうち corr-from 年以降")
    for ax, st in zip(axs.ravel(), stations):
        k = kiri[kiri.station == st].dropna(subset=["kiri"])
        cur = k.homog == k.homog.max()
        ax.axvspan(1981.5, 1989.5, color="#eeeeee", zorder=0)
        ax.axvline(2013.5, color="gray", ls=":", lw=1.2)
        ax.bar(k.yr[~cur], k.kiri[~cur], color="#b9e6b9", edgecolor="#777", lw=.5, label="霧日数(以前の区間)")
        ax.bar(k.yr[cur], k.kiri[cur], color="#2ecc40", edgecolor="k", lw=.6, label="霧日数(最新の区間)")
        for y in k.yr[k.homog.diff().fillna(0) != 0]:
            ax.plot(y, 0, "^", color="darkorange", mec="k", ms=10, clip_on=False, zorder=5)
        years = sorted(have[have.index.get_level_values(0) == st].index.get_level_values(1))
        m = days.reindex(pd.MultiIndex.from_product([[st], years]), fill_value=0)
        ax.plot(years, m.values, "-", color="k", marker="D", ms=4, lw=1.5, label=label)
        both = pd.DataFrame({"k": k.set_index("yr").kiri, "c": pd.Series(m.values, index=years)}).dropna()
        both = both[both.index.isin(k.yr[cur])]
        r1 = both.k.corr(both.c) if len(both) > 4 and both.c.std() > 0 else np.nan
        late = both[both.index >= a.corr_from]
        r2 = late.k.corr(late.c) if len(late) > 4 and late.c.std() > 0 else np.nan
        fmt = lambda r: f"{r:.2f}" if np.isfinite(r) else "-"
        print(f"  {st}: {fmt(r1)} (n={len(both)}) / {fmt(r2)} (n={len(late)})")
        ax.set_title(f"{st}   r = {fmt(r1)} (最新の区間 {both.index.min()}〜{both.index.max()}年, n={len(both)}) / "
                     f"{a.corr_from}年以降 r = {fmt(r2)} (n={len(late)})" if len(both) else st, loc="left", fontsize=10.5)
        ax.set_ylabel("日数")
        ax.set_xlim(1981, 2027)
        ax.set_ylim(0, ymax)
        ax.grid(alpha=.3)
    axs.ravel()[0].legend(loc="upper left", fontsize=8, ncol=2)
    axs.ravel()[-1].set_xlabel("年  (灰色の帯: 観測が3時間おき以下の年 / 点線: 湿度の記録が変わった年 / ▲: 観測環境が変わった年)")
    fig.tight_layout()
    if a.save:
        os.makedirs(os.path.dirname(os.path.abspath(a.save)), exist_ok=True)
        fig.savefig(a.save, dpi=110)
        print("保存:", a.save)
    plt.show()


if __name__ == "__main__":
    main()
