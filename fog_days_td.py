"""霧日数(棒)に、夏季(6〜8月)の「海面水温 − 露点温度」の年平均(折れ線, 右の軸)を重ねて表示する。官署ごとに1段。
海面水温 − 露点温度 が大きい(露点温度が海面水温より低い=空気が乾いている)年ほど霧日数が少なく、小さい年ほど多いはず、という見方を確かめる。
fog_plot.py と fog_days.py と fog_plot_td_sst.py と同じフォルダに置いて実行します。
露点温度は「東北地方沿岸気象官署時別値露点温度」フォルダ、霧日数は tohoku.csv (気象庁の月別値) を使います。

    python fog_days_td.py                     # 全官署
    python fog_days_td.py --stations 八戸 宮古   # 官署を選ぶ
    python fog_days_td.py --save out/fog_days_td.png

棒: 気象庁の霧日数 (6〜8月の合計)。 均質番号 (観測環境) が変わった年に▲を付け、最新の区間を濃い緑にします。
折れ線 (赤, 右の軸。上ほど小さい向きに反転してあり、棒と同じ向きに動けば「差が小さい年ほど霧日数が多い」): その年の6〜8月の、露点温度の観測時刻ごとに「その日の海面水温 − 露点温度」を求めた平均。
タイトルの r: 霧日数と折れ線の相関 (最新の区間 / --corr-from 年以降の最新の区間)。負なら「差が小さい年ほど霧日数が多い」。
注意: 相対湿度の記録は2013年ごろに変わっていて (点線)、露点温度はその影響を受けます。1989年以前 (灰色の帯) は3時間おきの観測です。
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fog_days as fd
import fog_plot as fp
import fog_plot_td_sst as td


def yearly_sst_minus_td(sst, coast, dew):
    """(官署, 年) ごとの 夏季の「海面水温 − 露点温度」の平均 (Series)。"""
    d = fp.fog_dataset(sst, coast)[["time", "station", "SST"]].merge(dew, on=["time", "station"], how="left")
    d = d.dropna(subset=["SST", "Td"])
    day = (d["time"] - pd.Timedelta(hours=1)).dt.normalize()   # 0時は前日24時
    d = d[day.dt.month.isin([6, 7, 8])]
    return (d["SST"] - d["Td"]).groupby([d["station"], day[d.index].dt.year]).mean().rename_axis(["station", "yr"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=fp.HERE, help="海面水温 *.txt のあるフォルダ")
    p.add_argument("--coast-dir", default=fp.COAST_DIR)
    p.add_argument("--dew-dir", default=td.DEW_DIR, help="露点温度の時別値CSVのフォルダ")
    p.add_argument("--kiri", default=os.path.join(fp.HERE, "tohoku.csv"), help="霧日数の月別値CSV")
    p.add_argument("--stations", nargs="*", help="官署名 (省略で全官署)")
    p.add_argument("--corr-from", type=int, default=2014, help="もう1つの相関を取る最初の年 (湿度の記録が変わった後)")
    p.add_argument("--save", help="PNGの保存先 (省略で画面表示のみ)")
    a = p.parse_args()

    fp.setup_font()
    kiri = fd.load_kiri(a.kiri)
    coast = fp.load_coast(a.coast_dir)
    if coast is None:
        raise SystemExit("沿岸官署のCSVが見つかりません: " + a.coast_dir)
    gap = yearly_sst_minus_td(fp.summer(fp.load(a.dir)), coast, td.load_dew(a.dew_dir))
    stations = [s for s in (a.stations or fp.STATION_SST) if s in set(kiri["station"])]

    fig, axs = plt.subplots(len(stations), 1, figsize=(14, 2.9 * len(stations)), squeeze=False)
    twins = []
    print("官署別: 霧日数と「海面水温−露点温度」の相関  最新の区間 / 最新の区間のうち corr-from 年以降")
    for ax, st in zip(axs.ravel(), stations):
        k = kiri[kiri.station == st].dropna(subset=["kiri"])
        cur = k.homog == k.homog.max()
        ax.axvspan(1981.5, 1989.5, color="#eeeeee", zorder=0)
        ax.axvline(2013.5, color="gray", ls=":", lw=1.2)
        ax.bar(k.yr[~cur], k.kiri[~cur], color="#b9e6b9", edgecolor="#777", lw=.5, label="霧日数(以前の区間)")
        ax.bar(k.yr[cur], k.kiri[cur], color="#2ecc40", edgecolor="k", lw=.6, label="霧日数(最新の区間)")
        for y in k.yr[k.homog.diff().fillna(0) != 0]:
            ax.plot(y, 0, "^", color="darkorange", mec="k", ms=10, clip_on=False, zorder=5)
        g = gap[gap.index.get_level_values(0) == st].droplevel(0)
        ax2 = ax.twinx()
        twins.append(ax2)
        ax2.plot(g.index, g.values, "-", color="#cc3311", marker="o", ms=3.5, lw=1.4, label="海面水温−露点温度 (右の軸、上ほど小さい)")
        ax2.invert_yaxis()   # 上が「差が小さい」: 差が小さい年ほど霧日数が多い関係なら、棒と折れ線が同じ向きに動く
        ax2.set_ylabel("海面水温−露点温度 (℃)\n(上ほど差が小さい=霧が出やすい)", color="#cc3311", fontsize=9)
        ax2.tick_params(axis="y", colors="#cc3311")
        both = pd.DataFrame({"k": k.set_index("yr").kiri, "g": g}).dropna()
        both = both[both.index.isin(k.yr[cur])]
        r1 = both.k.corr(both.g) if len(both) > 4 else np.nan
        late = both[both.index >= a.corr_from]
        r2 = late.k.corr(late.g) if len(late) > 4 else np.nan
        fmt = lambda r: f"{r:.2f}" if np.isfinite(r) else "-"
        print(f"  {st}: {fmt(r1)} (n={len(both)}) / {fmt(r2)} (n={len(late)})")
        ax.set_title(f"{st}   r = {fmt(r1)} (最新の区間 {both.index.min()}〜{both.index.max()}年, n={len(both)}) / "
                     f"{a.corr_from}年以降 r = {fmt(r2)} (n={len(late)})", loc="left", fontsize=10.5)
        ax.set_ylabel("霧日数 (日)")
        ax.set_xlim(1981, 2027)
        ax.grid(alpha=.3)
    # 縦軸の範囲を全官署で同じにする (左: 霧日数 0〜最大、右: 海面水温−露点温度 の全官署の最小〜最大。右は上ほど小さい向き)
    ymax = kiri[kiri.station.isin(stations)].kiri.max() * 1.05
    gg = gap[gap.index.get_level_values(0).isin(stations)]
    pad = (gg.max() - gg.min()) * .05
    for ax, ax2 in zip(axs.ravel(), twins):
        ax.set_ylim(0, ymax)
        ax2.set_ylim(gg.max() + pad, gg.min() - pad)
    h1, l1 = axs.ravel()[0].get_legend_handles_labels()
    h2, l2 = twins[0].get_legend_handles_labels()
    axs.ravel()[0].legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, ncol=3)
    axs.ravel()[-1].set_xlabel("年  (灰色の帯: 観測が3時間おき以下の年 / 点線: 湿度の記録が変わった年 / ▲: 観測環境が変わった年)")
    fig.tight_layout()
    if a.save:
        os.makedirs(os.path.dirname(os.path.abspath(a.save)), exist_ok=True)
        fig.savefig(a.save, dpi=110)
        print("保存:", a.save)
    plt.show()


if __name__ == "__main__":
    main()
