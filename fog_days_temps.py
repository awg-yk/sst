"""霧日数(棒)に、夏季(6〜8月)の気温・海面水温・露点温度の年平均(折れ線, 右の軸)を重ねて表示する。官署ごとに1段、縦軸は全官署で同じ範囲。
fog_plot.py と fog_days.py と同じフォルダに置いて実行します。
露点温度は「東北地方沿岸気象官署時別値」フォルダの時別値CSV、霧日数は tohoku.csv (気象庁の月別値) を使います。

    python fog_days_temps.py                     # 全官署
    python fog_days_temps.py --stations 八戸 宮古   # 官署を選ぶ
    python fog_days_temps.py --save out/fog_days_temps.png

棒: 気象庁の霧日数 (6〜8月の合計)。均質番号 (観測環境) が変わった年に▲を付け、最新の区間を濃い緑にします。
折れ線 (右の軸): 赤=気温、青=海面水温、緑=露点温度の、その年の6〜8月の平均。3つがそろった観測時刻だけで平均します
                (観測時刻ごとに、その日の海面水温を対応させる)。
注意: 相対湿度の記録は2013年ごろに変わっていて (点線)、露点温度はその影響を受けます。1989年以前 (灰色の帯) は3時間おきの観測です。
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

import fog_days as fd
import fog_plot as fp

LINES = (("Ta", "気温", "#cc3311"), ("SST", "海面水温", "#0077bb"), ("Td", "露点温度", "#228833"))


def yearly_means(sst, coast):
    """(官署, 年) ごとの 夏季の気温・海面水温・露点温度の平均 (DataFrame, 列 Ta SST Td)。3つがそろった時刻だけ。"""
    d = fp.fog_dataset(sst, coast)[["time", "station", "Ta", "SST", "Td"]]
    d = d.dropna(subset=["Ta", "SST", "Td"])
    day = (d["time"] - pd.Timedelta(hours=1)).dt.normalize()   # 0時は前日24時
    d = d[day.dt.month.isin([6, 7, 8])]
    return d[["Ta", "SST", "Td"]].groupby([d["station"], day[d.index].dt.year]).mean().rename_axis(["station", "yr"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=fp.SST_DIR, help="海面水温 *.txt のあるフォルダ")
    p.add_argument("--coast-dir", default=fp.COAST_DIR)
    p.add_argument("--kiri", default=os.path.join(fp.HERE, "tohoku.csv"), help="霧日数の月別値CSV")
    p.add_argument("--stations", nargs="*", help="官署名 (省略で全官署)")
    p.add_argument("--save", help="PNGの保存先 (省略で画面表示のみ)")
    a = p.parse_args()

    fp.setup_font()
    kiri = fd.load_kiri(a.kiri)
    coast = fp.load_coast(a.coast_dir)
    if coast is None:
        raise SystemExit("沿岸官署のCSVが見つかりません: " + a.coast_dir)
    means = yearly_means(fp.summer(fp.load(a.dir)), coast)
    stations = [s for s in (a.stations or fp.STATION_SST) if s in set(kiri["station"])]

    # 縦軸の範囲を全官署で同じにする (左: 霧日数 0〜最大、右: 気温・海面水温・露点温度の全官署の最小〜最大)
    ymax = kiri[kiri.station.isin(stations)].kiri.max() * 1.05
    mm = means[means.index.get_level_values(0).isin(stations)]
    tlo, thi = mm.min().min(), mm.max().max()
    pad = (thi - tlo) * .05

    fig, axs = plt.subplots(len(stations), 1, figsize=(14, 2.9 * len(stations)), squeeze=False)
    twins = []
    for ax, st in zip(axs.ravel(), stations):
        k = kiri[kiri.station == st].dropna(subset=["kiri"])
        cur = k.homog == k.homog.max()
        ax.axvspan(1981.5, 1989.5, color="#eeeeee", zorder=0)
        ax.axvline(2013.5, color="gray", ls=":", lw=1.2)
        ax.bar(k.yr[~cur], k.kiri[~cur], color="#b9e6b9", edgecolor="#777", lw=.5, label="霧日数(以前の区間)")
        ax.bar(k.yr[cur], k.kiri[cur], color="#2ecc40", edgecolor="k", lw=.6, label="霧日数(最新の区間)")
        for y in k.yr[k.homog.diff().fillna(0) != 0]:
            ax.plot(y, 0, "^", color="darkorange", mec="k", ms=10, clip_on=False, zorder=5)
        m = means[means.index.get_level_values(0) == st].droplevel(0)
        ax2 = ax.twinx()
        twins.append(ax2)
        for col, name, color in LINES:
            ax2.plot(m.index, m[col], "-", color=color, marker="o", ms=3, lw=1.3, label=f"{name} (右の軸)")
        ax2.set_ylabel("気温・水温 (℃)")
        ax2.set_ylim(tlo - pad, thi + pad)
        ax.set_title(st, loc="left", fontsize=11)
        ax.set_ylabel("霧日数 (日)")
        ax.set_xlim(1981, 2027)
        ax.set_ylim(0, ymax)
        ax.grid(alpha=.3)
    h1, l1 = axs.ravel()[0].get_legend_handles_labels()
    h2, l2 = twins[0].get_legend_handles_labels()
    axs.ravel()[0].legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, ncol=5)
    axs.ravel()[-1].set_xlabel("年  (灰色の帯: 観測が3時間おき以下の年 / 点線: 湿度の記録が変わった年 / ▲: 観測環境が変わった年)")
    fig.tight_layout()
    if a.save:
        os.makedirs(os.path.dirname(os.path.abspath(a.save)), exist_ok=True)
        fig.savefig(a.save, dpi=110)
        print("保存:", a.save)
    plt.show()


if __name__ == "__main__":
    main()
