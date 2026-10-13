"""夏季 (6〜8月) の「気温 − 海面水温」の年ごとの平均を、官署ごとに示す図。気温差が年とともに広がっているかを見る。
fog_plot.py と同じフォルダに置いて実行します (fog_plot.py の読み込み処理を使います)。

    python fog_plot_dt_year.py                      # 沿岸10官署を並べた図を画面に表示
    python fog_plot_dt_year.py --from-year 1990     # この年以降だけ使う
    python fog_plot_dt_year.py --save-dir out       # 画面に出さずPNGを保存

各官署のグラフ: 点と線=その年の6〜8月の「気温 − 海面水温」の平均 (気温の観測時刻ごとに、その日の海面水温を引いた平均)。
              直線=年との直線の当てはめ。直線の傾きを凡例に ℃/10年 で書く (+なら年とともに気温が海面水温より高くなる方向)。
基準 0℃ (気温=海面水温) を点線で示す。
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fog_plot as fp


def yearly_dt(d, from_year):
    """官署ごとに、年ごとの 気温−海面水温 の平均 (Series, index=年)。"""
    d = d.dropna(subset=["Ta", "SST"])
    day = (d["time"] - pd.Timedelta(hours=1)).dt.normalize()   # 0時は前日24時
    d = d[day.dt.month.isin([6, 7, 8]) & (day.dt.year >= from_year)]
    return (d["Ta"] - d["SST"]).groupby(day.dt.year[d.index]).mean()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=fp.SST_DIR, help="海面水温 *.txt のあるフォルダ")
    p.add_argument("--coast-dir", default=fp.COAST_DIR, help="沿岸官署の時別値CSVのフォルダ")
    p.add_argument("--from-year", type=int, default=1982, help="この年以降を使う (既定 1982)")
    p.add_argument("--save-dir", help="画面に出さずPNGをこのフォルダに保存する")
    a = p.parse_args()

    fp.setup_font()
    sst = fp.summer(fp.load(a.dir))
    coast = fp.load_coast(a.coast_dir)
    if coast is None:
        raise SystemExit(f"沿岸官署のCSVが見つかりません: {a.coast_dir} (--coast-dir で指定)")
    d = fp.fog_dataset(sst, coast)
    names = [s for s in fp.STATION_SST if s in set(d["station"])]
    series = {s: yearly_dt(d[d["station"] == s], a.from_year) for s in names}
    lo = min(v.min() for v in series.values())
    hi = max(v.max() for v in series.values())

    ncol = (len(names) + 1) // 2
    fig, axs = plt.subplots(2, ncol, figsize=(3.6 * ncol + .5, 7.5), sharex=True, sharey=True, squeeze=False)
    print("官署別: 年ごとの 気温−海面水温 の傾き (℃/10年)")
    for ax, s in zip(axs.ravel(), names):
        v = series[s].dropna()
        ax.axhline(0, color="k", ls=":", lw=.8)
        ax.plot(v.index, v.values, "o-", ms=3, lw=.8, color="#4477aa")
        slope, icpt = np.polyfit(v.index, v.values, 1)
        ax.plot(v.index, slope * v.index + icpt, color="#cc3311", lw=1.5, label=f"傾き {slope * 10:+.2f}℃/10年")
        print(f"  {s}: {slope * 10:+.2f}")
        ax.set(title=f"{s} (海域: {fp.STATION_SST[s]})")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=.3)
    for ax in axs.ravel()[len(names):]:
        ax.axis("off")
    pad = (hi - lo) * .15
    axs[0, 0].set_ylim(lo - pad, hi + pad * 2.5)
    for ax in axs[:, 0]:
        ax.set_ylabel("気温 − 海面水温 の平均 (℃)")
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.text(.01, .004, "点線=気温と海面水温が等しい (0℃)。赤線=直線の当てはめで、傾きが+なら年とともに気温のほうが海面水温より高くなっている。"
             "\n夏季(6〜8月)の気温の観測時刻すべてについて、その日の海面水温を引いた値の年平均。1989年以前の気温は3時間おき。", fontsize=8)
    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)
        path = os.path.join(a.save_dir, "fog_dt_year.png")
        fig.savefig(path, dpi=130)
        print("保存:", path)
    else:
        plt.show()


if __name__ == "__main__":
    main()
