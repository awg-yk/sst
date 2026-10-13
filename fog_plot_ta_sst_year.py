"""夏季 (6〜8月) の気温と海面水温の年ごとの平均を、官署ごとに折れ線で示す図。
fog_plot.py と同じフォルダに置いて実行します (fog_plot.py の読み込み処理を使います)。

    python fog_plot_ta_sst_year.py                  # 沿岸10官署を並べた図を画面に表示
    python fog_plot_ta_sst_year.py --from-year 1990 # この年以降だけ使う
    python fog_plot_ta_sst_year.py --save-dir out   # 画面に出さずPNGを保存

各官署のグラフ: 赤=気温の年平均、青=海面水温の年平均 (どちらも同じ観測時刻の値で平均)。細い直線は年との直線の当てはめで、
              傾きを凡例に ℃/10年 で書く。1989年以前の気温は3時間おき。
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fog_plot as fp


def yearly_means(d, from_year):
    """官署ひとつ分のデータから、年ごとの (気温の平均, 海面水温の平均) の DataFrame を返す。夏季の、両方がそろった時刻だけ。"""
    d = d.dropna(subset=["Ta", "SST"])
    day = (d["time"] - pd.Timedelta(hours=1)).dt.normalize()   # 0時は前日24時
    ok = day.dt.month.isin([6, 7, 8]) & (day.dt.year >= from_year)
    return d.loc[ok, ["Ta", "SST"]].groupby(day[ok].dt.year).mean()


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
    means = {s: yearly_means(d[d["station"] == s], a.from_year) for s in names}

    ncol = (len(names) + 1) // 2
    fig, axs = plt.subplots(2, ncol, figsize=(3.6 * ncol + .5, 7.5), sharex=True, sharey=True, squeeze=False)
    print("官署別: 傾き (℃/10年)  気温 / 海面水温")
    for ax, s in zip(axs.ravel(), names):
        m = means[s].dropna()
        for col, color, label in (("Ta", "#cc3311", "気温"), ("SST", "#0077bb", "海面水温")):
            ax.plot(m.index, m[col], "o-", ms=3, lw=.8, color=color)
            slope, icpt = np.polyfit(m.index, m[col], 1)
            ax.plot(m.index, slope * m.index + icpt, color=color, lw=1.6, alpha=.6, label=f"{label} {slope * 10:+.2f}℃/10年")
            print(f"  {s}: {label} {slope * 10:+.2f}")
        ax.set(title=f"{s} (海域: {fp.STATION_SST[s]})")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=.3)
    for ax in axs.ravel()[len(names):]:
        ax.axis("off")
    lo = min(v.min().min() for v in means.values())
    hi = max(v.max().max() for v in means.values())
    axs[0, 0].set_ylim(lo - 1, hi + 2.5)
    for ax in axs[:, 0]:
        ax.set_ylabel("夏季(6〜8月)の平均 (℃)")
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.text(.01, .004, "赤=気温、青=海面水温の年平均 (気温の観測時刻ごとに、その日の海面水温を対応させて平均)。細い直線=直線の当てはめ、傾きは凡例。"
             "\n1989年以前の気温は3時間おき。", fontsize=8)
    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)
        path = os.path.join(a.save_dir, "fog_ta_sst_year.png")
        fig.savefig(path, dpi=130)
        print("保存:", path)
    else:
        plt.show()


if __name__ == "__main__":
    main()
