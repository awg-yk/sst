"""気温と海面水温の年ごとの平均を、視程<1kmだった時刻だけ (既定)、または夜〜朝の時刻だけに絞って、官署ごとに折れ線で示す図。夏季 (6〜8月)。
fog_plot.py と同じフォルダに置いて実行します (fog_plot.py の読み込み処理を使います)。

    python fog_plot_ta_sst_year_low.py                  # 視程<1kmだった時刻だけの平均 (沿岸10官署を並べて表示)
    python fog_plot_ta_sst_year_low.py --night          # 夜〜朝 (21時〜翌9時) の時刻だけの平均
    python fog_plot_ta_sst_year_low.py --min-hours 20   # 視程<1kmの回数がこれ未満の年は点を打たない (既定 10)
    python fog_plot_ta_sst_year_low.py --save-dir out   # 画面に出さずPNGを保存

各官署のグラフ: 赤=気温、青=海面水温の年平均 (どちらも同じ時刻の値で平均)。細い直線は年との直線の当てはめで、傾きを凡例に ℃/10年 で書く。
視程の記録は1989年からなので、視程<1kmの図は1989年以降です (2019年以前は視程が間引かれている官署が多く、回数が少ない年があります)。
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fog_plot as fp


def yearly_means(d, from_year, night):
    """官署ひとつ分のデータから、年ごとの (気温の平均, 海面水温の平均, 時刻の数) の DataFrame を返す。
    視程<1kmだった時刻 (night=False) か、夜〜朝の時刻 (night=True) だけ。"""
    d = d.dropna(subset=["Ta", "SST"])
    day = (d["time"] - pd.Timedelta(hours=1)).dt.normalize()   # 0時は前日24時
    ok = day.dt.month.isin([6, 7, 8]) & (day.dt.year >= from_year)
    if night:
        h = d["time"].dt.hour   # 時別値は「その時までの1時間」なので 21〜24時, 1〜9時 が 夜〜朝
        ok &= (h >= 21) | (h <= 9)
    else:
        ok &= d["fog"] == 1
    g = d.loc[ok, ["Ta", "SST"]].groupby(day[ok].dt.year)
    return g.mean().assign(n=g.size())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=fp.HERE, help="海面水温 *.txt のあるフォルダ")
    p.add_argument("--coast-dir", default=fp.COAST_DIR, help="沿岸官署の時別値CSVのフォルダ")
    p.add_argument("--from-year", type=int, default=1982, help="この年以降を使う (既定 1982)")
    p.add_argument("--night", action="store_true", help="視程<1kmの代わりに、夜〜朝 (21時〜翌9時) の時刻だけで平均する")
    p.add_argument("--min-hours", type=int, default=10, help="年ごとの時刻の数がこれ未満なら、その年は描かない (既定 10)")
    p.add_argument("--save-dir", help="画面に出さずPNGをこのフォルダに保存する")
    a = p.parse_args()

    fp.setup_font()
    sst = fp.summer(fp.load(a.dir))
    coast = fp.load_coast(a.coast_dir)
    if coast is None:
        raise SystemExit(f"沿岸官署のCSVが見つかりません: {a.coast_dir} (--coast-dir で指定)")
    d = fp.fog_dataset(sst, coast)
    names = [s for s in fp.STATION_SST if s in set(d["station"])]
    means = {s: yearly_means(d[d["station"] == s], a.from_year, a.night) for s in names}
    what = "夜〜朝 (21時〜翌9時) の時刻" if a.night else "視程<1kmだった時刻"

    ncol = (len(names) + 1) // 2
    fig, axs = plt.subplots(2, ncol, figsize=(3.6 * ncol + .5, 7.5), sharex=True, sharey=True, squeeze=False)
    print(f"官署別: {what}だけの平均の傾き (℃/10年)  気温 / 海面水温 / 気温−海面水温")
    for ax, s in zip(axs.ravel(), names):
        total = int(means[s]["n"].sum())
        m = means[s][means[s]["n"] >= a.min_hours]   # 時刻の数が少ない年は描かない
        title = f"{s} (海域: {fp.STATION_SST[s]})"
        if len(m) >= 3:
            sl = {}
            for col, color, label in (("Ta", "#cc3311", "気温"), ("SST", "#0077bb", "海面水温")):
                ax.plot(m.index, m[col], "o-", ms=3, lw=.8, color=color)
                sl[col], icpt = np.polyfit(m.index, m[col], 1)
                ax.plot(m.index, sl[col] * m.index + icpt, color=color, lw=1.6, alpha=.6, label=f"{label} {sl[col] * 10:+.2f}℃/10年")
            print(f"  {s}: {sl['Ta'] * 10:+.2f} / {sl['SST'] * 10:+.2f} / {(sl['Ta'] - sl['SST']) * 10:+.2f}")
            ax.legend(fontsize=7, loc="upper left")
        ax.set_title(title + f"  計{total}時間", fontsize=10)
        if len(m) < 3:
            ax.text(.5, .5, f"{a.min_hours}時間以上ある年が少なく描けない", transform=ax.transAxes, ha="center", fontsize=8, color="#666")
        ax.grid(alpha=.3)
    for ax in axs.ravel()[len(names):]:
        ax.axis("off")
    vals = [v[v["n"] >= a.min_hours][["Ta", "SST"]] for v in means.values() if (v["n"] >= a.min_hours).any()]
    lo, hi = min(v.min().min() for v in vals), max(v.max().max() for v in vals)
    axs[0, 0].set_ylim(lo - 1, hi + 2.5)
    for ax in axs[:, 0]:
        ax.set_ylabel(f"{what}の平均 (℃)")
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.text(.01, .004, f"赤=気温、青=海面水温の、夏季(6〜8月)の{what}だけの年平均 (気温の観測時刻ごとに、その日の海面水温を対応させて平均)。"
             f"\n細い直線=直線の当てはめ、傾きは凡例。1年の時刻の数が{a.min_hours}未満の年は描いていない。タイトルの時間数は、描かなかった年も含めた時刻の合計。", fontsize=8)
    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)
        path = os.path.join(a.save_dir, "fog_ta_sst_year_" + ("night" if a.night else "low") + ".png")
        fig.savefig(path, dpi=130)
        print("保存:", path)
    else:
        plt.show()


if __name__ == "__main__":
    main()
