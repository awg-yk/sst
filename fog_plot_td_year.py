"""夏季 (6〜8月) の露点温度と海面水温の年ごとの平均を、官署ごとに折れ線で示す図。
1枚目: 露点温度と海面水温の年平均。 2枚目: 「露点温度 − 海面水温」の年平均 (霧の出やすさに関わる、空気が海面で冷やされる余地)。
fog_plot.py と fog_plot_td_sst.py と同じフォルダに置いて実行します (露点温度は「東北地方沿岸気象官署時別値露点温度」フォルダ、1984年〜)。

    python fog_plot_td_year.py                  # 2枚の図を画面に順に表示
    python fog_plot_td_year.py --from-year 1990 # この年以降だけ使う
    python fog_plot_td_year.py --save-dir out   # 画面に出さずPNGを保存

各官署のグラフ: 点と線=年平均 (露点温度の観測時刻ごとに、その日の海面水温を対応させて平均)。細い直線は年との直線の当てはめで、傾きを凡例に ℃/10年 で書く。
露点温度は気温と相対湿度から求められた値で、相対湿度の記録は2013年前後で変わっています (年平均の段差に注意)。
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fog_plot as fp
import fog_plot_td_sst as td


def yearly_means(d, from_year):
    """官署ひとつ分のデータから、年ごとの (露点温度の平均, 海面水温の平均) の DataFrame を返す。夏季の、両方がそろった時刻だけ。"""
    d = d.dropna(subset=["Td", "SST"])
    day = (d["time"] - pd.Timedelta(hours=1)).dt.normalize()   # 0時は前日24時
    ok = day.dt.month.isin([6, 7, 8]) & (day.dt.year >= from_year)
    return d.loc[ok, ["Td", "SST"]].groupby(day[ok].dt.year).mean()


def trend_line(ax, m, col, color, label):
    ax.plot(m.index, m[col], "o-", ms=3, lw=.8, color=color)
    slope, icpt = np.polyfit(m.index, m[col], 1)
    ax.plot(m.index, slope * m.index + icpt, color=color, lw=1.6, alpha=.6, label=f"{label} {slope * 10:+.2f}℃/10年")
    return slope


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=fp.HERE, help="海面水温 *.txt のあるフォルダ")
    p.add_argument("--coast-dir", default=fp.COAST_DIR, help="沿岸官署の時別値CSVのフォルダ")
    p.add_argument("--dew-dir", default=td.DEW_DIR, help="露点温度の時別値CSVのフォルダ")
    p.add_argument("--from-year", type=int, default=1982, help="この年以降を使う (既定 1982。露点温度は1984年から)")
    p.add_argument("--save-dir", help="画面に出さずPNGをこのフォルダに保存する")
    a = p.parse_args()

    fp.setup_font()
    coast = fp.load_coast(a.coast_dir)
    if coast is None:
        raise SystemExit(f"沿岸官署のCSVが見つかりません: {a.coast_dir} (--coast-dir で指定)")
    d = fp.fog_dataset(fp.summer(fp.load(a.dir)), coast).merge(td.load_dew(a.dew_dir), on=["time", "station"], how="left")
    names = [s for s in fp.STATION_SST if s in set(d["station"])]
    means = {s: yearly_means(d[d["station"] == s], a.from_year).dropna() for s in names}
    ncol = (len(names) + 1) // 2

    # 1枚目: 露点温度と海面水温
    fig, axs = plt.subplots(2, ncol, figsize=(3.6 * ncol + .5, 7.5), sharex=True, sharey=True, squeeze=False)
    print("官署別: 傾き (℃/10年)  露点温度 / 海面水温 / 露点温度−海面水温")
    for ax, s in zip(axs.ravel(), names):
        m = means[s]
        st = trend_line(ax, m, "Td", "#228833", "露点温度")
        ss = trend_line(ax, m, "SST", "#0077bb", "海面水温")
        print(f"  {s}: {st * 10:+.2f} / {ss * 10:+.2f} / {(st - ss) * 10:+.2f}")
        ax.set(title=f"{s} (海域: {fp.STATION_SST[s]})")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=.3)
    for ax in axs.ravel()[len(names):]:
        ax.axis("off")
    lo = min(v[["Td", "SST"]].min().min() for v in means.values())
    hi = max(v[["Td", "SST"]].max().max() for v in means.values())
    axs[0, 0].set_ylim(lo - 1, hi + 2.5)
    for ax in axs[:, 0]:
        ax.set_ylabel("夏季(6〜8月)の平均 (℃)")
    fig.tight_layout(rect=(0, .05, 1, 1))
    note = ("緑=露点温度、青=海面水温の年平均 (露点温度の観測時刻ごとに、その日の海面水温を対応させて平均)。細い直線=直線の当てはめ、傾きは凡例。"
            "\n露点温度は気温と相対湿度から求められた値で、相対湿度の記録は2013年前後で変わっている。1989年以前の観測は3時間おき。")
    fig.text(.01, .004, note, fontsize=8)
    figs = [(fig, "fog_td_year.png")]

    # 2枚目: 露点温度 − 海面水温
    fig2, axs2 = plt.subplots(2, ncol, figsize=(3.6 * ncol + .5, 7.5), sharex=True, sharey=True, squeeze=False)
    diff = {s: means[s]["Td"] - means[s]["SST"] for s in names}
    for ax, s in zip(axs2.ravel(), names):
        v = diff[s]
        ax.axhline(0, color="k", ls=":", lw=.8)
        ax.plot(v.index, v.values, "o-", ms=3, lw=.8, color="#4477aa")
        slope, icpt = np.polyfit(v.index, v.values, 1)
        ax.plot(v.index, slope * v.index + icpt, color="#cc3311", lw=1.5, label=f"傾き {slope * 10:+.2f}℃/10年")
        ax.set(title=f"{s} (海域: {fp.STATION_SST[s]})")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=.3)
    for ax in axs2.ravel()[len(names):]:
        ax.axis("off")
    lo, hi = min(v.min() for v in diff.values()), max(v.max() for v in diff.values())
    axs2[0, 0].set_ylim(lo - .5, hi + 1.2)
    for ax in axs2[:, 0]:
        ax.set_ylabel("露点温度 − 海面水温 の平均 (℃)")
    fig2.tight_layout(rect=(0, .05, 1, 1))
    fig2.text(.01, .004, "点線=露点温度と海面水温が等しい (0℃)。赤線=直線の当てはめで、傾きが+なら年とともに露点温度が海面水温より高く (空気が海面で冷やされて飽和しやすく) なっている。"
              "\n露点温度の観測時刻ごとに、その日の海面水温を引いた値の年平均。相対湿度の記録は2013年前後で変わっている。", fontsize=8)
    figs.append((fig2, "fog_td_sst_year.png"))

    for f, name in figs:
        if a.save_dir:
            os.makedirs(a.save_dir, exist_ok=True)
            path = os.path.join(a.save_dir, name)
            f.savefig(path, dpi=130)
            print("保存:", path)
        else:
            pass
    if not a.save_dir:
        plt.show()


if __name__ == "__main__":
    main()
