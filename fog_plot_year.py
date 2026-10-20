"""夏季 (6〜8月) の気温・露点温度・海面水温の年ごとの平均を、官署ごとに折れ線で示す図 (3種類から選ぶ)。
fog_plot.py と同じフォルダに置いて実行します (露点温度は「東北地方沿岸気象官署時別値」フォルダの時別値CSV、1984年〜)。

    python fog_plot_year.py                     # 3種類の図を画面に順に表示
    python fog_plot_year.py --what temps        # 気温・露点温度・海面水温の年平均だけ
    python fog_plot_year.py --what dt gap       # 「気温−海面水温」と「露点温度−海面水温」の年平均
    python fog_plot_year.py --what ta_long      # 夏の平均気温だけ、1930年代から (気象庁 月別値の3か月平均 「東北地方3か月平均.csv」を使う)
    python fog_plot_year.py --stations 八戸 宮古 大船渡 石巻 小名浜   # 官署を指定 (省略でデータのある官署すべて)
    python fog_plot_year.py --from-year 1990    # この年以降だけ使う
    python fog_plot_year.py --save-dir out      # 画面に出さずPNGを保存

--what:
    temps  気温 (赤)・海面水温 (青)・露点温度 (緑) の年平均。3つがそろった観測時刻だけで平均する。
    dt     「気温 − 海面水温」の年平均。気温の観測時刻すべてを使う (露点温度は使わない。1982年から)。
    ta_long 夏 (6〜8月) の平均気温だけの折れ線 (1930年代から)。気象庁 月別値の3か月平均 (東北地方3か月平均.csv) を使う。--from-year は使わない。
    gap    「露点温度 − 海面水温」の年平均 (空気が海面で冷やされる余地)。temps と同じ観測時刻。

各官署のグラフ: 点と線=年平均 (観測時刻ごとに、その日の海面水温を対応させて平均)。細い直線は年との直線の当てはめで、傾きを ℃/10年 で書く。
露点温度は気温と相対湿度から求められた値で、相対湿度の記録は2013年前後で変わっています (年平均の段差に注意)。1989年以前の観測は3時間おきです。
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fog_plot as fp

WHAT = ("temps", "dt", "gap", "ta_long")


def summer_days(d, from_year):
    """夏季 (6〜8月) の行だけにして、日付 (0時は前日24時) を返す。"""
    day = (d["time"] - pd.Timedelta(hours=1)).dt.normalize()
    ok = day.dt.month.isin([6, 7, 8]) & (day.dt.year >= from_year)
    return d[ok], day[ok]


def yearly_means3(d, from_year):
    """官署ひとつ分のデータから、年ごとの (気温, 露点温度, 海面水温) の平均。夏季の、3つがそろった時刻だけ。"""
    d, day = summer_days(d.dropna(subset=["Ta", "Td", "SST"]), from_year)
    return d[["Ta", "Td", "SST"]].groupby(day.dt.year).mean()


def yearly_dt(d, from_year):
    """官署ひとつ分のデータから、年ごとの 気温−海面水温 の平均 (Series)。夏季の、気温と海面水温がそろった時刻。"""
    d, day = summer_days(d.dropna(subset=["Ta", "SST"]), from_year)
    return (d["Ta"] - d["SST"]).groupby(day.dt.year).mean()


def trend_line(ax, m, col, color, label):
    ax.plot(m.index, m[col], "o-", ms=3, lw=.8, color=color)
    slope, icpt = np.polyfit(m.index, m[col], 1)
    ax.plot(m.index, slope * m.index + icpt, color=color, lw=1.6, alpha=.6, label=f"{label} {slope * 10:+.2f}℃/10年")
    return slope


def panels(names):
    ncol = (len(names) + 1) // 2
    fig, axs = plt.subplots(2, ncol, figsize=(3.6 * ncol + .5, 7.5), sharex=True, sharey=True, squeeze=False)
    for ax in axs.ravel()[len(names):]:
        ax.axis("off")
    return fig, axs


def temps_figure(names, means):
    fig, axs = panels(names)
    print("官署別: 傾き (℃/10年)  気温 / 露点温度 / 海面水温 / 露点温度−海面水温")
    for ax, s in zip(axs.ravel(), names):
        m = means[s]
        sa = trend_line(ax, m, "Ta", "#cc3311", "気温")
        ss = trend_line(ax, m, "SST", "#0077bb", "海面水温")
        st = trend_line(ax, m, "Td", "#228833", "露点温度")
        print(f"  {s}: {sa * 10:+.2f} / {st * 10:+.2f} / {ss * 10:+.2f} / {(st - ss) * 10:+.2f}")
        ax.set(title=f"{s} (海域: {fp.STATION_SST[s]})")
        ax.legend(fontsize=6.5, loc="upper left")
        ax.grid(alpha=.3)
    lo = min(v[["Ta", "Td", "SST"]].min().min() for v in means.values())
    hi = max(v[["Ta", "Td", "SST"]].max().max() for v in means.values())
    axs[0, 0].set_ylim(lo - 1, hi + 3.5)
    for ax in axs[:, 0]:
        ax.set_ylabel("夏季(6〜8月)の平均 (℃)")
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.text(.01, .004, "赤=気温、青=海面水温、緑=露点温度の年平均 (観測時刻ごとに、その日の海面水温を対応させて平均)。細い直線=直線の当てはめ、傾きは凡例。"
             "\n露点温度は気温と相対湿度から求められた値で、相対湿度の記録は2013年前後で変わっている。1989年以前の観測は3時間おき。", fontsize=8)
    return fig


def diff_figure(names, series, ylabel, note, ylim, label):
    """差の年平均の図。series: {官署: Series(年)}。ylim(lo, hi) は縦軸の範囲を返す関数。label は傾きの出力の見出し。"""
    fig, axs = panels(names)
    if label:
        print(label)
    for ax, s in zip(axs.ravel(), names):
        v = series[s].dropna()
        ax.axhline(0, color="k", ls=":", lw=.8)
        ax.plot(v.index, v.values, "o-", ms=3, lw=.8, color="#4477aa")
        slope, icpt = np.polyfit(v.index, v.values, 1)
        ax.plot(v.index, slope * v.index + icpt, color="#cc3311", lw=1.5, label=f"傾き {slope * 10:+.2f}℃/10年")
        if label:
            print(f"  {s}: {slope * 10:+.2f}")
        ax.set(title=f"{s} (海域: {fp.STATION_SST[s]})")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=.3)
    lo, hi = min(v.min() for v in series.values()), max(v.max() for v in series.values())
    axs[0, 0].set_ylim(*ylim(lo, hi))
    for ax in axs[:, 0]:
        ax.set_ylabel(ylabel)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.text(.01, .004, note, fontsize=8)
    return fig


def ta_long_figure(names, temps):
    """夏の平均気温 (月別値の3か月平均) の年ごとの折れ線。1930年代から。"""
    fig, axs = panels(names)
    print("官署別: 夏の平均気温の傾き (℃/10年)  全期間")
    for ax, s in zip(axs.ravel(), names):
        v = pd.Series(temps[(s, "平均気温")]).sort_index().dropna()
        m = pd.DataFrame({"Ta": v})
        slope = trend_line(ax, m, "Ta", "#cc3311", "気温")
        print(f"  {s}: {slope * 10:+.2f} ({v.index.min()}〜{v.index.max()}年, {len(v)}年)")
        ax.set(title=f"{s} ({v.index.min()}年〜)")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=.3)
    for ax in axs[:, 0]:
        ax.set_ylabel("夏季(6〜8月)の平均気温 (℃)")
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.text(.01, .004, "気象庁 月別値の3か月平均 (6〜8月の平均気温)。点と線=年ごとの値、細い直線=直線の当てはめで、傾きは凡例 (℃/10年)。", fontsize=8)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=fp.SST_DIR, help="海面水温 *.txt のあるフォルダ")
    p.add_argument("--coast-dir", default=fp.COAST_DIR, help="沿岸官署の時別値CSVのフォルダ")
    p.add_argument("--what", nargs="*", choices=WHAT, default=list(WHAT), help="描く図 (省略で3種類すべて)")
    p.add_argument("--stations", nargs="*", help="官署名 (省略で、データのある官署すべて)")
    p.add_argument("--temps", help="ta_long で使う3か月平均気温のCSV (省略で 東北地方3か月平均.csv)")
    p.add_argument("--from-year", type=int, default=1982, help="この年以降を使う (既定 1982。露点温度は1984年から)")
    p.add_argument("--save-dir", help="画面に出さずPNGをこのフォルダに保存する")
    a = p.parse_args()

    fp.setup_font()
    coast = fp.load_coast(a.coast_dir)
    if coast is None:
        raise SystemExit(f"沿岸官署のCSVが見つかりません: {a.coast_dir} (--coast-dir で指定)")
    d = fp.fog_dataset(fp.summer(fp.load(a.dir)), coast)
    names = [s for s in fp.STATION_SST if s in set(d["station"]) and (not a.stations or s in a.stations)]
    by = {s: d[d["station"] == s] for s in names}

    figs = []
    if "ta_long" in a.what:
        from fog_plot_ta_fog import TEMP_CSV, load_temps      # 循環importを避けるため、ここで読み込む
        figs.append((ta_long_figure(names, load_temps(a.temps or TEMP_CSV)), "fog_year_ta_long.png"))
    means = {}
    if "temps" in a.what or "gap" in a.what:
        means = {s: yearly_means3(by[s], a.from_year).dropna() for s in names}
    if "temps" in a.what:
        figs.append((temps_figure(names, means), "fog_year_temps.png"))
    if "dt" in a.what:
        pad = lambda lo, hi: ((hi - lo) * .15, (hi - lo) * .15 * 2.5)
        figs.append((diff_figure(
            names, {s: yearly_dt(by[s], a.from_year) for s in names}, "気温 − 海面水温 の平均 (℃)",
            "点線=気温と海面水温が等しい (0℃)。赤線=直線の当てはめで、傾きが+なら年とともに気温のほうが海面水温より高くなっている。"
            "\n夏季(6〜8月)の気温の観測時刻すべてについて、その日の海面水温を引いた値の年平均。1989年以前の気温は3時間おき。",
            lambda lo, hi: (lo - pad(lo, hi)[0], hi + pad(lo, hi)[1]), "官署別: 年ごとの 気温−海面水温 の傾き (℃/10年)"), "fog_year_dt.png"))
    if "gap" in a.what:
        figs.append((diff_figure(
            names, {s: means[s]["Td"] - means[s]["SST"] for s in names}, "露点温度 − 海面水温 の平均 (℃)",
            "点線=露点温度と海面水温が等しい (0℃)。赤線=直線の当てはめで、傾きが+なら年とともに露点温度が海面水温より高く (空気が海面で冷やされて飽和しやすく) なっている。"
            "\n露点温度の観測時刻ごとに、その日の海面水温を引いた値の年平均。相対湿度の記録は2013年前後で変わっている。",
            lambda lo, hi: (lo - .5, hi + 1.2), None), "fog_year_gap.png"))

    for fig, name in figs:
        if a.save_dir:
            os.makedirs(a.save_dir, exist_ok=True)
            path = os.path.join(a.save_dir, name)
            fig.savefig(path, dpi=130)
            print("保存:", path)
            plt.close(fig)
    if not a.save_dir:
        plt.show()


if __name__ == "__main__":
    main()
