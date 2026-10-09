"""年(横軸) × 温度(縦軸, 1℃ごと) の表 (ヒートマップ)。夏季 (6〜8月) の気温と海面水温が、毎年どの温度にどれだけあったかを官署ごとに示す。
fog_plot.py と同じフォルダに置いて実行します (fog_plot.py の読み込み処理を使います)。

    python fog_plot_year_temp.py                       # 沿岸10官署を1枚ずつ画面に表示
    python fog_plot_year_temp.py --stations 宮古 酒田   # 官署を指定
    python fog_plot_year_temp.py --save-dir out        # 画面に出さずPNGを保存

上段: 気温 (時別値。1989年以前は3時間おき)。 下段: その官署に対応する海域の海面水温 (日別値)。
色: その年の6〜8月のうち、その温度だった割合(%)。年ごとに合計が100%。気温と海面水温で色の基準は別 (各段とも全官署で共通)。回数ではなく割合なので、観測の間隔が違う年も比べられる。
空白: その温度は一度も無かった。 区切り: 縦軸は1℃ごとで、目盛りの数字はマスの境目 (例: 16と17の間のマスは16.0℃以上17.0℃未満)。
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fog_plot as fp


def year_temp_table(values, years, lo, hi):
    """values: index=時刻, 値=温度 の Series。行=温度(lo〜hi の整数℃), 列=年, 値=その年のうちその温度だった割合(%)。"""
    v = values.dropna()
    t = pd.DataFrame({"year": v.index.year, "temp": np.floor(v.values).astype(int)})
    cnt = t.pivot_table(index="temp", columns="year", values="temp", aggfunc="count")
    cnt = cnt.reindex(index=range(lo, hi + 1), columns=years).fillna(0)
    total = cnt.sum()
    return cnt / total.where(total > 0) * 100


def heat(ax, tab, title, vmax, cmap, ylabel):
    years, lo, hi = list(tab.columns), tab.index.min(), tab.index.max()
    y0, y1 = years[0], years[-1]
    im = ax.imshow(np.where(tab.values > 0, tab.values, np.nan), origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                   extent=(y0 - .5, y1 + .5, lo, hi + 1))  # 縦のマスの辺が整数℃の位置に来る
    ax.set_xticks(years[::2] if len(years) > 25 else years)
    ax.set_yticks(range(lo, hi + 2, 2))
    ax.tick_params(labelsize=7)
    ax.tick_params(axis="x", rotation=90)
    ax.set(title=title, ylabel=ylabel)
    return im


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=fp.HERE, help="海面水温 *.txt のあるフォルダ")
    p.add_argument("--coast-dir", default=fp.COAST_DIR, help="沿岸官署の時別値CSVのフォルダ")
    p.add_argument("--from-year", type=int, default=1982, help="この年以降を表示する (既定 1982)")
    p.add_argument("--stations", nargs="*", help="官署名 (省略で沿岸10官署)")
    p.add_argument("--save-dir", help="画面に出さずPNGをこのフォルダに保存する")
    a = p.parse_args()

    fp.setup_font()
    sst = fp.summer(fp.load(a.dir))
    sst = sst[sst.index.year >= a.from_year]
    coast = fp.load_coast(a.coast_dir)
    if coast is None:
        raise SystemExit(f"沿岸官署のCSVが見つかりません: {a.coast_dir} (--coast-dir で指定)")
    coast = coast[coast["station"].isin(fp.STATION_SST)].copy()
    day = (coast["time"] - pd.Timedelta(hours=1)).dt.normalize()   # 0時は前日24時
    coast = coast[day.dt.month.isin([6, 7, 8]) & (day.dt.year >= a.from_year)]
    names = [s for s in (a.stations or fp.STATION_SST) if s in set(coast["station"])]
    years = list(range(a.from_year, int(max(sst.index.year.max(), coast["time"].dt.year.max())) + 1))

    ta = {s: coast[coast["station"] == s].set_index("time")["Ta"] for s in names}
    sea = {s: sst[fp.STATION_SST[s]] for s in names}
    lo = int(np.floor(min(min(v.min() for v in ta.values()), min(v.min() for v in sea.values()))))
    hi = int(np.floor(max(max(v.max() for v in ta.values()), max(v.max() for v in sea.values()))))
    tabs = {s: (year_temp_table(ta[s], years, lo, hi), year_temp_table(sea[s], years, lo, hi)) for s in names}
    vmax = [max(np.nanmax(pair[i].values) for pair in tabs.values()) for i in (0, 1)]   # 気温・海面水温それぞれ、全官署で共通の色の基準
    cmap = plt.get_cmap("YlGnBu")
    note = "色=その年の6〜8月のうち、その温度だった割合(%)。空白=その温度は無かった。縦軸の目盛りはマスの境目 (例: 16と17の間は16.0℃以上17.0℃未満)。"

    for s in names:
        fig, axs = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
        ims = [heat(axs[0], tabs[s][0], f"{s}  気温 (6〜8月)", vmax[0], cmap, "気温 (℃)"),
               heat(axs[1], tabs[s][1], f"海面水温 (海域: {fp.STATION_SST[s]}, 6〜8月)", vmax[1], cmap, "海面水温 (℃)")]
        axs[1].set_xlabel("年")
        fig.tight_layout(rect=(0, .03, .9, 1))
        for ax, im in zip(axs, ims):   # 上下で色の基準が違うので、それぞれに色の目盛りをつける
            bb = ax.get_position()
            fig.colorbar(im, cax=fig.add_axes([.92, bb.y0 + .05 * bb.height, .015, .9 * bb.height])).set_label("その年のうちの割合 (%)")
        fig.text(.01, .004, note, fontsize=7)
        if a.save_dir:
            os.makedirs(a.save_dir, exist_ok=True)
            path = os.path.join(a.save_dir, f"fog_year_temp_{s}.png")
            fig.savefig(path, dpi=130)
            print("保存:", path)
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    main()
