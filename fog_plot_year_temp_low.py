"""視程が1km未満だった時の、年(横軸) × 温度(縦軸, 1℃ごと) のヒートマップ。夏季 (6〜8月) だけ。
上段: そのときの気温。 下段: そのときの海面水温 (官署に対応する海域)。
fog_plot.py と同じフォルダに置いて実行します (fog_plot.py の読み込み処理を使います)。

    python fog_plot_year_temp_low.py                       # 沿岸10官署を1枚ずつ画面に表示
    python fog_plot_year_temp_low.py --stations 宮古 酒田   # 官署を指定
    python fog_plot_year_temp_low.py --save-dir out        # 画面に出さずPNGを保存

色: 視程<1kmだった時刻の回数 (割合ではなく回数)。空白: 0回。全官署で同じ色の基準 (気温と海面水温は別)。
視程の記録は1989年から、2019年以前は間引かれている官署が多いので、古い年は回数が少なくなります。
縦軸の目盛りはマスの境目 (例: 16と17の間のマスは16.0℃以上17.0℃未満)。
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import fog_plot as fp


def low_table(values, years, lo, hi):
    """values: 視程<1kmだった時刻の温度 (index=時刻)。行=温度(整数℃), 列=年, 値=回数。"""
    v = values.dropna()
    cnt = (v.groupby([np.floor(v.values).astype(int), v.index.year]).size().unstack()
           .reindex(index=range(lo, hi + 1), columns=years).fillna(0))
    return cnt


def heat(ax, tab, title, vmax, cmap, ylabel):
    """色=回数。0回のマスは空白。縦のマスの辺が整数℃の位置に来る。"""
    years, lo, hi = list(tab.columns), tab.index.min(), tab.index.max()
    im = ax.imshow(np.where(tab.values > 0, tab.values, np.nan), origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                   extent=(years[0] - .5, years[-1] + .5, lo, hi + 1))
    ax.set_xticks(years[::2])
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
    d, _ = fp.fog_valid(fp.summer(fp.load(a.dir)), a)
    d = d[(d["fog"] == 1) & d["Ta"].notna() & d["SST"].notna()]
    d = d[d["time"].dt.month.isin([6, 7, 8]) | ((d["time"].dt.hour == 0) & (d["time"].dt.month == 9) & (d["time"].dt.day == 1))]
    names = [s for s in (a.stations or fp.STATION_SST) if s in set(d["station"])]
    years = list(range(a.from_year, int(d["time"].dt.year.max()) + 1))
    low = {s: d[d["station"] == s].set_index("time") for s in names}
    lo = int(np.floor(min(min(v["Ta"].min(), v["SST"].min()) for v in low.values())))
    hi = int(np.floor(max(max(v["Ta"].max(), v["SST"].max()) for v in low.values())))
    tabs = {s: (low_table(low[s]["Ta"], years, lo, hi), low_table(low[s]["SST"], years, lo, hi)) for s in names}
    vmax = [max(t[i].values.max() for t in tabs.values()) for i in (0, 1)]   # 気温・海面水温それぞれ、全官署で共通の色の基準
    cmap = plt.get_cmap("magma_r")
    note = "色=視程<1kmだった時刻の回数。空白=0回。縦軸の目盛りはマスの境目 (例: 16と17の間は16.0℃以上17.0℃未満)。"

    for s in names:
        fig, axs = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
        n = int(tabs[s][0].values.sum())
        ims = [heat(axs[0], tabs[s][0], f"{s}  視程<1kmだった時の気温 (6〜8月, 計{n}回)", vmax[0], cmap, "気温 (℃)"),
               heat(axs[1], tabs[s][1], f"視程<1kmだった時の海面水温 (海域: {fp.STATION_SST[s]})", vmax[1], cmap, "海面水温 (℃)")]
        axs[1].set_xlabel("年")
        fig.tight_layout(rect=(0, .03, .9, 1))
        for ax, im in zip(axs, ims):
            bb = ax.get_position()
            fig.colorbar(im, cax=fig.add_axes([.92, bb.y0 + .05 * bb.height, .015, .9 * bb.height])).set_label("視程<1kmだった回数")
        fig.text(.01, .004, note, fontsize=7)
        if a.save_dir:
            os.makedirs(a.save_dir, exist_ok=True)
            path = os.path.join(a.save_dir, f"fog_year_temp_low_{s}.png")
            fig.savefig(path, dpi=130)
            print("保存:", path)
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    main()
