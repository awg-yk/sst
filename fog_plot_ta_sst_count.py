"""気温(横軸) × 海面水温(縦軸) を1℃ごとに区切った、時間数 (該当した時間) の図を、年代ごとに並べて示す (官署ごとに1枚)。
視程<1kmの条件は使いません。色もマス内の数字も、その気温・海面水温の組み合わせだった時間 (夏季6〜8月、毎時) です。
年代が進むと、時間の多いマス (濃い部分) が高温側 (右上) へ移るかを見ます。
fog_plot.py と同じフォルダに置いて実行します。気温が毎時になる1990年以降だけを使います。

    python fog_plot_ta_sst_count.py                         # 官署ごとに1枚 (年代を横に並べる) を順に画面に表示
    python fog_plot_ta_sst_count.py --stations 宮古 八戸     # 官署を指定
    python fog_plot_ta_sst_count.py --periods 1990 2000 2010 2020   # 年代の区切り (この年から次の区切りの前年まで。既定は 1990 2000 2010 2020)
    python fog_plot_ta_sst_count.py --save-dir out          # 画面に出さずPNGを保存

色: 時間数 (全官署・全年代で同じ基準)。空白: 該当なし。破線: 気温 = 海面水温。目盛りはマスの境目 (1℃ごと)。
年代ごとの年数が違うので (2020年代は2026年まで)、年数が短い年代は、時間数が少なくなります (題に年数を書いています)。
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize

import fog_plot as fp


def count_table(d, lo, hi):
    """気温 (横) × 海面水温 (縦) を1℃ごとに区切った、該当した時間の表。範囲は lo〜hi ℃。"""
    tx = np.floor(d["Ta"]).astype(int)
    ty = np.floor(d["SST"]).astype(int)
    idx = pd.RangeIndex(lo, hi + 1)
    n = pd.crosstab(ty, tx).reindex(index=idx, columns=idx, fill_value=0)
    return n


def heat(ax, n, title, norm, cmap):
    lo, hi = n.index.min(), n.index.max()
    nv = n.values
    ext = (lo, hi + 1, lo, hi + 1)
    ax.imshow(np.where(nv > 0, nv, np.nan), origin="lower", extent=ext, aspect="equal", cmap=cmap, norm=norm)
    ax.plot([lo, hi + 1], [lo, hi + 1], "k--", lw=.8, alpha=.6)
    for i in range(nv.shape[0]):
        for j in range(nv.shape[1]):
            if nv[i, j] > 0:
                ax.text(lo + j + .5, lo + i + .5, f"{int(nv[i, j])}", ha="center", va="center", fontsize=5.5,
                        color="w" if norm(nv[i, j]) > .55 else "k")
    ax.set_xlim(lo, hi + 1)
    ax.set_ylim(lo, hi + 1)
    ax.set_xticks(range(lo, hi + 2, 1))
    ax.set_yticks(range(lo, hi + 2, 1))
    ax.tick_params(labelsize=7)
    ax.set(title=title, xlabel="気温 (℃)", ylabel="海面水温 (℃)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=fp.SST_DIR, help="海面水温 *.txt のあるフォルダ")
    p.add_argument("--coast-dir", default=fp.COAST_DIR, help="沿岸官署の時別値CSVのフォルダ")
    p.add_argument("--periods", type=int, nargs="+", default=[1990, 2000, 2010, 2020], help="年代の区切りとなる開始年 (小さい順)")
    p.add_argument("--stations", nargs="*", help="官署名 (省略で沿岸10官署)")
    p.add_argument("--save-dir", help="画面に出さずPNGをこのフォルダに保存する")
    a = p.parse_args()

    fp.setup_font()
    coast = fp.load_coast(a.coast_dir)
    if coast is None:
        raise SystemExit(f"沿岸官署のCSVが見つかりません: {a.coast_dir} (--coast-dir で指定)")
    d = fp.fog_dataset(fp.summer(fp.load(a.dir)), coast)
    d = d[d["Ta"].notna() & d["SST"].notna()]
    year = (d["time"] - pd.Timedelta(hours=1)).dt.year
    bounds = sorted(a.periods)
    last_year = int(year.max())
    keep = year >= bounds[0]
    d, year = d[keep], year[keep]
    pid = np.searchsorted(bounds, year.values, side="right") - 1
    allst = [s for s in fp.STATION_SST if s in set(d["station"])]
    lo = int(np.floor(min(d["Ta"].quantile(.001), d["SST"].quantile(.001))))
    hi = int(np.ceil(max(d["Ta"].quantile(.999), d["SST"].quantile(.999))))
    tabs = {(s, i): count_table(d[(d["station"] == s) & (pid == i)], lo, hi) for s in allst for i in range(len(bounds))}
    norm = Normalize(0, max(t.values.max() for t in tabs.values()))   # 全官署・全年代で共通の色の基準
    cmap = plt.get_cmap("magma_r")
    for s in [s for s in (a.stations or allst) if s in allst]:
        total = sum(tabs[(s, i)] for i in range(len(bounds)))
        ys, xs = np.where(total.values > 0)
        clo = max(lo, min(total.index[ys.min()], total.index[xs.min()]) - 1)
        chi = min(hi, max(total.index[ys.max()], total.index[xs.max()]) + 1)
        fig, axs = plt.subplots(1, len(bounds), figsize=(6.4 * len(bounds) + 1.2, 7), squeeze=False)
        for i in range(len(bounds)):
            end = bounds[i + 1] - 1 if i + 1 < len(bounds) else last_year
            n = tabs[(s, i)].loc[clo:chi, clo:chi]
            heat(axs[0, i], n, f"{s} {bounds[i]}〜{end}年 ({end - bounds[i] + 1}年)\n該当 {int(n.values.sum())}時間", norm, cmap)
            axs[0, i].title.set_fontsize(10)
        fig.tight_layout(rect=(0, .06, .94, 1))
        fig.text(.01, .004, "色とマス内の数字: その気温・海面水温の組み合わせだった時間 (夏季6〜8月、気温と海面水温がそろった時刻。視程の条件なし)。空白=該当なし。"
                 "\n破線=気温と海面水温が等しい。目盛りはマスの境目 (各マスは例えば16.0℃以上17.0℃未満)。年代ごとの年数が違うので、年数が短い年代は時間数が少なくなる。", fontsize=8)
        bb = axs[0, -1].get_position()
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=fig.add_axes([.955, bb.y0 + .1 * bb.height, .012, .8 * bb.height])).set_label("時間数 (時間)", fontsize=9)
        if a.save_dir:
            os.makedirs(a.save_dir, exist_ok=True)
            path = os.path.join(a.save_dir, f"fog_ta_sst_count_{s}.png")
            fig.savefig(path, dpi=130)
            print("保存:", path)
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    main()
