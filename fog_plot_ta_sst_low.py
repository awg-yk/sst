"""視程が1km未満になった時の、気温(横軸) × 海面水温(縦軸) の2次元ヒートマップ (1℃ごと)。
fog_plot.py と fog_plot_ta_sst.py と同じフォルダに置いて実行します。

    python fog_plot_ta_sst_low.py                   # 沿岸10官署を並べた図を画面に表示
    python fog_plot_ta_sst_low.py --each            # 官署ごとに1枚ずつ表示 (マス内に回数も表示)
    python fog_plot_ta_sst_low.py --each --save-dir out   # 官署ごとのPNGを保存
    python fog_plot_ta_sst_low.py --from-year 2014  # この年以降だけ使う

色: 視程<1kmだった時刻の回数。全官署で同じ色の基準。 空白: 視程<1kmは一度も無かった。
破線: 気温 = 海面水温。 区切りと目盛りは fog_plot_ta_sst.py と同じ (目盛りはマスの境目)。
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import fog_plot as fp
import fog_plot_ta_sst as ts


def heat(ax, k, title, norm, cmap, numbers):
    """色=視程<1kmだった時刻の回数。0回のマスは空白。"""
    lo, hi = k.index.min(), k.index.max()
    kv = k.values
    ext = (lo, hi + 1, lo, hi + 1)
    ax.imshow(np.where(kv > 0, kv, np.nan), origin="lower", extent=ext, aspect="equal", cmap=cmap, norm=norm)
    ax.plot([lo, hi + 1], [lo, hi + 1], "k--", lw=.8, alpha=.6)  # 気温 = 海面水温
    if numbers:
        for i in range(kv.shape[0]):
            for j in range(kv.shape[1]):
                if kv[i, j] > 0:
                    ax.text(lo + j + .5, lo + i + .5, f"{int(kv[i, j])}", ha="center", va="center", fontsize=6,
                            color="w" if norm(kv[i, j]) > .55 else "k")
    ax.set_xlim(lo, hi + 1)
    ax.set_ylim(lo, hi + 1)
    ax.set_xticks(range(lo, hi + 2, 1 if numbers else 5))
    ax.set_yticks(range(lo, hi + 2, 1 if numbers else 5))
    ax.tick_params(labelsize=7)
    ax.set(title=title, xlabel="気温 (℃)", ylabel="海面水温 (℃)")
    return ScalarMappable(norm=norm, cmap=cmap)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=fp.HERE, help="海面水温 *.txt のあるフォルダ")
    p.add_argument("--coast-dir", default=fp.COAST_DIR, help="沿岸官署の時別値CSVのフォルダ")
    p.add_argument("--from-year", type=int, help="この年以降だけ使う")
    p.add_argument("--hourly-only", action="store_true", help="官署ごとに、視程が毎時になった年以降だけを使う")
    p.add_argument("--each", action="store_true", help="官署ごとに1枚ずつ表示する")
    p.add_argument("--stations", nargs="*", help="官署名 (省略で沿岸10官署)")
    p.add_argument("--save-dir", help="--each のとき、画面に出さずPNGをこのフォルダに保存する")
    a = p.parse_args()

    fp.setup_font()
    d, _ = fp.fog_valid(fp.summer(fp.load(a.dir)), a)
    d = d[d["vis"].notna()]
    allst = [s for s in fp.STATION_SST if s in set(d["station"])]
    lo, hi = ts.temp_range(d)
    ks = {s: ts.ta_sst_table(d[d["station"] == s], lo, hi)[1] for s in allst}
    norm = Normalize(0, max(1, max(k.values.max() for k in ks.values())))   # 全官署で共通の色の基準
    cmap = plt.get_cmap("magma_r")
    names = [s for s in (a.stations or allst) if s in ks]
    title = lambda s: f"{s} (海域: {fp.STATION_SST[s]}) 視程<1km {int(ks[s].values.sum())}回"
    note = "色=視程<1kmだった時刻の回数。空白=視程<1kmは無かった。破線=気温と海面水温が等しい。目盛りはマスの境目 (各マスは例えば16.0℃以上17.0℃未満)"

    if a.each:
        for s in names:
            fig, ax = plt.subplots(figsize=(10, 8.5))
            sm = heat(ax, ks[s], title(s), norm, cmap, numbers=True)
            fig.colorbar(sm, ax=ax, shrink=.85).set_label("視程<1kmだった回数")
            fig.tight_layout(rect=(0, .05, 1, 1))
            fig.text(.01, .004, note, fontsize=7)
            if a.save_dir:
                os.makedirs(a.save_dir, exist_ok=True)
                path = os.path.join(a.save_dir, f"fog_ta_sst_low_{s}.png")
                fig.savefig(path, dpi=130)
                print("保存:", path)
                plt.close(fig)
            else:
                plt.show()
        return

    ncol = (len(names) + 1) // 2
    fig, axs = plt.subplots(2, ncol, figsize=(3.9 * ncol + 1, 8), squeeze=False)
    for ax, s in zip(axs.ravel(), names):
        sm = heat(ax, ks[s], title(s), norm, cmap, numbers=False)
    for ax in axs.ravel()[len(names):]:
        ax.axis("off")
    fig.tight_layout(rect=(0, .06, .93, 1))
    fig.text(.01, .004, note, fontsize=9)
    cax = fig.add_axes([.945, .2, .015, .6])
    fig.colorbar(sm, cax=cax).set_label("視程<1kmだった回数", fontsize=9)
    plt.show()


if __name__ == "__main__":
    main()
