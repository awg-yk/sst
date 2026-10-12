"""気温(横軸) × 海面水温(縦軸) の視程<1km出現率マップ (fog_plot_ta_sst.py と同じ図) を、年代ごとに並べて示す図。
色の濃い部分が年代とともに減るか、右上(気温・海面水温が高いほう)へ移るかを見る。
fog_plot.py と fog_plot_ta_sst.py と同じフォルダに置いて実行します。

    python fog_plot_ta_sst_period.py                         # 官署ごとに1枚 (年代を横に並べる) を順に画面に表示
    python fog_plot_ta_sst_period.py --stations 宮古 八戸     # 官署を指定
    python fog_plot_ta_sst_period.py --periods 1989 2000 2010 2020   # 年代の区切り (この年から次の区切りの前年まで。既定は 1989 2000 2010 2020)
    python fog_plot_ta_sst_period.py --save-dir out          # 画面に出さずPNGを保存

色: 視程<1kmの出現率(%) = 視程<1kmだった回数 ÷ 該当した回数。全官署・全年代で同じ色の基準。
灰色: 視程<1kmが0回、または該当した回数が40回未満。空白: 該当なし。破線: 気温 = 海面水温。目盛りはマスの境目 (1℃ごと)。
視程の記録は1989年からです。2019年以前は視程が間引かれている官署が多く、年代ごとの該当した回数が少なくなります。
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import fog_plot as fp
import fog_plot_ta_sst as ts


def period_label(bounds, i, last_year):
    end = bounds[i + 1] - 1 if i + 1 < len(bounds) else last_year
    return f"{bounds[i]}〜{end}年"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=fp.HERE, help="海面水温 *.txt のあるフォルダ")
    p.add_argument("--coast-dir", default=fp.COAST_DIR, help="沿岸官署の時別値CSVのフォルダ")
    p.add_argument("--periods", type=int, nargs="+", default=[1989, 2000, 2010, 2020], help="年代の区切りとなる開始年 (小さい順)")
    p.add_argument("--stations", nargs="*", help="官署名 (省略で沿岸10官署)")
    p.add_argument("--save-dir", help="画面に出さずPNGをこのフォルダに保存する")
    a = p.parse_args()

    fp.setup_font()
    args = argparse.Namespace(coast_dir=a.coast_dir, from_year=None, hourly_only=False)
    d, _ = fp.fog_valid(fp.summer(fp.load(a.dir)), args)
    d = d[d["vis"].notna()]
    bounds = sorted(a.periods)
    year = d["time"].dt.year
    last_year = int(year.max())
    d = d[year >= bounds[0]]
    pid = np.searchsorted(bounds, d["time"].dt.year.values, side="right") - 1
    allst = [s for s in fp.STATION_SST if s in set(d["station"])]
    lo, hi = ts.temp_range(d)
    tabs = {(s, i): ts.ta_sst_table(d[(d["station"] == s) & (pid == i)], lo, hi) for s in allst for i in range(len(bounds))}
    norm = fp.fog_norm(list(tabs.values()))          # 全官署・全年代で共通の色の基準
    cmap = plt.get_cmap("magma_r")
    names = [s for s in (a.stations or allst) if s in allst]
    note = ("色=視程<1kmの出現率 (視程<1kmだった回数÷該当した回数)。灰色=視程<1kmが0回、または該当した回数が40回未満。空白=該当なし。"
            "\n破線=気温と海面水温が等しい。目盛りはマスの境目 (各マスは例えば16.0℃以上17.0℃未満)。2019年以前は視程が間引かれている官署が多い。")

    for s in names:
        fig, axs = plt.subplots(1, len(bounds), figsize=(4.4 * len(bounds) + 1, 5.6), squeeze=False, sharey=True)
        for i, ax in enumerate(axs[0]):
            n, k = tabs[(s, i)]
            sm = ts.heat(ax, n, k, f"{s} {period_label(bounds, i, last_year)}\n該当{int(n.values.sum())}時間, 視程<1km {int(k.values.sum())}時間",
                         norm, cmap, numbers=False)
            ax.title.set_fontsize(9)
            if i:
                ax.set_ylabel("")
        fig.tight_layout(rect=(0, .07, .93, 1))
        fig.text(.01, .004, note, fontsize=8)
        fig.colorbar(sm, cax=fig.add_axes([.945, .25, .012, .55])).set_label("視程<1kmの出現率 (%)", fontsize=9)
        if a.save_dir:
            os.makedirs(a.save_dir, exist_ok=True)
            path = os.path.join(a.save_dir, f"fog_ta_sst_period_{s}.png")
            fig.savefig(path, dpi=130)
            print("保存:", path)
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    main()
