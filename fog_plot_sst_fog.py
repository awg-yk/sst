"""夏季 (6〜8月) の海面水温の平均 (横軸) と 公式の霧日数 (縦軸) の散布図を、官署ごとに示す図。1点=1年。
海面水温が高い年ほど霧日数が少ないかを見る。fog_plot.py と fog_days.py と同じフォルダに置いて実行します。

    python fog_plot_sst_fog.py                      # 官署ごとの図を画面に表示
    python fog_plot_sst_fog.py --highlight 2023     # 印をつける年 (既定2023)
    python fog_plot_sst_fog.py --save-dir out       # 画面に出さずPNGを保存

点: 濃い青=最新の均質区間 (観測環境が変わった最後の年以降)、薄い青=それより前の年。赤い点=--highlight の年。
直線は全部の年への直線の当てはめ。題の r は 全部の年 / 最新の均質区間だけ の相関係数。
霧日数は、観測環境が変わる年 (観測の自動化など) の前後で水準が変わるので、全部の年の相関には、その影響が混ざります。
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import fog_days as fd
import fog_plot as fp
from fog_plot_year import panels


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=fp.SST_DIR, help="海面水温 *.txt のあるフォルダ")
    p.add_argument("--kiri", default=os.path.join(fp.HERE, "東北地方気象官署月別値霧日数.csv"), help="霧日数の月別値CSV")
    p.add_argument("--highlight", type=int, default=2023, help="赤で印をつける年")
    p.add_argument("--save-dir", help="画面に出さずPNGをこのフォルダに保存する")
    a = p.parse_args()

    fp.setup_font()
    kiri = fd.load_kiri(a.kiri)
    sst = fp.summer(fp.load(a.dir))
    yearly = {}
    for s, area in fp.STATION_SST.items():
        k = kiri[kiri["station"] == s].dropna(subset=["kiri"]).set_index("yr")
        m = sst[area].groupby(sst.index.year).mean().dropna()
        yrs = sorted(set(k.index) & set(m.index))
        if len(yrs) >= 10:
            yearly[s] = (m.loc[yrs], k.loc[yrs, "kiri"], k.loc[yrs, "homog"])
    names = list(yearly)

    fig, axs = panels(names)
    for ax in axs.ravel():
        ax.sharex(axs.ravel()[0])
    print("官署別: 相関係数 r (全部の年 / 最新の均質区間)")
    for ax, s in zip(axs.ravel(), names):
        x, y, h = yearly[s]
        last = (h == h.max()).values
        r_all = np.corrcoef(x, y)[0, 1]
        r_last = np.corrcoef(x[last], y[last])[0, 1] if last.sum() >= 5 else np.nan
        ax.scatter(x[~last], y[~last], s=18, color="#a6c8e0", zorder=2)
        ax.scatter(x[last], y[last], s=18, color="#0077bb", zorder=3)
        if a.highlight in x.index:
            ax.scatter([x[a.highlight]], [y[a.highlight]], s=40, color="#cc3311", zorder=4)
            ax.annotate(str(a.highlight), (x[a.highlight], y[a.highlight]), textcoords="offset points", xytext=(4, 4), fontsize=8, color="#cc3311")
        slope, icpt = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, slope * xs + icpt, color="gray", lw=1.2, zorder=1)
        print(f"  {s}: {r_all:+.2f} / {r_last:+.2f}  ({len(x)}年, 最新区間 {last.sum()}年)")
        ax.set_title(f"{s} (海域: {fp.STATION_SST[s]})\nr={r_all:+.2f} / 最新区間 r={r_last:+.2f}", fontsize=9)
        ax.grid(alpha=.3)
    for ax in axs[-1, :]:
        ax.set_xlabel("夏季の海面水温の平均 (℃)")
    for ax in axs[:, 0]:
        ax.set_ylabel("霧日数 (6〜8月の合計, 日)")
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.text(.01, .004, f"1点=1年。濃い青=最新の均質区間、薄い青=それより前 (観測環境が変わる前)、赤={a.highlight}年。直線=全部の年の当てはめ。題の r=全部の年 / 最新の均質区間。\n"
             "海面水温は各官署に対応する海域の6〜8月の平均。霧日数は気象庁の公式値 (観測の自動化の前後で水準が変わる)。相関は因果を示さない。", fontsize=8)
    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)
        path = os.path.join(a.save_dir, "fog_sst_fog.png")
        fig.savefig(path, dpi=130)
        print("保存:", path)
    else:
        plt.show()


if __name__ == "__main__":
    main()
