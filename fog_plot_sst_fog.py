"""夏(6〜8月)の平均海面水温 と 夏の霧日数(気象庁の公式値)の散布図 (ポスター用)。
海面水温は 1982年からなので、年代の区切りは既定で 1982 / 2004 の2つ (全期間をほぼ半分に分ける)。官署ごとに最寄りの海域を使う (fog_plot.STATION_SST)。
相関は因果を示さない。

使い方:
  py fog_plot_sst_fog.py --kiri 東北地方気象官署月別値霧日数.csv                  # 画面に出す
  py fog_plot_sst_fog.py --kiri ... --save-dir out                                # PNGを保存
  py fog_plot_sst_fog.py --kiri ... --periods 1982 2004 --stations 八戸 宮古 大船渡 石巻 小名浜
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import fog_days as fd
import fog_plot as fp
from fog_plot_ta_fog import poster_figure

STATIONS = ["八戸", "宮古", "大船渡", "石巻", "小名浜"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--kiri", default=os.path.join(fp.HERE, "東北地方気象官署月別値霧日数.csv"), help="霧日数の月別値CSV")
    p.add_argument("--periods", type=int, nargs="+", default=[1982, 2004], help="年代の区切りとなる開始年 (小さい順)。既定 1982 2004 (前半22年・後半23年)")
    p.add_argument("--stations", nargs="*", default=STATIONS, help="官署名 (既定: 太平洋側の5官署)")
    p.add_argument("--size-cm", type=float, nargs=2, default=[40, 22], metavar=("幅", "高さ"), help="貼り付ける大きさ (cm)。既定 40 22")
    p.add_argument("--fontsize", type=float, default=24, help="目盛りの文字の大きさ (pt)。既定24")
    p.add_argument("--save-dir", help="画面に出さずPNGをこのフォルダに保存する")
    a = p.parse_args()

    fp.setup_font()
    kiri = fd.load_kiri(a.kiri)
    sst = fp.summer(fp.load())
    mean = sst.groupby(sst.index.year).mean()      # 年 x 海域: 夏の平均海面水温
    data = {}
    for s in a.stations:
        area = fp.STATION_SST[s]
        k = kiri[kiri["station"] == s].dropna(subset=["kiri"]).set_index("yr")
        yrs = sorted(y for y in k.index if y in mean.index and not np.isnan(mean.loc[y, area]))
        if len(yrs) >= 10:
            data[s] = (np.array(yrs), np.array([mean.loc[y, area] for y in yrs]), k.loc[yrs, "kiri"].values, k.loc[yrs, "homog"].values)
            r = np.corrcoef(data[s][1], data[s][2])[0, 1]
            print(f"{s} ({area}): {len(yrs)}年, 相関係数 r={r:+.2f}")
    names = [s for s in a.stations if s in data]
    fig = poster_figure(data, names, sorted(a.periods), "夏季の平均海面水温", a)
    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)
        path = os.path.join(a.save_dir, "fog_sst_fog_poster.png")
        fig.savefig(path, dpi=300)
        print("保存:", path)
    else:
        plt.show()


if __name__ == "__main__":
    main()
