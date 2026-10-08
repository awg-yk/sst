"""霧(視程<1km)の発生条件のグラフを描く。 sst_tool.py と同じフォルダに置いて実行します。

    python fog_plot.py                  # 全期間
    python fog_plot.py --from-year 2020 # 視程が毎時になった2020年以降だけ

出力 (out/ フォルダ):
    fog_distribution.png            霧の時/霧でない時の「気温−海面水温」と「相対湿度」の分布
    fog_probability.png             相対湿度 × (気温−海面水温) ごとの霧の発生率
    fog_probability_by_station.png  官署別の同じ図
    fog_*.csv                       上の元になった集計表
"""
import argparse

import sst_tool as t

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--dir", default=t.HERE, help="海面水温 *.txt のあるフォルダ")
p.add_argument("--coast-dir", default=t.COAST_DIR, help="沿岸官署の時別値CSVのフォルダ")
p.add_argument("--from-year", type=int, help="この年以降だけ使う")
a = p.parse_args()
t.cmd_fog(t.summer(t.load(a.dir)), a)
