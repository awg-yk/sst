"""霧(視程<1km)の発生率を官署別に示す図を、ファイルに保存せず画面に表示する。
sst_tool.py と同じフォルダに置いて実行します。

    python fog_plot.py                  # 全期間
    python fog_plot.py --from-year 2020 # 視程が毎時になった2020年以降だけ

図: 官署ごとの、相対湿度 × (気温−海面水温) の霧の発生率(%)。観測数15未満の階級は空欄。
"""
import argparse

import matplotlib.pyplot as plt

import sst_tool as t

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--dir", default=t.HERE, help="海面水温 *.txt のあるフォルダ")
p.add_argument("--coast-dir", default=t.COAST_DIR, help="沿岸官署の時別値CSVのフォルダ")
p.add_argument("--from-year", type=int, help="この年以降だけ使う")
a = p.parse_args()
if not hasattr(t, "fog_valid"):
    raise SystemExit("sst_tool.py が古いです。GitHub の最新の sst_tool.py に置き換えてください。")
t.setup_font()
_, v = t.fog_valid(t.summer(t.load(a.dir)), a)
t.fog_station_figure(v)
plt.show()
