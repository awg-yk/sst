"""相対湿度の範囲ごと (縦) × 年 (横) の、視程<1kmの出現率を示す図 (官署ごとに1枚)。マス内の数字は 上=視程<1kmだった時間 / 下=該当した時間。
霧のときの湿度が、年によってどの範囲に出ているかを見る。80%未満は除き、80〜100%未満を2%刻み、最上段は100%ちょうど。
fog_plot.py と同じフォルダに置いて実行します。

    python fog_plot_rh_year.py                    # 官署ごとの図を画面に順に表示
    python fog_plot_rh_year.py --stations 宮古 八戸
    python fog_plot_rh_year.py --from-year 1993   # 開始年 (既定1990。気温・湿度は1990年から毎時)
    python fog_plot_rh_year.py --save-dir out     # 画面に出さずPNGを保存

色: 視程<1kmの出現率(%) = 視程<1kmだった時間 ÷ 該当した時間 (全官署で同じ基準)。灰色: 視程<1kmが0回、または該当した時間が40時間未満。空白: 該当なし。
視程が毎時になる前の年は、視程が観測された時刻だけを使うので、該当した時間が少なく、灰色が多くなります。
相対湿度100%の記録は2013年ごろから増えています (観測方法の変更の公式な記録は確認できていない)。
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap

import fog_plot as fp

LOW = 80   # これ未満の湿度は除く


def station_tables(coast, station, years):
    """官署ひとつの、湿度の範囲 (行) × 年 (列) の (該当した時間, うち視程<1kmだった時間)。夏季 (6〜8月)、視程と湿度がそろった時刻。"""
    d = coast[(coast["station"] == station) & coast["RH"].notna() & coast["vis"].notna()]
    day = (d["time"] - pd.Timedelta(hours=1)).dt.normalize()
    d = d[day.dt.month.isin([6, 7, 8])]
    year = (d["time"] - pd.Timedelta(hours=1)).dt.year
    rows = [(a, a + 2) for a in range(LOW, 100, 2)] + [(100, None)]
    n = np.zeros((len(rows), len(years)))
    k = np.zeros_like(n)
    for j, y in enumerate(years):
        e = d[year == y]
        for i, (a, b) in enumerate(rows):
            m = (e["RH"] >= 100) if b is None else ((e["RH"] >= a) & (e["RH"] < b))
            n[i, j] = m.sum()
            k[i, j] = (m & (e["vis"] < fp.FOG_KM)).sum()
    return rows, pd.DataFrame(n), pd.DataFrame(k)


def draw(station, rows, n, k, years, norm, cmap):
    fig, ax = plt.subplots(figsize=(19, 6.6))
    ext = (years[0], years[-1] + 1, 0, len(rows))
    rate = fp.fog_rate(n, k).values
    ax.imshow(np.where(n.values > 0, 1.0, np.nan), origin="lower", extent=ext, aspect="auto", cmap=ListedColormap(["#e4e4e4"]), vmin=0, vmax=1)
    ax.imshow(rate, origin="lower", extent=ext, aspect="auto", cmap=cmap, norm=norm)
    for i in range(len(rows)):
        for j in range(len(years)):
            if n.values[i, j] > 0:
                colored = np.isfinite(rate[i, j])
                col = "w" if colored and norm(rate[i, j]) > .55 else ("k" if colored else "#888")
                ax.text(years[0] + j + .5, i + .5, f"{int(k.values[i, j])}\n{int(n.values[i, j])}", ha="center", va="center", fontsize=5.3, linespacing=1.0, color=col)
    ax.set_yticks(np.arange(len(rows)) + .5)
    ax.set_yticklabels([f"{a}〜{b}" for a, b in rows[:-1]] + ["100 (ちょうど)"], fontsize=8)
    ax.set_xticks(np.arange(years[0], years[-1] + 1, 5) + .5)
    ax.set_xticklabels(np.arange(years[0], years[-1] + 1, 5))
    ax.axhline(len(rows) - 1, color="k", lw=.8)
    ax.set(xlabel="年 (夏季6〜8月)", ylabel="相対湿度 (%)",
           title=f"{station}: 相対湿度の範囲ごとの、視程<1kmの出現率 (色) と 回数 (上=視程<1kmだった時間 / 下=該当した時間)")
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=.85).set_label("視程<1kmの出現率 (%)")
    fig.text(.01, .01, f"{LOW}%未満は除外。灰色=視程<1kmが0回、または該当した時間が{fp.MIN_N}時間未満。空白=該当なし。視程の観測時刻が間引かれた年 (自動化前) は該当した時間が少なく、灰色が増える。"
             "\n相対湿度100%の記録は2013年ごろから増えている (観測方法の変更の公式な記録は確認できていない)。", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coast-dir", default=fp.COAST_DIR, help="沿岸官署の時別値CSVのフォルダ")
    p.add_argument("--stations", nargs="*", help="官署名 (省略で沿岸10官署)")
    p.add_argument("--from-year", type=int, default=1990, help="開始年 (既定1990)")
    p.add_argument("--save-dir", help="画面に出さずPNGをこのフォルダに保存する")
    a = p.parse_args()

    fp.setup_font()
    coast = fp.load_coast(a.coast_dir)
    if coast is None:
        raise SystemExit(f"沿岸官署のCSVが見つかりません: {a.coast_dir} (--coast-dir で指定)")
    years = list(range(a.from_year, int(coast["time"].dt.year.max()) + 1))
    allst = [s for s in fp.STATION_SST if s in set(coast["station"])]
    tabs = {s: station_tables(coast, s, years) for s in allst}
    norm = fp.fog_norm([(n, k) for _, n, k in tabs.values()])   # 全官署で共通の色の基準
    cmap = plt.get_cmap("magma_r")
    for s in [s for s in (a.stations or allst) if s in tabs]:
        rows, n, k = tabs[s]
        fig = draw(s, rows, n, k, years, norm, cmap)
        if a.save_dir:
            os.makedirs(a.save_dir, exist_ok=True)
            path = os.path.join(a.save_dir, f"fog_rh_year_{s}.png")
            fig.savefig(path, dpi=130)
            print("保存:", path)
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    main()
