"""気温(横軸) × 海面水温(縦軸) の視程<1kmの図を、年代ごとに並べて示す図 (官署ごとに1枚)。
視程<1kmの出現率 (fog_plot_temp_sst.py --each と同じ図) を年代ごとに並べる。マス内の数字は 上=視程<1kmだった時間 / 下=該当した時間。
色の濃い部分が年代とともに減るか、右上(気温・海面水温が高いほう)へ移るかを見る。
fog_plot.py と fog_plot_temp_sst.py と同じフォルダに置いて実行します。

    python fog_plot_ta_sst_period.py                         # 官署ごとに1枚 (年代を横に並べる) を順に画面に表示
    python fog_plot_ta_sst_period.py --stations 宮古 八戸     # 官署を指定
    python fog_plot_ta_sst_period.py --periods 1989 2000 2010 2020   # 年代の区切り (この年から次の区切りの前年まで。既定は 1989 2000 2010 2020)
    python fog_plot_ta_sst_period.py --save-dir out          # 画面に出さずPNGを保存

色: 視程<1kmの出現率(%) = 視程<1kmだった時間 ÷ 該当した時間 (全官署・全年代で同じ色の基準)。 灰色: 視程<1kmが0回、または該当した時間が40時間未満。 空白: 該当なし。
破線: 気温 = 海面水温。 目盛りはマスの境目 (1℃ごと)。
図の範囲は、その官署で視程<1kmがあった温度の範囲に絞っています。
視程の記録は1989年からです。2019年以前は視程が間引かれている官署が多く、官署ごとに視程が毎時になった年が違います。
その年を各図の下に書きます (年代の題にも、毎時の期間が全部か一部かを書きます)。
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

import fog_plot as fp
import fog_plot_temp_sst as ts


def period_label(bounds, i, last_year):
    end = bounds[i + 1] - 1 if i + 1 < len(bounds) else last_year
    return f"{bounds[i]}〜{end}年", bounds[i], end


def crop(k, margin=1):
    """視程<1kmが1回でもあった温度の範囲 (気温・海面水温の両方) に、前後 margin ℃を足した範囲 (lo, hi) を返す。k は全年代の合計。"""
    v = k.values
    ys, xs = np.where(v > 0)
    idx = k.index
    lo = min(idx[ys.min()], idx[xs.min()]) - margin
    hi = max(idx[ys.max()], idx[xs.max()]) + margin
    return max(lo, idx.min()), min(hi, idx.max())


def hourly_note(start, y0, y1):
    """視程が毎時になった年 start と、年代 y0〜y1 の関係を短い文字にする。"""
    if start is None or start > y1:
        return "毎時の期間なし"
    return "全期間が毎時" if start <= y0 else f"{start}年から毎時"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=fp.SST_DIR, help="海面水温 *.txt のあるフォルダ")
    p.add_argument("--coast-dir", default=fp.COAST_DIR, help="沿岸官署の時別値CSVのフォルダ")
    p.add_argument("--periods", type=int, nargs="+", default=[1989, 2000, 2010, 2020], help="年代の区切りとなる開始年 (小さい順)")
    p.add_argument("--stations", nargs="*", help="官署名 (省略で沿岸10官署)")
    p.add_argument("--save-dir", help="画面に出さずPNGをこのフォルダに保存する")
    a = p.parse_args()

    fp.setup_font()
    coast = fp.load_coast(a.coast_dir)
    if coast is None:
        raise SystemExit(f"沿岸官署のCSVが見つかりません: {a.coast_dir} (--coast-dir で指定)")
    start = fp.hourly_start_years(coast)   # 官署ごとの、視程が毎時になった年
    d = fp.fog_dataset(fp.summer(fp.load(a.dir)), coast)
    d = d[d["vis"].notna() & d["Ta"].notna() & d["SST"].notna()]
    bounds = sorted(a.periods)
    year = d["time"].dt.year
    last_year = int(year.max())
    d = d[year >= bounds[0]]
    pid = np.searchsorted(bounds, d["time"].dt.year.values, side="right") - 1
    allst = [s for s in fp.STATION_SST if s in set(d["station"])]
    lo, hi = ts.temp_range(d)
    tabs = {(s, i): ts.ta_sst_table(d[(d["station"] == s) & (pid == i)], lo, hi) for s in allst for i in range(len(bounds))}
    norm = fp.fog_norm(list(tabs.values()))                                                  # 出現率の色の基準 (全官署・全年代で共通)
    cmap = plt.get_cmap("magma_r")
    names = [s for s in (a.stations or allst) if s in allst]

    for s in names:
        total = sum(tabs[(s, i)][1] for i in range(len(bounds)))
        clo, chi = crop(total)
        fig, axs = plt.subplots(1, len(bounds), figsize=(6.4 * len(bounds) + 1.2, 7), squeeze=False)
        for i in range(len(bounds)):
            n, k = (t.loc[clo:chi, clo:chi] for t in tabs[(s, i)])
            label, y0, y1 = period_label(bounds, i, last_year)
            head = f"{s} {label} ({hourly_note(start.get(s), y0, y1)})\n該当{int(n.values.sum())}時間, 視程<1km {int(k.values.sum())}時間"
            sm = ts.heat(axs[0, i], n, k, head, norm, cmap, numbers=True)
            axs[0, i].title.set_fontsize(10)
        fig.tight_layout(rect=(0, .07, .94, 1))
        fig.text(.01, .004,
                 "マス内の数字: 上=視程<1kmだった時間 / 下=該当した時間。色=視程<1kmの出現率 (上÷下)。灰色=視程<1kmが0回、または該当した時間が40時間未満。空白=該当なし。"
                 f"\n破線=気温と海面水温が等しい。目盛りはマスの境目 (各マスは例えば16.0℃以上17.0℃未満)。{s}は視程が毎時になった年={start.get(s) or '不明'} (観測の自動化 {fp.AUTOMATION.get(s, '不明')}、気象観測統計指針 表3.4－6。それ以前は目視で間引かれた時刻だけの観測)。",
                 fontsize=8)
        bb = axs[0, -1].get_position()
        fig.colorbar(sm, cax=fig.add_axes([.955, bb.y0 + .1 * bb.height, .012, .8 * bb.height])).set_label("視程<1kmの出現率 (%)", fontsize=9)
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
