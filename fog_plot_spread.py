"""夏季 (6〜8月) の「気温 − 露点温度」の年平均を、官署ごとに折れ線で示す図。
相対湿度の記録のしかたが変わった年 (相対湿度100%が記録されるようになった年) を境に、トレンドラインを分けて引きます。
fog_plot.py と同じフォルダに置いて実行します。

    python fog_plot_spread.py                       # 官署ごとの図を画面に表示
    python fog_plot_spread.py --breaks 宮古=2013 八戸=2012   # 境の年を指定 (省略した官署は自動)
    python fog_plot_spread.py --no-break            # 境を作らず、全期間に1本だけ引く
    python fog_plot_spread.py --save-dir out        # 画面に出さずPNGを保存

気温 − 露点温度が小さいほど湿度が高い (1℃で相対湿度およそ6%)。
境の年の自動判定: 夏の相対湿度100%の時間が、前の3年の平均は1%未満、その年から3年の平均は 2% 以上 (3年とも1%以上) になる、2008年以降でいちばん新しい年 (境のあとが6年以上残る年まで)。判定できない官署 (酒田など) は境なし。
夏の相対湿度100%の時間は、1998年ごろと2006年ごろにも増減していて (官署による)、記録のしかたが何度か変わった可能性があります。
時別値CSVの均質番号は、気温・露点温度・相対湿度のどれもすべて1で、記録のしかたの変更は示されていません。
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fog_plot as fp
from fog_plot_year import panels

FROM_YEAR = 1984   # 露点温度は1984年から
RH100_PCT = 2.0    # 境の年の判定: 夏の相対湿度100%の時間割合(%)


def yearly(coast, station):
    """官署ひとつの、夏季の年ごとの (気温−露点温度の平均, 相対湿度100%の時間割合%)。気温と露点温度がそろった時刻だけ。"""
    d = coast[(coast["station"] == station) & coast["Ta"].notna() & coast["Td"].notna()]
    day = (d["time"] - pd.Timedelta(hours=1)).dt.normalize()
    ok = day.dt.month.isin([6, 7, 8]) & (day.dt.year >= FROM_YEAR)
    d, year = d[ok], day[ok].dt.year
    spread = (d["Ta"] - d["Td"]).groupby(year).mean()
    rh100 = (d["RH"] >= 100).groupby(year).mean() * 100
    return spread, rh100


def detect_break(rh100):
    """相対湿度100%の時間が増えた境の年を返す。2008年以降で、その年から3年の平均が RH100_PCT 以上 (3年とも1%以上) で、その前の3年の平均が1%未満になる、いちばん新しい年 (境のあとが6年以上残る年まで)。無ければ None。"""
    for y in range(int(rh100.index.max()) - 5, FROM_YEAR + 2, -1):
        before, after = rh100.reindex([y - 3, y - 2, y - 1]).mean(), rh100.reindex([y, y + 1, y + 2]).mean()
        if y >= 2008 and before < 1 and after >= RH100_PCT and rh100.reindex([y, y + 1, y + 2]).min() >= 1:
            return y
    return None


def trend(ax, s, color, label):
    if len(s) < 3:
        return None
    slope, icpt = np.polyfit(s.index, s.values, 1)
    ax.plot(s.index, slope * s.index + icpt, color=color, lw=2, alpha=.8, label=f"{label} {slope * 10:+.2f}℃/10年")
    return slope


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coast-dir", default=fp.COAST_DIR, help="沿岸官署の時別値CSVのフォルダ")
    p.add_argument("--breaks", nargs="*", default=[], help="境の年を官署ごとに指定 (例: 宮古=2013)。指定した官署は自動判定しない")
    p.add_argument("--no-break", action="store_true", help="境を作らず、全期間に1本だけトレンドラインを引く")
    p.add_argument("--save-dir", help="画面に出さずPNGをこのフォルダに保存する")
    a = p.parse_args()

    fp.setup_font()
    coast = fp.load_coast(a.coast_dir)
    if coast is None:
        raise SystemExit(f"沿岸官署のCSVが見つかりません: {a.coast_dir} (--coast-dir で指定)")
    given = {kv.split("=")[0]: int(kv.split("=")[1]) for kv in a.breaks}
    names = [s for s in fp.STATION_SST if s in set(coast["station"])]
    data = {s: yearly(coast, s) for s in names}

    fig, axs = panels(names)
    print("官署別: 境の年 / 傾き (℃/10年) 全期間, 境の前, 境の後")
    for ax, s in zip(axs.ravel(), names):
        sp, rh100 = data[s]
        sp = sp.dropna()
        brk = None if a.no_break else given.get(s, detect_break(rh100))
        ax.plot(sp.index, sp.values, "o-", ms=3, lw=.8, color="#4477aa")
        whole = trend(ax, sp, "#999999", "全期間")
        parts = ""
        if brk is not None:
            ax.axvline(brk - .5, color="k", ls="--", lw=.8)
            sa = trend(ax, sp[sp.index < brk], "#228833", f"〜{brk - 1}年")
            sb = trend(ax, sp[sp.index >= brk], "#cc3311", f"{brk}年〜")
            parts = f", {sa * 10:+.2f}, {sb * 10:+.2f}" if sa is not None and sb is not None else ""
        print(f"  {s}: {brk or '境なし'} / {whole * 10:+.2f}{parts}")
        ax.set(title=f"{s} (海域: {fp.STATION_SST[s]})")
        ax.legend(fontsize=6.5, loc="upper right")
        ax.grid(alpha=.3)
    for ax in axs[:, 0]:
        ax.set_ylabel("気温 − 露点温度 の夏季平均 (℃)")
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.text(.01, .004, "気温 − 露点温度が小さいほど湿度が高い。破線=相対湿度の記録のしかたが変わった年 (夏の相対湿度100%の時間が増えた年。官署ごとに自動判定)、直線=その前後と全期間の直線の当てはめ。"
             "\n時別値の均質番号はすべて1で、変更は示されていない。1989年以前の観測は3時間おき。", fontsize=8)
    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)
        path = os.path.join(a.save_dir, "fog_year_spread.png")
        fig.savefig(path, dpi=130)
        print("保存:", path)
    else:
        plt.show()


if __name__ == "__main__":
    main()
