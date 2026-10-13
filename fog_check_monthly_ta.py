"""気象庁の月別値の平均気温 (東北地方沿岸気象官署月別値平均気温.csv) が、時別値の気温から求めた月平均と合っているかを確かめる。
fog_plot.py と同じフォルダに置いて実行します (fog_plot.py の読み込み処理を使います)。

    python fog_check_monthly_ta.py                         # 表を表示し、差の図を画面に表示
    python fog_check_monthly_ta.py --save out/check.png    # 図をPNGに保存
    python fog_check_monthly_ta.py --monthly my.csv        # 月別値のCSVを指定

比べるもの: 月別値の平均気温 (品質情報が8の月だけ) と、時別値 (6〜8月、品質情報が8の気温) の月平均。
差 = 時別値の月平均 − 月別値。 時別値は 1982年〜1989年が3時間おき、1990年からが毎時です。
月別値は日平均気温 (1日の観測値の平均) の月平均なので、1日の観測の数が少ない年は差が出ます。
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fog_plot as fp

MONTHLY = os.path.join(fp.HERE, "東北地方沿岸気象官署月別値平均気温.csv")


def load_monthly(path):
    """月別値CSV -> 縦長の表 (station, yr, mon, official)。品質情報が8以外の月は除く。"""
    rows = fp._read_rows(path)
    names = rows[2]
    out = []
    for st in dict.fromkeys(n for n in names[1:] if n):
        c = next(i for i in range(1, len(names)) if names[i] == st)
        for r in rows[5:]:
            if not r or "/" not in r[0] or len(r) <= c + 1:
                continue
            y, m = map(int, r[0].split("/"))
            if r[c + 1] == "8" and r[c] != "":
                out.append((st, y, m, float(r[c])))
    return pd.DataFrame(out, columns=["station", "yr", "mon", "official"])


def hourly_monthly(coast):
    """時別値の気温から、官署ごと・月ごとの平均と、使えた観測の数を求める (0時は前日24時として月に数える)。"""
    c = coast[coast["station"].isin(fp.STATION_SST)].dropna(subset=["Ta"]).copy()
    day = (c["time"] - pd.Timedelta(hours=1)).dt.normalize()
    c = c[day.dt.month.isin([6, 7, 8])]
    day = day[c.index]
    g = c.groupby([c["station"], day.dt.year.rename("yr"), day.dt.month.rename("mon")])["Ta"]
    return g.agg(hourly="mean", n="count").reset_index()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coast-dir", default=fp.COAST_DIR, help="沿岸官署の時別値CSVのフォルダ")
    p.add_argument("--monthly", default=MONTHLY, help="月別値の平均気温のCSV")
    p.add_argument("--save", help="図のPNGの保存先 (省略で画面表示)")
    a = p.parse_args()

    fp.setup_font()
    coast = fp.load_coast(a.coast_dir)
    if coast is None:
        raise SystemExit("沿岸官署のCSVが見つかりません: " + a.coast_dir)
    m = hourly_monthly(coast).merge(load_monthly(a.monthly), on=["station", "yr", "mon"], how="inner")
    m["diff"] = m["hourly"] - m["official"]
    m["period"] = np.where(m["yr"] < 1990, "1982〜1989年 (3時間おき)", "1990年〜 (毎時)")
    stations = [s for s in fp.STATION_SST if s in set(m["station"])]

    print("差 = 時別値の月平均 − 月別値 (℃)。 |差| が0.1℃を超えた月の割合も表示")
    summary = (m.groupby(["period"])["diff"].agg(月数="count", 平均="mean", 標準偏差="std", 最大絶対値=lambda x: x.abs().max())
               .assign(割合_差0_1超=m.groupby("period")["diff"].apply(lambda x: (x.abs() > 0.1).mean())).round(3))
    print(summary.to_string())
    print("\n官署別 (1990年〜):")
    late = m[m["yr"] >= 1990]
    by = late.groupby("station")["diff"].agg(月数="count", 平均="mean", 標準偏差="std", 最大絶対値=lambda x: x.abs().max()).reindex(stations).round(3)
    print(by.to_string())
    bad = late[late["diff"].abs() > 0.1].sort_values("diff", key=abs, ascending=False).head(15)
    if len(bad):
        print("\n1990年以降で差が大きい月 (上位15):")
        print(bad[["station", "yr", "mon", "hourly", "official", "diff", "n"]].round(2).to_string(index=False))

    ncol = (len(stations) + 1) // 2
    fig, axs = plt.subplots(2, ncol, figsize=(3.6 * ncol + .5, 7.5), sharex=True, sharey=True, squeeze=False)
    colors = {6: "#4477aa", 7: "#ee6677", 8: "#228833"}
    for ax, st in zip(axs.ravel(), stations):
        d = m[m["station"] == st]
        ax.axhline(0, color="k", lw=.8)
        ax.axvspan(1981.5, 1989.5, color="#eeeeee", zorder=0)
        for mon, color in colors.items():
            x = d[d["mon"] == mon]
            ax.plot(x["yr"], x["diff"], "o", ms=3, color=color, label=f"{mon}月")
        ax.set_title(st, fontsize=11)
        ax.grid(alpha=.3)
    for ax in axs.ravel()[len(stations):]:
        ax.axis("off")
    axs[0, 0].legend(fontsize=8, loc="upper left")
    for ax in axs[:, 0]:
        ax.set_ylabel("時別値の月平均 − 月別値 (℃)")
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.text(.01, .004, "点=1か月ごとの差。0なら一致。灰色の帯=1982〜1989年 (時別値が3時間おきなので、日平均と少し違う値になりうる)。", fontsize=8)
    if a.save:
        os.makedirs(os.path.dirname(os.path.abspath(a.save)), exist_ok=True)
        fig.savefig(a.save, dpi=130)
        print("保存:", a.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
