"""夏季 (6〜8月) の気温 (横軸) と 公式の霧日数 (縦軸) の散布図を、官署ごとに示す図。1点=1年。
気温は「東北地方3か月平均.csv」(気象庁 月別値の3か月平均: 平均気温、日最高気温の平均、日最低気温の平均) を使う。
fog_plot.py と fog_days.py と同じフォルダに置いて実行します。

    python fog_plot_ta_fog.py                       # 3種類 (平均気温、日最高気温、日最低気温) の図を順に画面に表示
    python fog_plot_ta_fog.py --what mean           # mean=平均気温 / max=日最高気温の平均 / min=日最低気温の平均
    python fog_plot_ta_fog.py --highlight 2023      # 黒で印をつける年 (省略すると、どの年にも印をつけない)
    python fog_plot_ta_fog.py --periods 1931 1960 1990 2010   # 年代ごとに色を分けて、年代ごとに回帰直線を引く (この年から次の区切りの前年まで)
    python fog_plot_ta_fog.py --save-dir out        # 画面に出さずPNGを保存

--periods を付けないとき、点: 濃い赤=最新の均質区間 (霧日数の観測環境が変わった最後の年以降)、薄い赤=それより前の年。黒=--highlight で指定した年。直線は全部の年への直線の当てはめ。
題の r は 全部の年 / 最新の均質区間だけ の相関係数。
--periods を付けたとき、点と直線は年代ごとに色分けし (直線は各年代の直線の当てはめ)、凡例に年代・年数・傾き (日/℃)・相関係数 r を書く。霧日数は公式値で、観測環境が変わる年 (観測の自動化など) の前後で水準が変わります。
"""
import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

import fog_days as fd
import fog_plot as fp
from fog_plot_year import panels

TEMP_CSV = os.path.join(fp.HERE, "東北地方3か月平均.csv")
PERIOD_COLORS = ["#7f7f7f", "#0077bb", "#ee7733", "#cc3311", "#009988", "#aa3377"]
WHAT = {"mean": ("平均気温", "夏季の平均気温"), "max": ("日最高気温の平均", "夏季の日最高気温の平均"), "min": ("日最低気温の平均", "夏季の日最低気温の平均")}


def load_temps(path):
    """3か月平均のCSV (cp932/utf-8) -> {(官署, 項目名): Series(年 -> 値)}。品質情報が8(正常)の値だけ。"""
    for enc in ("utf-8-sig", "cp932"):
        try:
            rows = list(csv.reader(open(path, encoding=enc, newline="")))
            break
        except UnicodeDecodeError:
            continue
    h = next(i for i, r in enumerate(rows) if r and r[0] == "集計開始")
    names, titles = rows[h], rows[h + 1]
    out = {}
    for i in range(2, len(names), 3):   # 値・品質情報・均質番号の3列ずつ
        vals = {}
        for r in rows[h + 3:]:
            if r and r[0] and len(r) > i + 1 and r[i] != "" and r[i + 1] == "8":
                vals[int(r[0].split("/")[0])] = float(r[i])
        out[(names[i], titles[i].split("(")[0])] = vals
    return out


def poster_figure(data, names, bounds, label, a):
    """ポスター用の図: 官署を2×3に並べ、年代ごとに色分けして回帰直線を引く。凡例は空いたパネルに1つだけ置く。
    貼り付けるサイズ (--size-cm) で描くので、文字の大きさ (pt) は、そのまま印刷の大きさになる。"""
    fs = a.fontsize
    plt.rcParams.update({"font.size": fs, "axes.labelsize": fs, "xtick.labelsize": fs, "ytick.labelsize": fs, "axes.titlesize": fs * 1.2})
    ncol = 3
    nrow = (len(names) + ncol - 1) // ncol
    fig, axs = plt.subplots(nrow, ncol, figsize=(a.size_cm[0] / 2.54, a.size_cm[1] / 2.54), sharey=True, squeeze=False)
    last_year = max(int(data[s][0].max()) for s in names)
    handles = {}
    for ax, s in zip(axs.ravel(), names):
        yr, x, y, h = data[s]
        pid = np.searchsorted(bounds, yr, side="right") - 1
        for i, b0 in enumerate(bounds):
            mk = pid == i
            if mk.sum() < 3:
                continue
            end = bounds[i + 1] - 1 if i + 1 < len(bounds) else last_year
            col = PERIOD_COLORS[i % len(PERIOD_COLORS)]
            slope, icpt = np.polyfit(x[mk], y[mk], 1)
            ax.scatter(x[mk], y[mk], s=fs * 2.2, color=col, zorder=2, alpha=.9)
            xs = np.array([x[mk].min(), x[mk].max()])
            ax.plot(xs, slope * xs + icpt, color=col, lw=fs / 7, zorder=3)
            handles[i] = (col, f"{b0}〜{end}年")
        ax.set_title(s, pad=4)
        ax.grid(alpha=.3)
        ax.locator_params(axis="x", nbins=4)
        ax.locator_params(axis="y", nbins=4)
    spare = list(axs.ravel()[len(names):])
    for ax in spare:
        ax.axis("off")
    from matplotlib.lines import Line2D
    items = [Line2D([0], [0], marker="o", color=c, lw=fs / 7, markersize=fs / 2.2, label=t) for _, (c, t) in sorted(handles.items())]
    leg_ax = spare[0] if spare else axs[0, -1]
    leg_ax.legend(handles=items, loc="center", fontsize=fs * .95, frameon=False, title="年代 (直線は年代ごとの回帰直線)", title_fontsize=fs * .8)
    for r in range(nrow):
        for c in range(ncol):
            ax = axs[r, c]
            below_empty = r + 1 >= nrow or not axs[r + 1, c].axison
            if ax.axison and below_empty:       # 下にパネルが無い列にも、横軸の名前を付ける
                ax.set_xlabel(f"{label} (℃)")
    for r in range(nrow):
        axs[r, 0].set_ylabel("霧日数 (日)")
    fig.tight_layout(pad=.4, w_pad=.6, h_pad=.8)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--temps", default=TEMP_CSV, help="3か月平均気温のCSV")
    p.add_argument("--kiri", default=os.path.join(fp.HERE, "東北地方気象官署月別値霧日数.csv"), help="霧日数の月別値CSV")
    p.add_argument("--what", nargs="*", choices=list(WHAT), default=list(WHAT), help="横軸の気温 (省略で3種類すべて)")
    p.add_argument("--periods", type=int, nargs="+", help="年代の区切りとなる開始年 (小さい順)。指定すると、年代ごとに色分けして回帰直線を引く")
    p.add_argument("--stations", nargs="*", help="官署名 (省略で、データのある官署すべて)")
    p.add_argument("--poster", action="store_true", help="ポスター用: 官署を2×3に並べ、文字を大きくし、凡例を空いたパネルに1つだけ置く (--periods と一緒に使う)")
    p.add_argument("--size-cm", type=float, nargs=2, default=[40, 22], metavar=("幅", "高さ"), help="--poster のときの、貼り付ける大きさ (cm)。既定 40 22")
    p.add_argument("--fontsize", type=float, default=24, help="--poster のときの、目盛りの文字の大きさ (pt)。既定24")
    p.add_argument("--highlight", type=int, help="黒で印をつける年 (省略すると印をつけない)")
    p.add_argument("--save-dir", help="画面に出さずPNGをこのフォルダに保存する")
    a = p.parse_args()

    fp.setup_font()
    kiri = fd.load_kiri(a.kiri)
    temps = load_temps(a.temps)
    figs = []
    for key in a.what:
        item, label = WHAT[key]
        data = {}
        for s in fp.STATION_SST:
            t = temps.get((s, item))
            k = kiri[kiri["station"] == s].dropna(subset=["kiri"]).set_index("yr")
            if not t:
                continue
            yrs = sorted(set(t) & set(k.index))
            if len(yrs) >= 10:
                data[s] = (np.array(yrs), np.array([t[y] for y in yrs]), k.loc[yrs, "kiri"].values, k.loc[yrs, "homog"].values)
        names = [n for n in (a.stations or fp.STATION_SST) if n in data]
        if a.poster:
            if not a.periods:
                raise SystemExit("--poster は --periods と一緒に使います (例: --periods 1931 1960 1990 2010)")
            figs.append((poster_figure(data, names, sorted(a.periods), label, a), f"fog_ta_fog_{key}_poster.png"))
            continue
        fig, axs = panels(names)
        for ax in axs.ravel():
            ax.sharex(axs.ravel()[0])
        print(f"{item}: 官署別 相関係数 r (全部の年 / 最新の均質区間)")
        for ax, s in zip(axs.ravel(), names):
            yr, x, y, h = data[s]
            last = h == h.max()
            if a.periods:
                bounds = sorted(a.periods)
                pid = np.searchsorted(bounds, yr, side="right") - 1
                for i, b0 in enumerate(bounds):
                    mk = pid == i
                    if mk.sum() < 3:
                        continue
                    end = bounds[i + 1] - 1 if i + 1 < len(bounds) else int(yr.max())
                    col = PERIOD_COLORS[i % len(PERIOD_COLORS)]
                    slope, icpt = np.polyfit(x[mk], y[mk], 1)
                    r_p = np.corrcoef(x[mk], y[mk])[0, 1]
                    ax.scatter(x[mk], y[mk], s=16, color=col, zorder=2)
                    xs = np.array([x[mk].min(), x[mk].max()])
                    ax.plot(xs, slope * xs + icpt, color=col, lw=1.8, zorder=3,
                            label=f"{b0}〜{end}年 ({mk.sum()}年) {slope:+.1f}日/℃ r={r_p:+.2f}")
                    print(f"  {s} {b0}〜{end}年 ({mk.sum()}年): 傾き {slope:+.2f} 日/℃, r={r_p:+.2f}")
                ax.legend(fontsize=6, loc="upper right")
                ax.set_title(f"{s}", fontsize=10)
            else:
                r_all = np.corrcoef(x, y)[0, 1]
                r_last = np.corrcoef(x[last], y[last])[0, 1] if last.sum() >= 5 else np.nan
                ax.scatter(x[~last], y[~last], s=18, color="#e8b4ac", zorder=2)
                ax.scatter(x[last], y[last], s=18, color="#cc3311", zorder=3)
                slope, icpt = np.polyfit(x, y, 1)
                xs = np.array([x.min(), x.max()])
                ax.plot(xs, slope * xs + icpt, color="gray", lw=1.2, zorder=1)
                print(f"  {s}: {r_all:+.2f} / {r_last:+.2f}  ({len(x)}年, 最新区間 {last.sum()}年)")
                ax.set_title(f"{s}\nr={r_all:+.2f} / 最新区間 r={r_last:+.2f}  ({len(x)}年)", fontsize=9)
            if a.highlight and a.highlight in yr:
                i = list(yr).index(a.highlight)
                ax.scatter([x[i]], [y[i]], s=40, color="k", zorder=4)
                ax.annotate(str(a.highlight), (x[i], y[i]), textcoords="offset points", xytext=(4, 4), fontsize=8)
            ax.grid(alpha=.3)
        for ax in axs[-1, :]:
            ax.set_xlabel(f"{label} (℃)")
        for ax in axs[:, 0]:
            ax.set_ylabel("霧日数 (6〜8月の合計, 日)")
        fig.tight_layout(rect=(0, .05, 1, 1))
        hl = f"、黒={a.highlight}年" if a.highlight else ""
        note = (f"1点=1年。色=年代、直線=その年代の直線の当てはめ (凡例に年数・傾き・r){hl}。\n" if a.periods else
                f"1点=1年。濃い赤=最新の均質区間、薄い赤=それより前 (霧日数の観測環境が変わる前){hl}。直線=全部の年の当てはめ。題の r=全部の年 / 最新の均質区間。\n")
        fig.text(.01, .004, note +
                 "霧日数は気象庁の公式値 (観測の自動化の前後で水準が変わる)。気温は気象庁 月別値の3か月平均。相関は因果を示さない。", fontsize=8)
        figs.append((fig, f"fog_ta_fog_{key}.png"))
    for fig, name in figs:
        if a.save_dir:
            os.makedirs(a.save_dir, exist_ok=True)
            path = os.path.join(a.save_dir, name)
            fig.savefig(path, dpi=130)
            print("保存:", path)
            plt.close(fig)
    if not a.save_dir:
        plt.show()


if __name__ == "__main__":
    main()
