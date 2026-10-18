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
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize

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


def heat_share(ax, n, k, total_n, title, norm, cmap, numbers=False):
    """色=そのマスの視程<1km時間 ÷ その年代の全観測時間 (%)。全マスを足すと、その年代の視程<1kmの出現率になる。
    年代ごとに観測時間の数が違っても (年代の長さ・毎時でない時期の間引き)、同じ尺度で比べられる。灰色=該当なし (観測がないマス)。"""
    lo, hi = n.index.min(), n.index.max()
    nv = n.values
    share = np.where(nv > 0, k.values / total_n * 100, np.nan)
    ext = (lo, hi + 1, lo, hi + 1)
    ax.imshow(np.where(nv > 0, 1.0, np.nan), origin="lower", extent=ext, aspect="equal", cmap=ListedColormap(["#e4e4e4"]), vmin=0, vmax=1)
    ax.imshow(np.where(share > 0, share, np.nan), origin="lower", extent=ext, aspect="equal", cmap=cmap, norm=norm)
    ax.plot([lo, hi + 1], [lo, hi + 1], "k--", lw=.8, alpha=.6)
    if numbers:                                  # マス内の数字: 上=視程<1kmだった時間 / 下=該当した時間
        for i in range(nv.shape[0]):
            for j in range(nv.shape[1]):
                if nv[i, j] > 0:
                    dark = np.isfinite(share[i, j]) and share[i, j] > 0 and norm(share[i, j]) > 0.55
                    ax.text(lo + j + .5, lo + i + .5, f"{int(k.values[i, j])}\n{int(nv[i, j])}", ha="center", va="center",
                            fontsize=5.5, linespacing=1.0, color="w" if dark else ("k" if share[i, j] > 0 else "#888"))
        ax.set_xticks(range(lo, hi + 2))
        ax.set_yticks(range(lo, hi + 2))
        ax.tick_params(labelsize=7)
        ax.set(xlabel="気温 (℃)", ylabel="海面水温 (℃)")
    ax.set_xlim(lo, hi + 1)
    ax.set_ylim(lo, hi + 1)
    ax.set_title(title)
    return ScalarMappable(norm=norm, cmap=cmap)


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
    p.add_argument("--poster", action="store_true", help="ポスター用: マス内の数字と下の注記をなくし、文字を大きくして、--size-cm の大きさで描く")
    p.add_argument("--size-cm", type=float, nargs=2, default=[40, 12.5], metavar=("幅", "高さ"), help="--poster のときの、貼り付ける大きさ (cm)。既定 40 12.5")
    p.add_argument("--fontsize", type=float, default=22, help="--poster のときの、目盛りの文字の大きさ (pt)。既定22")
    p.add_argument("--value", choices=["rate", "share"], default="rate",
                   help="色の意味。rate=出現率 (視程<1km時間 ÷ そのマスの該当時間, 既定) / share=全観測時間に占める割合 (視程<1km時間 ÷ その年代の全観測時間。全マスの合計=その年代の出現率で、年代どうしを比べられる)")
    p.add_argument("--combine", action="store_true", help="--poster --value share のとき、指定した官署を1枚の図に縦に並べる (カラーバーは1つ)。年代の区切りは全官署で同じにする")
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
    if a.value == "share":                                                                   # 全観測時間に占める割合の色の基準 (全官署・全年代で共通)
        tot_n = {key: t[0].values.sum() for key, t in tabs.items()}
        top = max(float((t[1].values / tot_n[key] * 100).max()) for key, t in tabs.items())
        norm = Normalize(0, top)
    names = [s for s in (a.stations or allst) if s in allst]

    if a.combine:
        if not (a.poster and a.value == "share"):
            raise SystemExit("--combine は --poster --value share と一緒に使います")
        fs = a.fontsize
        fig, axs = plt.subplots(len(names), len(bounds), figsize=(a.size_cm[0] / 2.54, a.size_cm[1] / 2.54), squeeze=False, sharey="row")
        for r, s in enumerate(names):
            total = sum(tabs[(s, i)][1] for i in range(len(bounds)))
            clo, chi = crop(total)
            for i in range(len(bounds)):
                ax = axs[r, i]
                n, k = (t.loc[clo:chi, clo:chi] for t in tabs[(s, i)])
                label, y0, y1 = period_label(bounds, i, last_year)
                sm = heat_share(ax, n, k, tot_n[(s, i)], label, norm, cmap)
                tot = tabs[(s, i)][1].values.sum() / tot_n[(s, i)] * 100
                print(f"{s} {label}: 全観測 {int(tot_n[(s, i)])}時間, 視程<1km {int(tabs[(s, i)][1].values.sum())}時間 = {tot:.2f}%")
                ax.text(.5, 1.01, f"視程1km未満の\n割合の合計 {tot:.1f}%", transform=ax.transAxes, ha="center", va="bottom", fontsize=fs * .72, linespacing=1.1)
                ticks = [t for t in range(-(-clo // 5) * 5, chi + 2, 5) if clo < t < chi + 1]
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)
                ax.tick_params(labelsize=fs)
                ax.set_title(label.replace("〜", "～") if r == 0 else "", fontsize=fs * 1.1, pad=fs * 2.4)
                ax.set_xlabel("気温 (℃)" if r == len(names) - 1 else "", fontsize=fs)
                ax.set_ylabel(f"{s}\n海面水温 (℃)" if i == 0 else "", fontsize=fs)
        fig.tight_layout(pad=.9, w_pad=.5, h_pad=.6, rect=(0, 0, .865, 1))
        top, bot = axs[0, -1].get_position(), axs[-1, -1].get_position()
        cb = fig.colorbar(sm, cax=fig.add_axes([.895, bot.y0 + .05 * (top.y1 - bot.y0), .013, .9 * (top.y1 - bot.y0)]))
        cb.set_label("全時間に対する視程1km未満の割合 (%)", fontsize=fs * .75, labelpad=fs * .5)
        cb.ax.tick_params(labelsize=fs * .75)
        if a.save_dir:
            os.makedirs(a.save_dir, exist_ok=True)
            path = os.path.join(a.save_dir, "fog_ta_sst_period_" + "_".join(names) + "_poster.png")
            fig.savefig(path, dpi=130)
            print("保存:", path)
            plt.close(fig)
        else:
            plt.show()
        return

    for s in names:
        total = sum(tabs[(s, i)][1] for i in range(len(bounds)))
        clo, chi = crop(total)
        if a.poster:
            fs = a.fontsize
            fig, axs = plt.subplots(1, len(bounds), figsize=(a.size_cm[0] / 2.54, a.size_cm[1] / 2.54), squeeze=False, sharey=True)
            for i in range(len(bounds)):
                n, k = (t.loc[clo:chi, clo:chi] for t in tabs[(s, i)])
                label, y0, y1 = period_label(bounds, i, last_year)
                if a.value == "share":
                    sm = heat_share(axs[0, i], n, k, tot_n[(s, i)], label, norm, cmap)
                    ax = axs[0, i]
                    total = tabs[(s, i)][1].values.sum() / tot_n[(s, i)] * 100
                    print(f"{s} {label}: 全観測 {int(tot_n[(s, i)])}時間, 視程<1km {int(tabs[(s, i)][1].values.sum())}時間 = {total:.2f}%")
                    ax.text(.5, 1.01, f"視程1km未満の割合の合計 {total:.1f}%", transform=ax.transAxes, ha="center", va="bottom", fontsize=fs * .62)
                    ax.set_xlabel("気温 (℃)", fontsize=fs)
                    ax.set_xticks([t for t in range(-(-clo // 5) * 5, chi + 2, 5) if clo < t < chi + 1])
                    ax.set_yticks([t for t in range(-(-clo // 5) * 5, chi + 2, 5) if clo < t < chi + 1])
                    ax.tick_params(labelsize=fs)
                    ax.set_title(label.replace("〜", "～"), fontsize=fs * 1.1, pad=fs * 1.1)
                    ax.set_ylabel("海面水温 (℃)" if i == 0 else "", fontsize=fs)
                    continue
                sm = ts.heat(axs[0, i], n, k, label, norm, cmap, numbers=False)
                ax = axs[0, i]
                ax.tick_params(labelsize=fs)
                ticks = [t for t in range(-(-clo // 5) * 5, chi + 2, 5)]   # 5℃ごとの目盛り (隣のパネルと文字がぶつからないよう、端は避ける)
                ticks = [t for t in ticks if clo < t < chi + 1]
                ax.set_xticks(ticks)
                ax.set_yticks(ticks)
                ax.set_title(label, fontsize=fs * 1.1, pad=4)
                ax.set_xlabel("気温 (℃)", fontsize=fs)
                ax.set_ylabel("海面水温 (℃)" if i == 0 else "", fontsize=fs)
            fig.tight_layout(pad=.9, w_pad=.5, rect=(0, 0, .885, 1))
            bb = axs[0, -1].get_position()
            cb = fig.colorbar(sm, cax=fig.add_axes([.915, bb.y0 + .08 * bb.height, .014, .84 * bb.height]))
            cb.set_label("視程1km未満の割合 (%)" if a.value == "rate" else "全時間に対する視程1km未満の割合 (%)", fontsize=fs * .8)
            cb.ax.tick_params(labelsize=fs * .8)
        else:
            fig, axs = plt.subplots(1, len(bounds), figsize=(6.4 * len(bounds) + 1.2, 7), squeeze=False)
            for i in range(len(bounds)):
                n, k = (t.loc[clo:chi, clo:chi] for t in tabs[(s, i)])
                label, y0, y1 = period_label(bounds, i, last_year)
                head = f"{s} {label} ({hourly_note(start.get(s), y0, y1)})\n該当{int(n.values.sum())}時間, 視程<1km {int(k.values.sum())}時間"
                if a.value == "share":
                    kt = tabs[(s, i)][1].values.sum()      # 題は、図に出していない端のマスも含めた、その年代の全体
                    head = f"{s} {label} ({hourly_note(start.get(s), y0, y1)})\n全観測{int(tot_n[(s, i)])}時間, 視程<1km {int(kt)}時間 = {kt / tot_n[(s, i)] * 100:.1f}%"
                    sm = heat_share(axs[0, i], n, k, tot_n[(s, i)], head, norm, cmap, numbers=True)
                else:
                    sm = ts.heat(axs[0, i], n, k, head, norm, cmap, numbers=True)
                axs[0, i].title.set_fontsize(10)
            fig.tight_layout(rect=(0, .07, .94, 1))
            colnote = ("色=視程<1km時間 ÷ その年代の全観測時間 (%) (全マスの合計=題の%)。灰色=観測はあったが視程<1kmは0時間。空白=観測なし。" if a.value == "share" else
                       "色=視程<1kmの出現率 (上÷下)。灰色=視程<1kmが0回、または該当した時間が40時間未満。空白=該当なし。")
            fig.text(.01, .004,
                     "マス内の数字: 上=視程<1kmだった時間 / 下=該当した時間。" + colnote +
                     f"\n破線=気温と海面水温が等しい。目盛りはマスの境目 (各マスは例えば16.0℃以上17.0℃未満)。{s}は視程が毎時になった年={start.get(s) or '不明'} (観測の自動化 {fp.AUTOMATION.get(s, '不明')}、気象観測統計指針 表3.4－6。それ以前は目視で間引かれた時刻だけの観測)。",
                     fontsize=8)
            bb = axs[0, -1].get_position()
            fig.colorbar(sm, cax=fig.add_axes([.955, bb.y0 + .1 * bb.height, .012, .8 * bb.height])).set_label("視程<1kmの出現率 (%)" if a.value == "rate" else "全観測時間に占める視程<1kmの割合 (%)", fontsize=9)
        if a.save_dir:
            os.makedirs(a.save_dir, exist_ok=True)
            path = os.path.join(a.save_dir, f"fog_ta_sst_period_{s}{'_poster' if a.poster else ''}.png")
            fig.savefig(path, dpi=130)
            print("保存:", path)
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    main()
