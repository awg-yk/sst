"""気温または露点温度(横軸) × 海面水温(縦軸) を1℃ごとに区切って、視程<1kmの出現率を示す図 (相対湿度は使わない)。
fog_plot.py と同じフォルダに置いて実行します (fog_plot.py の読み込み処理を使います)。
露点温度は「東北地方沿岸気象官署時別値」フォルダの時別値CSV (1984年〜) にあります。

    python fog_plot_temp_sst.py                   # 気温×海面水温。沿岸10官署を並べた図を画面に表示
    python fog_plot_temp_sst.py --x td            # 露点温度×海面水温
    python fog_plot_temp_sst.py --each            # 官署ごとに1枚ずつ表示 (マス内に回数も表示)
    python fog_plot_temp_sst.py --each --save-dir out   # 官署ごとのPNGを保存
    python fog_plot_temp_sst.py --from-year 2014  # この年以降だけ使う
    python fog_plot_temp_sst.py --hourly-only     # 官署ごとに、視程が毎時になった年以降だけを使う

色: 視程<1kmの出現率(%) = 視程<1kmだった回数 ÷ 該当した回数。全官署で同じ色の基準。
灰色: 視程<1kmが0回、または該当した回数が40回未満。空白: 該当なし。
破線: 気温(または露点温度) = 海面水温。 マス内の数字 (--each のとき): 上=視程<1kmだった回数 / 下=該当した回数。
区切り: 1℃ごとで、目盛りの数字はマスの境目です。例: 気温16と17の間のマスは「16.0℃以上、17.0℃未満」(16.9℃もこのマス)。
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap

import fog_plot as fp

X = {"ta": ("Ta", "気温"), "td": ("Td", "露点温度")}   # --x の選択肢: (表の列, 名前)


def ta_sst_table(d, lo, hi, x="Ta"):
    """横軸の温度(列 x: 既定は気温 Ta)×海面水温(縦) を1℃ごとに区切った (該当した回数, うち視程<1kmだった回数)。範囲は lo〜hi ℃。"""
    d = d.dropna(subset=[x, "SST", "fog"]).copy()
    d["tx"] = np.floor(d[x]).astype(int)
    d["ty"] = np.floor(d["SST"]).astype(int)
    idx = pd.Index(range(lo, hi + 1))
    n = d.pivot_table(index="ty", columns="tx", values="fog", aggfunc="count").reindex(index=idx, columns=idx).fillna(0)
    k = d.pivot_table(index="ty", columns="tx", values="fog", aggfunc="sum").reindex(index=idx, columns=idx).fillna(0)
    return n, k


def temp_range(d, x="Ta"):
    """データのある横軸の温度・海面水温の範囲 (整数℃)。両方の軸で同じ範囲にして、対角線を見やすくする。"""
    d = d.dropna(subset=[x, "SST", "fog"])
    lo = int(np.floor(min(d[x].quantile(.001), d["SST"].quantile(.001))))
    hi = int(np.ceil(max(d[x].quantile(.999), d["SST"].quantile(.999))))
    return lo, hi


def heat(ax, n, k, title, norm, cmap, numbers, xname="気温"):
    """色=視程<1kmの出現率(%)。灰色=視程<1kmが0回、または該当した回数が MIN_N 未満。空白=該当なし。"""
    lo, hi = n.index.min(), n.index.max()
    nv, kv = n.values, k.values
    rate = fp.fog_rate(n, k).values
    # マスの辺が整数℃の位置に来るようにする。lo のマスは lo℃以上 lo+1℃未満で、目盛りの数字はマスの境目
    ext = (lo, hi + 1, lo, hi + 1)
    ax.imshow(np.where(nv > 0, 1.0, np.nan), origin="lower", extent=ext, aspect="equal",
              cmap=ListedColormap(["#e4e4e4"]), vmin=0, vmax=1)
    ax.imshow(rate, origin="lower", extent=ext, aspect="equal", cmap=cmap, norm=norm)
    ax.plot([lo, hi + 1], [lo, hi + 1], "k--", lw=.8, alpha=.6)  # 横軸の温度 = 海面水温
    if numbers:
        for i in range(nv.shape[0]):
            for j in range(nv.shape[1]):
                if nv[i, j] > 0:
                    colored = np.isfinite(rate[i, j])
                    dark = colored and norm(rate[i, j]) > 0.55
                    ax.text(lo + j + .5, lo + i + .5, f"{int(kv[i, j])}\n{int(nv[i, j])}", ha="center", va="center",
                            fontsize=5.5, linespacing=1.0, color="w" if dark else ("k" if colored else "#888"))
    ax.set_xlim(lo, hi + 1)
    ax.set_ylim(lo, hi + 1)
    ax.set_xticks(range(lo, hi + 2, 1 if numbers else 5))
    ax.set_yticks(range(lo, hi + 2, 1 if numbers else 5))
    ax.tick_params(labelsize=7)
    ax.set(title=title, xlabel=f"{xname} (℃)", ylabel="海面水温 (℃)")
    return ScalarMappable(norm=norm, cmap=cmap)


def load_data(a, x):
    """沿岸官署の時別値と海面水温から、視程と横軸の温度 (列 x) と海面水温がそろった時刻の表を返す。"""
    coast = fp.load_coast(a.coast_dir)
    if coast is None:
        raise SystemExit(f"沿岸官署のCSVが見つかりません: {a.coast_dir} (--coast-dir で指定)")
    if a.hourly_only:
        start = fp.hourly_start_years(coast)
        coast = coast[coast["time"].dt.year >= coast["station"].map(lambda s: start.get(s) or 9999)]
        print("視程が毎時になった年(この年以降を使用): " + ", ".join(f"{k} {v}" for k, v in start.items() if k in fp.STATION_SST))
    d = fp.fog_dataset(fp.summer(fp.load(a.dir)), coast)
    if a.from_year:
        d = d[d["time"].dt.year >= a.from_year]
    return d[d["vis"].notna() & d[x].notna() & d["SST"].notna()]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=fp.SST_DIR, help="海面水温 *.txt のあるフォルダ")
    p.add_argument("--coast-dir", default=fp.COAST_DIR, help="沿岸官署の時別値CSVのフォルダ")
    p.add_argument("--x", choices=list(X), default="ta", help="横軸の温度: ta=気温 (既定) / td=露点温度")
    p.add_argument("--from-year", type=int, help="この年以降だけ使う")
    p.add_argument("--hourly-only", action="store_true", help="官署ごとに、視程が毎時になった年以降だけを使う")
    p.add_argument("--each", action="store_true", help="官署ごとに1枚ずつ表示する")
    p.add_argument("--stations", nargs="*", help="官署名 (省略で沿岸10官署)")
    p.add_argument("--save-dir", help="--each のとき、画面に出さずPNGをこのフォルダに保存する")
    a = p.parse_args()

    fp.setup_font()
    x, xname = X[a.x]
    d = load_data(a, x)
    allst = [s for s in fp.STATION_SST if s in set(d["station"])]
    lo, hi = temp_range(d, x)
    tabs = {s: ta_sst_table(d[d["station"] == s], lo, hi, x) for s in allst}
    norm = fp.fog_norm(list(tabs.values()))          # 全官署で共通の色の基準
    cmap = plt.get_cmap("magma_r")
    names = [s for s in (a.stations or allst) if s in tabs]
    title = lambda s: f"{s} (海域: {fp.STATION_SST[s]})"
    note = (f"色=視程<1kmの出現率 (上÷下)。灰色=視程<1kmが0回、または該当した回数が{fp.MIN_N}回未満。空白=該当なし。"
            f"\n破線={xname}と海面水温が等しい。目盛りはマスの境目 (各マスは例えば16.0℃以上17.0℃未満)")

    if a.each:
        for s in names:
            fig, ax = plt.subplots(figsize=(10, 8.5))
            sm = heat(ax, *tabs[s], title(s), norm, cmap, True, xname)
            fig.colorbar(sm, ax=ax, shrink=.85).set_label("視程<1kmの出現率 (%)")
            fig.text(.01, .004, "マス内の数字: 上=視程<1kmだった回数 / 下=該当した回数。" + note, fontsize=7)
            fig.tight_layout(rect=(0, .05, 1, 1))
            if a.save_dir:
                os.makedirs(a.save_dir, exist_ok=True)
                path = os.path.join(a.save_dir, f"fog_{a.x}_sst_{s}.png")
                fig.savefig(path, dpi=130)
                print("保存:", path)
                plt.close(fig)
            else:
                plt.show()
        return

    ncol = (len(names) + 1) // 2
    fig, axs = plt.subplots(2, ncol, figsize=(3.9 * ncol + 1, 8), squeeze=False)
    for ax, s in zip(axs.ravel(), names):
        sm = heat(ax, *tabs[s], title(s), norm, cmap, False, xname)
    for ax in axs.ravel()[len(names):]:
        ax.axis("off")
    fig.tight_layout(rect=(0, .06, .93, 1))
    fig.text(.01, .004, note, fontsize=9)
    cax = fig.add_axes([.945, .2, .015, .6])
    fig.colorbar(sm, cax=cax).set_label("視程<1kmの出現率 (%)", fontsize=9)
    plt.show()


if __name__ == "__main__":
    main()
