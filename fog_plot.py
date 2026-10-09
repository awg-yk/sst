"""低視程(視程<1km)だった回数を、相対湿度 × (気温−海面水温) の階級ごとに官署別に示す図を、
ファイルに保存せず画面に表示する (このファイル1つで動きます)。

    python fog_plot.py                  # 全期間
    python fog_plot.py --from-year 2020 # 2020年以降だけ
    python fog_plot.py --hourly-only    # 官署ごとに、視程が毎時になった年以降だけ

既定は1982年からの全期間です (視程の記録は1989年から。1989年以前は図に入りません。
2019年以前は視程が3〜6時間おきなど間引かれている官署が多く、観測時刻の偏りがあります)。

必要: pip install numpy pandas matplotlib
データ (このファイルと同じフォルダに置く):
    海面水温   *.txt                  (気象庁 海面水温 日別値, 12海域)
    沿岸官署   東北地方沿岸気象官署時別値/*.csv  (気象庁 時別値, 年ごと)

図: 官署ごとに、マスの色=視程<1kmだった回数。マス内の数字は 上=視程<1kmだった回数 / 下=該当した回数。
    灰色=該当した時刻はあるが視程<1kmは無かった (数字は 0 / 該当回数)。空白=該当した時刻が無い。
"""
import argparse
import csv
import glob
import os

import matplotlib
import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
COAST_DIR = os.path.join(HERE, "東北地方沿岸気象官署時別値")  # 気象庁の時別値(1時間ごと)


def setup_font():
    have = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    for name in ("Yu Gothic", "Meiryo", "MS Gothic", "Hiragino Sans", "IPAexGothic",
                 "IPAGothic", "Noto Sans CJK JP", "WenQuanYi Zen Hei"):
        if name in have:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def load(data_dir=HERE):
    """全海域を読み込み、日付×海域名の DataFrame を返す。速報値(P)も含む。"""
    cols = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "*.txt"))):
        name = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path, skipinitialspace=True, dtype=str)
        df = df[df["yyyy"] != "yyyy"].dropna()  # 末尾に繰り返しヘッダがある
        date = pd.to_datetime(dict(year=df["yyyy"].astype(int), month=df["mm"].astype(int),
                                   day=df["dd"].astype(int)))
        s = pd.Series(df["Temp."].astype(float).values, index=date)
        s = s[~s.index.duplicated()].sort_index()
        cols[name] = s.where(s > -90)  # 欠測値対策
    if not cols:
        raise SystemExit("*.txt が見つかりません: " + data_dir)
    return pd.DataFrame(cols)


def summer(df):
    """毎年の 6/1〜8/31 だけを取り出す。"""
    return df[df.index.month.isin([6, 7, 8])]


def _read_rows(path):
    for enc in ("utf-8-sig", "cp932"):
        try:
            with open(path, encoding=enc, newline="") as f:
                return list(csv.reader(f))
        except UnicodeDecodeError:
            continue
    raise SystemExit("文字コードが判別できません: " + path)


# 官署 -> 最も近い海面水温の海域 (違うと思ったらここを直してください)
STATION_SST = {
    "むつ": "陸奥湾", "青森": "陸奥湾", "深浦": "青森県日本海沿岸", "八戸": "青森県太平洋沿岸",
    "秋田": "秋田県沿岸", "酒田": "山形県沿岸", "宮古": "岩手県北部沿岸", "大船渡": "岩手県南部沿岸",
    "石巻": "宮城県沿岸", "小名浜": "福島県沿岸",
}


FOG_KM = 1.0  # 視程がこれ未満 = 低視程 (霧とは限らない: 降水・煙霧なども含む)


def load_coast(coast_dir=COAST_DIR):
    """沿岸官署の時別値CSV(年ごと)を読み、時刻×官署の縦長DataFrame(Ta=気温, RH=相対湿度, vis=視程km)を返す。
    列の位置は見出し行から判定する。品質情報が8(正常)の値だけを使う。"""
    files = sorted(glob.glob(os.path.join(coast_dir, "*.csv")))
    if not files:
        return None
    out = []
    for path in files:
        rows = _read_rows(path)
        h = next((i for i, r in enumerate(rows) if r and r[0].startswith("年月日時")), None)
        if h is None:
            print("  読み飛ばし(形式が違う):", os.path.basename(path))
            continue
        names, titles = rows[h - 1], rows[h]
        data = []
        for r in rows[h + 1:]:
            if r and r[0] and r[0][0].isdigit():
                data.append(r)
        t = pd.to_datetime([r[0] for r in data], format="%Y/%m/%d %H:%M:%S", errors="coerce")
        ok = ~t.isna()
        data, t = [r for r, k in zip(data, ok) if k], t[ok]
        width = max(len(r) for r in data)
        data = [r + [""] * (width - len(r)) for r in data]
        arr = np.array(data, dtype=object)
        for st in dict.fromkeys(n for n in names[1:] if n):
            cols = [i for i in range(1, len(names)) if names[i] == st]
            vals = {}
            for key, title in (("Ta", "気温"), ("RH", "相対湿度"), ("vis", "視程")):
                c = next((i for i in cols if titles[i].startswith(title)), None)
                if c is None:
                    vals[key] = np.full(len(t), np.nan)
                    continue
                v = pd.to_numeric(pd.Series(arr[:, c]), errors="coerce").values
                q = arr[:, c + 1] if c + 1 < width else np.full(len(t), "8")
                vals[key] = np.where(q == "8", v, np.nan)
            out.append(pd.DataFrame({"time": t, "station": st, **vals}))
    return pd.concat(out, ignore_index=True)


def fog_dataset(sst, coast):
    """官署の各時刻に、その日の近い海域の海面水温を付ける。dT=気温-海面水温。"""
    c = coast[coast["station"].isin(STATION_SST)].copy()
    c["day"] = (c["time"] - pd.Timedelta(hours=1)).dt.normalize()  # 0時は前日24時
    long = sst.stack().rename("SST").rename_axis(["day", "area"]).reset_index()
    c["area"] = c["station"].map(STATION_SST)
    c = c.merge(long, on=["day", "area"], how="left")
    c["dT"] = c["Ta"] - c["SST"]
    c["fog"] = np.where(c["vis"].notna(), c["vis"] < FOG_KM, np.nan)
    return c


DT_BINS = [-np.inf] + list(range(-6, 7)) + [np.inf]  # 気温−海面水温は1℃刻み (両端だけ -6未満 と 6以上)


RH_BINS = [-np.inf, 80, 85, 90, 95, np.inf]  # 湿度は 80%未満、80〜85、85〜90、90〜95、95以上 (湿度計の誤差が5%程度なので100%も95以上に含める)
MIN_N = 40  # 該当した回数がこれ未満の階級は、出現率が不安定なので灰色にする


def _label(bins, fmt="{:g}", single=False):
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        if np.isinf(lo):
            out.append(f"{fmt.format(hi)}未満")
        elif np.isinf(hi):
            out.append(f"{fmt.format(lo)}以上")
        elif single and hi - lo == 1:
            out.append(fmt.format(lo))
        else:
            out.append(f"{fmt.format(lo)}〜{fmt.format(hi)}")
    return out


def fog_table(d):
    """気温-海面水温(dT) × 相対湿度 の階級ごとの (該当した回数, うち視程<1kmだった回数)。"""
    d = d.dropna(subset=["dT", "RH", "fog"]).copy()
    d["dTc"] = pd.cut(d["dT"], DT_BINS, right=False, labels=_label(DT_BINS))
    d["RHc"] = pd.cut(d["RH"], RH_BINS, right=False, labels=_label(RH_BINS, single=True))
    n = d.pivot_table(index="RHc", columns="dTc", values="fog", aggfunc="count", observed=False).fillna(0)
    k = d.pivot_table(index="RHc", columns="dTc", values="fog", aggfunc="sum", observed=False)
    return n, k.reindex(index=n.index, columns=n.columns).fillna(0)


FOG_NOTE = (f"マス内の数字: 上=視程<1kmだった回数 / 下=該当した回数。色=視程<1kmの出現率 (上÷下)。"
            f"灰色=視程<1kmが0回、または該当した回数が{MIN_N}回未満。空白=該当なし。\n目盛りはマスの境目 (例: 湿度85と90の間のマスは85%以上90%未満。<は未満、+は以上)")


def fog_rate(n, k):
    """階級ごとの出現率 (%) = 視程<1kmだった回数 ÷ 該当した回数。
    該当した回数が MIN_N 未満、または視程<1kmが0回の階級は NaN (図では灰色)。"""
    rate = k / n.where(n > 0) * 100
    return rate.where((n >= MIN_N) & (k > 0))


def fog_norm(tables):
    """全官署で共通の色の基準 (出現率の最大に合わせる)。tables: [(n, k), ...]"""
    from matplotlib.colors import Normalize
    top = max([fog_rate(n, k).values[np.isfinite(fog_rate(n, k).values)].max(initial=0) for n, k in tables] + [1])
    return Normalize(0, max(10.0, float(top)))


def fog_heat(ax, n, k, title, norm, cmap):
    """色=視程<1kmの出現率(%)。灰色=視程<1kmが0回、または該当した回数が MIN_N 未満。空白=該当なし。"""
    from matplotlib.colors import ListedColormap
    from matplotlib.cm import ScalarMappable
    nv, kv = n.values, k.values
    rate = fog_rate(n, k).values
    nrow, ncol = nv.shape
    ext = (0, ncol, 0, nrow)    # マスの辺が 0,1,2,... に来る。目盛りの数字はマスの境目 (区間の端の値)
    ax.imshow(np.where(nv > 0, 1.0, np.nan), origin="lower", extent=ext, aspect="auto",
              cmap=ListedColormap(["#e4e4e4"]), vmin=0, vmax=1)
    ax.imshow(rate, origin="lower", extent=ext, aspect="auto", cmap=cmap, norm=norm)
    # 目盛り: マスの境目に区間の端の値を書く。両端のマスは中央に「未満」「以上」を書く
    dt, rh = DT_BINS[1:-1], RH_BINS[1:-1]
    ax.set_xticks(range(1, ncol), [f"{v:g}" for v in dt], fontsize=7)
    ax.set_yticks(range(1, nrow), [f"{v:g}" for v in rh], fontsize=7)
    ax.set_xticks([.5, ncol - .5], [f"<{dt[0]:g}", f"{dt[-1]:g}+"], minor=True, fontsize=7)
    ax.set_yticks([.5, nrow - .5], [f"<{rh[0]:g}", f"{rh[-1]:g}+"], minor=True, fontsize=7)
    ax.tick_params(axis="x", which="minor", length=0, pad=12)   # 境目の数字より一段外側
    ax.tick_params(axis="y", which="minor", length=0, pad=22)
    for i in range(nrow):
        for j in range(ncol):
            if nv[i, j] > 0:
                colored = np.isfinite(rate[i, j])
                dark = colored and norm(rate[i, j]) > 0.55
                ax.text(j + .5, i + .5, f"{int(kv[i, j])}\n{int(nv[i, j])}", ha="center", va="center",
                        fontsize=6.5, linespacing=1.0, color="w" if dark else ("k" if colored else "#888"))
    ax.set(title=title, xlabel="気温 − 海面水温 (℃)", ylabel="相対湿度 (%)")
    return ScalarMappable(norm=norm, cmap=cmap)


def hourly_start_years(coast, full=0.9, floor=0.7):
    """官署ごとに、視程が毎時で記録されるようになった最初の年。
    その年の視程の記録率(6〜8月の全時刻のうち値のある割合)が90%以上で、以後の年も70%を下回らない最初の年。"""
    yr = coast["time"].dt.year
    cov = coast.groupby(["station", yr])["vis"].apply(lambda s: s.notna().mean()).unstack(0)
    start = {}
    for st in cov.columns:
        c = cov[st]
        start[st] = next((y for y in c.index if c.loc[y] >= full and c.loc[y:].min() >= floor), None)
    return start


def fog_valid(df, args):
    """沿岸官署を読み、海面水温と結合して (全データ, 視程・湿度・気温・海面水温がそろった時刻だけ) を返す。
    既定では、官署ごとに視程が毎時になった年以降だけを使う (間引かれた古い年は観測時刻が偏るので除く)。"""
    coast = load_coast(args.coast_dir)
    if coast is None:
        raise SystemExit(f"沿岸官署のCSVが見つかりません: {args.coast_dir} (--coast-dir で指定)")
    start = hourly_start_years(coast)
    if getattr(args, "hourly_only", False):
        coast = coast[coast["time"].dt.year >= coast["station"].map(lambda s: start.get(s) or 9999)]
        print("視程が毎時になった年(この年以降を使用): " + ", ".join(f"{k} {v}" for k, v in start.items() if k in STATION_SST))
    d = fog_dataset(df, coast)
    if args.from_year:
        d = d[d["time"].dt.year >= args.from_year]
    v = d.dropna(subset=["dT", "RH", "fog"])
    v.attrs["start"] = start if getattr(args, "hourly_only", False) else {}
    return d, v


def fog_station_figure(v):
    """官署別の 相対湿度 × 気温−海面水温 の図 (Figure) を返す。色=視程<1kmの出現率 (全官署で同じ色の基準)。"""
    stations = [s for s in STATION_SST if s in set(v["station"])]
    ncol = (len(stations) + 1) // 2
    start = v.attrs.get("start", {})
    tabs = {name: fog_table(v[v["station"] == name]) for name in stations}
    norm = fog_norm(list(tabs.values()))
    cmap = plt.get_cmap("magma_r")
    fig, axs = plt.subplots(2, ncol, figsize=(4.2 * ncol + 1, 8), squeeze=False)
    for ax, name in zip(axs.ravel(), stations):
        im = fog_heat(ax, *tabs[name], f"{name} (海域: {STATION_SST[name]}"
                      + (f", {start[name]}年〜" if start.get(name) else "") + ")", norm, cmap)
    fig.tight_layout(rect=(0, .03, .93, 1))
    fig.text(.01, .005, FOG_NOTE, fontsize=9)
    cax = fig.add_axes([.945, .2, .015, .6])
    fig.colorbar(im, cax=cax).set_label("視程<1kmの出現率 (%)", fontsize=9)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=HERE, help="海面水温 *.txt のあるフォルダ")
    p.add_argument("--coast-dir", default=COAST_DIR, help="沿岸官署の時別値CSVのフォルダ")
    p.add_argument("--from-year", type=int, help="この年以降だけ使う (官署ごとの毎時になった年と合わせ、遅い方を使う)")
    p.add_argument("--hourly-only", action="store_true", help="官署ごとに、視程が毎時になった年以降だけを使う (既定は1982年からの全期間。視程の記録は1989年から)")
    a = p.parse_args()
    setup_font()
    _, v = fog_valid(summer(load(a.dir)), a)
    fog_station_figure(v)
    plt.show()


if __name__ == "__main__":
    main()
