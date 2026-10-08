"""低視程(視程<1km)の出現率を官署別に示す図を、ファイルに保存せず画面に表示する (このファイル1つで動きます)。

    python fog_plot.py                  # 全期間
    python fog_plot.py --from-year 2020 # 視程が毎時になった2020年以降だけ

必要: pip install numpy pandas matplotlib
データ (このファイルと同じフォルダに置く):
    海面水温   *.txt                  (気象庁 海面水温 日別値, 12海域)
    沿岸官署   東北地方沿岸気象官署時別値/*.csv  (気象庁 時別値, 年ごと)

図: 官署ごとの、相対湿度 × (気温−海面水温) の低視程の出現率(色)。
    マス内の数字は 上=視程<1kmだった回数 / 下=該当した回数。灰色=該当回数が少なく発生率を出さない。
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


DT_BINS = [-np.inf, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, np.inf]


RH_BINS = [0, 70, 80, 85, 90, 93, 95, 97, 99, 100, 101]  # 湿度は整数で、低視程の時は99/100に集中する


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


def fog_table(d, min_n=30):
    """気温-海面水温(dT) × 相対湿度 の階級ごとの (低視程の出現率% [観測数min_n未満はNaN], 該当した回数, うち低視程の回数)。"""
    d = d.dropna(subset=["dT", "RH", "fog"]).copy()
    d["dTc"] = pd.cut(d["dT"], DT_BINS, right=False, labels=_label(DT_BINS))
    d["RHc"] = pd.cut(d["RH"], RH_BINS, right=False, labels=_label(RH_BINS, single=True))
    n = d.pivot_table(index="RHc", columns="dTc", values="fog", aggfunc="count", observed=False).fillna(0)
    p = d.pivot_table(index="RHc", columns="dTc", values="fog", aggfunc="mean", observed=False) * 100
    k = d.pivot_table(index="RHc", columns="dTc", values="fog", aggfunc="sum", observed=False).fillna(0)
    return p.where(n >= min_n), n, k


FOG_NOTE = "マス内の数字: 上=視程<1kmだった回数 / 下=該当した回数 (分子/分母)。色=発生率。灰色=該当回数が少なく発生率を出さない"


def fog_heat(ax, p, n, k, title, vmax=None):
    """マスの色=低視程の出現率。数字は 上=視程<1kmだった回数 / 下=その階級に該当した回数 (分子/分母)。
    該当回数が少なくて発生率を出さないマスは灰色で、回数だけ薄く表示する。"""
    from matplotlib.colors import ListedColormap
    ax.imshow(np.where(n.values > 0, 1.0, np.nan), origin="lower", aspect="auto",
              cmap=ListedColormap(["#e4e4e4"]), vmin=0, vmax=1)
    vmax = vmax or max(10, np.nanmax(p.values) if np.isfinite(p.values).any() else 10)
    im = ax.imshow(p.values.astype(float), origin="lower", aspect="auto", cmap="magma_r", vmin=0, vmax=vmax)
    ax.set_xticks(range(p.shape[1]), p.columns, rotation=60, fontsize=7)
    ax.set_yticks(range(p.shape[0]), p.index, fontsize=7)
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            if n.values[i, j] > 0:
                big = np.isfinite(p.values[i, j])
                ax.text(j, i, f"{int(k.values[i, j])}\n{int(n.values[i, j])}", ha="center", va="center",
                        fontsize=5.5, linespacing=1.0,
                        color=("w" if p.values[i, j] > 0.55 * vmax else "k") if big else "#888")
    ax.set(title=title, xlabel="気温 − 海面水温 (℃)", ylabel="相対湿度 (%)")
    return im


def fog_valid(df, args):
    """沿岸官署を読み、海面水温と結合して (全データ, 視程・湿度・気温・海面水温がそろった時刻だけ) を返す。"""
    coast = load_coast(args.coast_dir)
    if coast is None:
        raise SystemExit(f"沿岸官署のCSVが見つかりません: {args.coast_dir} (--coast-dir で指定)")
    d = fog_dataset(df, coast)
    if args.from_year:
        d = d[d["time"].dt.year >= args.from_year]
    return d, d.dropna(subset=["dT", "RH", "fog"])


def fog_station_figure(v):
    """官署別の 低視程の出現率(相対湿度 × 気温−海面水温) の図 (Figure) を返す。"""
    stations = [s for s in STATION_SST if s in set(v["station"])]
    ncol = (len(stations) + 1) // 2
    tabs = {name: fog_table(v[v["station"] == name], min_n=15) for name in stations}
    vmax = max(10, max(np.nanmax(t[0].values) for t in tabs.values() if np.isfinite(t[0].values).any()))
    fig, axs = plt.subplots(2, ncol, figsize=(4.2 * ncol + 1, 8), squeeze=False)
    for ax, name in zip(axs.ravel(), stations):  # 全官署で同じ色スケール
        im = fog_heat(ax, *tabs[name], f"{name} (海域: {STATION_SST[name]})", vmax=vmax)
    fig.tight_layout(rect=(0, .03, .93, 1))
    fig.text(.01, .005, FOG_NOTE, fontsize=9)
    cax = fig.add_axes([.945, .2, .015, .6])
    fig.colorbar(im, cax=cax, label="低視程(視程<1km)の出現率 %")
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=HERE, help="海面水温 *.txt のあるフォルダ")
    p.add_argument("--coast-dir", default=COAST_DIR, help="沿岸官署の時別値CSVのフォルダ")
    p.add_argument("--from-year", type=int, help="この年以降だけ使う")
    a = p.parse_args()
    setup_font()
    _, v = fog_valid(summer(load(a.dir)), a)
    fog_station_figure(v)
    plt.show()


if __name__ == "__main__":
    main()
