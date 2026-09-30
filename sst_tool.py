"""東北〜津軽海峡沿岸 海面水温(日別)の推移を調べるツール

データ: 同フォルダの *.txt (気象庁 海面水温 日別値 / 列: yyyy,mm,dd,areaNo.,flag,Temp.)
        flag: R=確定値, P=速報値

使い方:
    python sst_tool.py summary                 # 海域ごとの年平均・長期トレンド(℃/10年)を表示しCSV保存
    python sst_tool.py plot                    # 図を out/ に保存 (推移・偏差・年×日ヒートマップ・年平均トレンド)
    python sst_tool.py plot --area 宮城県沿岸   # 1海域だけ
    python sst_tool.py player --start 2020-01-01 --end 2020-12-31 --save out/player.gif
    python sst_tool.py player                  # 画面表示で再生 (要GUI)
必要: pip install numpy pandas matplotlib (--save gif は pillow)
"""
import argparse
import glob
import os

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
BASE_YEARS = (1991, 2020)  # 平年値の期間 (気象庁と同じ30年)
TMIN, TMAX = 0, 28

# 海域の代表点 (概算値。地図上の位置合わせ用なので必要に応じて修正してください)
AREAS = {
    113: ("津軽海峡の西側", 41.55, 140.15),
    114: ("青森県日本海沿岸", 40.85, 139.75),
    115: ("津軽海峡", 41.60, 140.75),
    116: ("津軽海峡の東側", 41.60, 141.40),
    117: ("青森県太平洋沿岸", 40.80, 141.90),
    130: ("陸奥湾", 41.05, 140.90),
    131: ("秋田県沿岸", 39.70, 139.70),
    132: ("岩手県北部沿岸", 40.05, 142.30),
    133: ("岩手県南部沿岸", 39.15, 142.30),
    134: ("山形県沿岸", 38.85, 139.45),
    135: ("宮城県沿岸", 38.15, 141.55),
    136: ("福島県沿岸", 37.35, 141.30),
}


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


def climatology(df):
    """平年値(日別)。うるう日を含めた月日ごとの平均を7日移動(循環)で平滑化。"""
    base = df[(df.index.year >= BASE_YEARS[0]) & (df.index.year <= BASE_YEARS[1])]
    key = base.index.strftime("%m-%d")
    clim = base.groupby(key).mean()
    ext = pd.concat([clim.iloc[-3:], clim, clim.iloc[:3]])
    return ext.rolling(7, center=True).mean().iloc[3:-3]


def anomaly(df):
    clim = climatology(df)
    key = df.index.strftime("%m-%d")
    return df - clim.reindex(key).values


def annual_trend(df):
    """年平均(欠測が多い年を除く)と線形トレンド(℃/10年)。今年は途中までなので除外。"""
    last_full = df.index.max().year - (0 if df.index.max().strftime("%m-%d") >= "12-25" else 1)
    yr = df[df.index.year <= last_full]
    cnt = yr.groupby(yr.index.year).count()
    mean = yr.groupby(yr.index.year).mean().where(cnt >= 330)
    rows = {}
    for c in mean:
        m = mean[c].dropna()
        slope = np.polyfit(m.index.values, m.values, 1)[0] * 10
        rows[c] = dict(第1年=m.index[0], 最終年=m.index[-1], 年平均_最初5年=m.iloc[:5].mean(),
                       年平均_最近5年=m.iloc[-5:].mean(), トレンド_10年あたり=slope)
    summ = pd.DataFrame(rows).T
    summ["差_最近5年-最初5年"] = summ["年平均_最近5年"] - summ["年平均_最初5年"]
    return mean, summ.sort_values("トレンド_10年あたり", ascending=False)


def cmd_summary(df, _):
    mean, summ = annual_trend(df)
    pd.set_option("display.width", 200, "display.float_format", "{:.2f}".format)
    summ = summ.astype({"第1年": int, "最終年": int})
    print(f"期間: {df.index.min():%Y-%m-%d} 〜 {df.index.max():%Y-%m-%d}  海域数: {df.shape[1]}")
    print(summ.to_string())
    os.makedirs(OUT, exist_ok=True)
    summ.to_csv(os.path.join(OUT, "trend_summary.csv"), encoding="utf-8-sig")
    mean.to_csv(os.path.join(OUT, "annual_mean.csv"), encoding="utf-8-sig")
    print("保存: out/trend_summary.csv, out/annual_mean.csv")


def cmd_plot(df, args):
    setup_font()
    os.makedirs(OUT, exist_ok=True)
    areas = [args.area] if args.area else list(df.columns)
    mean, _ = annual_trend(df)
    anom = anomaly(df)
    cmap = plt.get_cmap("tab20")

    # 1) 年平均の推移 + 回帰直線
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, c in enumerate(areas):
        m = mean[c].dropna()
        ax.plot(m.index, m.values, marker="o", ms=3, lw=1.2, color=cmap(i), label=c)
    ax.set(title="年平均海面水温の推移", xlabel="年", ylabel="℃")
    ax.grid(alpha=.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "annual_mean.png"), dpi=130)
    plt.close(fig)

    # 2) 平年偏差 (1年移動平均) : 温暖化・ここ数年の高水温が見える
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, c in enumerate(areas):
        ax.plot(anom[c].rolling(365, min_periods=300).mean(), lw=1.2, color=cmap(i), label=c)
    ax.axhline(0, color="k", lw=.8)
    ax.set(title=f"平年偏差(1年移動平均, 平年={BASE_YEARS[0]}-{BASE_YEARS[1]})", ylabel="℃")
    ax.grid(alpha=.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "anomaly_365d.png"), dpi=130)
    plt.close(fig)

    # 3) 海域ごと: 年×日ヒートマップ(平年偏差) と 直近の推移
    for c in areas:
        a = anom[c].dropna()
        piv = a.groupby([a.index.year, a.index.dayofyear]).mean().unstack()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw=dict(height_ratios=[3, 2]))
        im = ax1.imshow(piv.values, aspect="auto", cmap="RdBu_r", vmin=-4, vmax=4,
                        extent=[1, 366, piv.index.max() + .5, piv.index.min() - .5])
        fig.colorbar(im, ax=ax1, label="平年偏差 ℃")
        ax1.set(title=f"{c}: 平年偏差 (年×通日)", xlabel="通日", ylabel="年")
        clim = climatology(df)[c]
        recent = df[c][df.index >= df.index.max() - pd.Timedelta(days=730)]
        for y in sorted(set(recent.index.year)):
            r = recent[recent.index.year == y]
            ax2.plot(r.index.dayofyear, r.values, label=str(y))
        ax2.plot(np.arange(1, len(clim) + 1), clim.values, "k--", lw=1, label="平年")
        ax2.set(title="直近の水温と平年", xlabel="通日", ylabel="℃")
        ax2.grid(alpha=.3)
        ax2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"detail_{c}.png"), dpi=110)
        plt.close(fig)
    print("保存先:", OUT)


def cmd_player(df, args):
    from matplotlib.animation import FuncAnimation
    from matplotlib.widgets import Slider  # noqa: F401  (GUI表示時のみ使用)
    setup_font()
    if args.save:
        matplotlib.use("Agg")
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else start + pd.offsets.YearEnd(0)
    sub = df.loc[start:end]
    if sub.empty:
        raise SystemExit("指定期間にデータがありません")
    sub = sub.iloc[::args.step]
    by_name = {v[0]: v for v in AREAS.values()}
    names = [c for c in sub.columns if c in by_name]
    lon = np.array([by_name[c][2] for c in names])
    lat = np.array([by_name[c][1] for c in names])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set(xlim=(138.8, 143.0), ylim=(36.8, 42.2), aspect=1 / np.cos(np.radians(39.5)),
           xlabel="経度", ylabel="緯度")
    ax.grid(alpha=.3)
    sc = ax.scatter(lon, lat, s=900, c=np.zeros(len(names)), cmap="turbo", vmin=TMIN, vmax=TMAX,
                    edgecolors="k")
    fig.colorbar(sc, ax=ax, shrink=.7, label="海面水温 ℃")
    labels = [ax.text(x, y - .17, n, ha="center", fontsize=7) for n, x, y in zip(names, lon, lat)]
    vals = [ax.text(x, y, "", ha="center", va="center", fontsize=7, weight="bold")
            for x, y in zip(lon, lat)]
    title = ax.set_title("")
    del labels

    def draw(i):
        row = sub.iloc[i][names].values.astype(float)
        sc.set_array(row)
        for t, v in zip(vals, row):
            t.set_text("" if np.isnan(v) else f"{v:.1f}")
        title.set_text(f"海面水温 {sub.index[i]:%Y-%m-%d}")
        return sc, title, *vals

    ani = FuncAnimation(fig, draw, frames=len(sub), interval=args.interval, blit=False)
    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        ani.save(args.save, writer="pillow", fps=max(1, 1000 // args.interval))
        print("保存:", args.save)
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=HERE, help="*.txt のあるフォルダ")
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("summary")
    pp = sp.add_parser("plot")
    pp.add_argument("--area", help="海域名 (例: 宮城県沿岸)。省略で全海域")
    pl = sp.add_parser("player")
    pl.add_argument("--start", default="2020-01-01")
    pl.add_argument("--end")
    pl.add_argument("--step", type=int, default=1, help="何日ごとに再生するか")
    pl.add_argument("--interval", type=int, default=100, help="1コマのms")
    pl.add_argument("--save", help="GIF保存先 (指定時は画面表示しない)")
    args = p.parse_args()
    df = load(args.dir)
    {"summary": cmd_summary, "plot": cmd_plot, "player": cmd_player}[args.cmd](df, args)


if __name__ == "__main__":
    main()
