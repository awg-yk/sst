"""東北〜津軽海峡沿岸 海面水温(日別)の推移を調べるツール

データ: 同フォルダの *.txt (気象庁 海面水温 日別値 / 列: yyyy,mm,dd,areaNo.,flag,Temp.)
        flag: R=確定値, P=速報値

使い方:
    python sst_tool.py summary                 # 海域ごとの年平均・長期トレンド(℃/10年)を表示しCSV保存
    python sst_tool.py plot                    # 図を out/ に保存 (推移・偏差・年×日ヒートマップ・年平均トレンド)
    python sst_tool.py plot --area 宮城県沿岸   # 1海域だけ
    python sst_tool.py map --date 2025-08-15   # その日の海水温を東北の地図+等温線で表示(out/にPNG保存)
    python sst_tool.py viewer                  # 夏季(6/1〜8/31)のみ。年・月・日をボタン/スライダー/キーで操作
    python sst_tool.py --summer summary        # 夏季だけで集計 (plot / map / player にも付けられる)
    python sst_tool.py player --start 2025-06-01 --end 2025-09-30   # 期間を再生 (画面表示)
    python sst_tool.py player --start 2025-06-01 --end 2025-09-30 --step 3 --save out/player.gif
必要: pip install numpy pandas matplotlib (--save gif は pillow)
"""
import argparse
import glob
import math
import os

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
BASE_YEARS = (1991, 2020)  # 平年値の期間 (気象庁と同じ30年)

# 海域の代表点 (概算値。地図上の位置合わせ用なので必要に応じて修正してください)
AREAS = {
    113: ("津軽海峡の西側", 41.35, 140.35),
    114: ("青森県日本海沿岸", 40.85, 139.75),
    115: ("津軽海峡", 41.55, 140.70),
    116: ("津軽海峡の東側", 41.60, 141.50),
    117: ("青森県太平洋沿岸", 40.80, 141.90),
    130: ("陸奥湾", 41.05, 140.90),
    131: ("秋田県沿岸", 39.70, 139.70),
    132: ("岩手県北部沿岸", 40.05, 142.15),
    133: ("岩手県南部沿岸", 39.15, 142.20),
    134: ("山形県沿岸", 38.85, 139.45),
    135: ("宮城県沿岸", 38.20, 141.60),
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


def summer(df):
    """毎年の 6/1〜8/31 だけを取り出す。"""
    return df[df.index.month.isin([6, 7, 8])]


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


def annual_trend(df, expected=365):
    """年平均(欠測が多い年・途中までの年を除く)と線形トレンド(℃/10年)。
    expected: 1年に期待する日数 (夏季だけなら92)。"""
    cnt = df.groupby(df.index.year).count()
    mean = df.groupby(df.index.year).mean().where(cnt >= 0.9 * expected)
    rows = {}
    for c in mean:
        m = mean[c].dropna()
        slope = np.polyfit(m.index.values, m.values, 1)[0] * 10
        rows[c] = dict(第1年=m.index[0], 最終年=m.index[-1], 年平均_最初5年=m.iloc[:5].mean(),
                       年平均_最近5年=m.iloc[-5:].mean(), トレンド_10年あたり=slope)
    summ = pd.DataFrame(rows).T
    summ["差_最近5年-最初5年"] = summ["年平均_最近5年"] - summ["年平均_最初5年"]
    return mean, summ.sort_values("トレンド_10年あたり", ascending=False)


def cmd_summary(df, args):
    mean, summ = annual_trend(df, 92 if args.summer else 365)
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
    mean, _ = annual_trend(df, 92 if args.summer else 365)
    anom = anomaly(df)
    label = "夏季(6〜8月)" if args.summer else "年"
    cmap = plt.get_cmap("tab20")

    # 1) 年平均の推移 + 回帰直線
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, c in enumerate(areas):
        m = mean[c].dropna()
        ax.plot(m.index, m.values, marker="o", ms=3, lw=1.2, color=cmap(i), label=c)
    ax.set(title=f"{label}平均海面水温の推移", xlabel="年", ylabel="℃")
    ax.grid(alpha=.3)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "annual_mean.png"), dpi=130)
    plt.close(fig)

    # 2) 平年偏差 : 温暖化・ここ数年の高水温が見える
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, c in enumerate(areas):
        if args.summer:  # 夏季だけを繋ぐと移動平均は意味がないので年ごとの夏季平均
            a = anom[c].groupby(anom.index.year).mean()
            ax.plot(a.index, a.values, marker="o", ms=3, lw=1.2, color=cmap(i), label=c)
        else:
            ax.plot(anom[c].rolling(365, min_periods=300).mean(), lw=1.2, color=cmap(i), label=c)
    ax.axhline(0, color="k", lw=.8)
    ax.set(title=f"平年偏差({'夏季平均' if args.summer else '1年移動平均'}, 平年={BASE_YEARS[0]}-{BASE_YEARS[1]})", ylabel="℃")
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
                        extent=[piv.columns.min() - .5, piv.columns.max() + .5, piv.index.max() + .5, piv.index.min() - .5])
        fig.colorbar(im, ax=ax1, label="平年偏差 ℃")
        ax1.set(title=f"{c}: 平年偏差 (年×通日)", xlabel="通日", ylabel="年")
        clim = climatology(df)[c]
        recent = df[c][df.index >= df.index.max() - pd.Timedelta(days=730)]
        for y in sorted(set(recent.index.year)):
            r = recent[recent.index.year == y]
            ax2.plot(r.index.dayofyear, r.values, label=str(y))
        ax2.plot([pd.Timestamp(f"2001-{k}" if k != "02-29" else "2000-02-29").dayofyear for k in clim.index],
                 clim.values, "k--", lw=1, label="平年")
        ax2.set(title="直近の水温と平年", xlabel="通日", ylabel="℃")
        ax2.grid(alpha=.3)
        ax2.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"detail_{c}.png"), dpi=110)
        plt.close(fig)
    print("保存先:", OUT)


DEM_URL = "https://cyberjapandata.gsi.go.jp/xyz/dem_png/{z}/{x}/{y}.png"  # 国土地理院 標高タイル
EXTENT = (138.6, 143.4, 36.6, 42.2)  # 東北地方 (経度min, max, 緯度min, max)


def _tile_xy(lon, lat, z):
    n = 2 ** z
    return (lon + 180) / 360 * n, (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n


def load_land(z=8, cache_dir=os.path.join(HERE, "cache")):
    """国土地理院の標高タイルから陸地マスク(True=陸)を作る。取得済みはキャッシュ。
    ネットに繋がらない時は None を返し、海岸線なしで描画する。"""
    lon0, lon1, lat0, lat1 = EXTENT
    cache = os.path.join(cache_dir, f"land_z{z}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    import io
    import ssl
    import urllib.error
    import urllib.request
    from PIL import Image
    x0, y1 = _tile_xy(lon0, lat0, z)
    x1, y0 = _tile_xy(lon1, lat1, z)
    tx0, tx1, ty0, ty1 = int(x0), int(x1), int(y0), int(y1)
    W, H = (tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256
    land = np.zeros((H, W), bool)
    ctx = ssl.create_default_context()
    ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT  # Python3.13+ の証明書チェック厳格化への対処
    print(f"海岸線用の地図タイルを取得中 ({(tx1 - tx0 + 1) * (ty1 - ty0 + 1)} 枚) ...")
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            try:
                with urllib.request.urlopen(DEM_URL.format(z=z, x=tx, y=ty), timeout=30, context=ctx) as r:
                    im = np.array(Image.open(io.BytesIO(r.read())).convert("RGB")).astype(np.int64)
            except urllib.error.HTTPError as e:
                if e.code == 404:  # タイル無し = 海
                    continue
                print("  取得失敗:", e)
                return None
            except Exception as e:
                print("  取得失敗 (海岸線なしで続行):", e)
                return None
            v = im[..., 0] * 65536 + im[..., 1] * 256 + im[..., 2]
            land[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = v != 2 ** 23
    nx = 1000
    ny = int(nx * (lat1 - lat0) / ((lon1 - lon0) * math.cos(math.radians((lat0 + lat1) / 2))))
    lons, lats = np.linspace(lon0, lon1, nx), np.linspace(lat1, lat0, ny)
    px = np.clip(((lons + 180) / 360 * 2 ** z - tx0) * 256, 0, W - 1).astype(int)
    py = np.clip(((1 - np.arcsinh(np.tan(np.radians(lats))) / math.pi) / 2 * 2 ** z - ty0) * 256, 0, H - 1).astype(int)
    out = land[np.ix_(py, px)]
    os.makedirs(cache_dir, exist_ok=True)
    np.save(cache, out)
    return out


def interpolate(lon, lat, val, gx, gy, land, reach):
    """海域の代表値から格子へ逆距離加重で補間。陸地と、観測点から reach 度より遠い所は NaN。"""
    ok = ~np.isnan(val)
    d2 = (gx[..., None] - lon[ok]) ** 2 + ((gy[..., None] - lat[ok]) * 1.25) ** 2 + 1e-4
    w = 1 / d2 ** 1.5
    z = (w * val[ok]).sum(-1) / w.sum(-1)
    z[np.sqrt(d2.min(-1)) > reach] = np.nan
    if land is not None:
        z[land] = np.nan
    return z


class MapDrawer:
    def __init__(self, names, args, values):
        self.args = args
        by_name = {v[0]: v for v in AREAS.values()}
        self.names = [n for n in names if n in by_name]
        self.lon = np.array([by_name[n][2] for n in self.names])
        self.lat = np.array([by_name[n][1] for n in self.names])
        lon0, lon1, lat0, lat1 = EXTENT
        self.land = load_land()
        ny, nx = (self.land.shape if self.land is not None else (600, 800))
        self.gx, self.gy = np.meshgrid(np.linspace(lon0, lon1, nx), np.linspace(lat1, lat0, ny))
        self.fig, self.ax = plt.subplots(figsize=(8, 9))
        self.fig.subplots_adjust(left=.08, right=.92, top=.94, bottom=.06)
        lo = math.floor(np.nanmin(values) / args.interval_deg) * args.interval_deg
        hi = math.ceil(np.nanmax(values) / args.interval_deg) * args.interval_deg
        self.levels = np.arange(lo - args.interval_deg, hi + args.interval_deg * 1.5, args.interval_deg)
        self.cbar = None
        self.arts = []

    def draw(self, row, title):
        ax = self.ax
        for a in self.arts:
            if hasattr(a, "remove"):
                a.remove()
            else:  # 古いmatplotlibのContourSet
                for c in a.collections:
                    c.remove()
        self.arts = []
        ax.set(xlim=EXTENT[:2], ylim=EXTENT[2:], aspect=1 / np.cos(np.radians(39.5)),
               xlabel="経度", ylabel="緯度")
        z = interpolate(self.lon, self.lat, row, self.gx, self.gy, self.land, self.args.reach)
        z = np.ma.masked_invalid(z)
        if self.land is not None:
            self.arts.append(ax.imshow(np.where(self.land, 0.93, np.nan), extent=EXTENT, origin="upper",
                                       cmap="gray", vmin=0, vmax=1, aspect="auto", zorder=0.5))
        cf = ax.contourf(self.gx, self.gy, z, levels=self.levels, cmap="turbo", zorder=1)
        cl = ax.contour(self.gx, self.gy, z, levels=self.levels, colors="k", linewidths=0.5, zorder=2)
        ax.clabel(cl, fmt="%g", fontsize=8)  # ラベルは cl を消すと一緒に消える
        self.arts += [cf, cl]
        if self.land is not None:
            self.arts.append(ax.contour(self.gx, self.gy, self.land.astype(float), levels=[.5],
                                        colors="#444", linewidths=.8, zorder=3))
        if not hasattr(self, "pts"):
            self.pts = ax.scatter(self.lon, self.lat, s=14, c="w", edgecolors="k", zorder=4)
            self.txt = [ax.text(x, y + .06, "", ha="center", fontsize=8, weight="bold", zorder=5,
                                bbox=dict(fc="w", ec="none", alpha=.7, pad=.5))
                        for x, y in zip(self.lon, self.lat)]
            self.cbar = self.fig.colorbar(cf, ax=ax, shrink=.6, label="海面水温 ℃")
        for t, v in zip(self.txt, row):
            t.set_text("" if np.isnan(v) else f"{v:.1f}")
        ax.set_title(title)


def cmd_map(df, args):
    setup_font()
    day = pd.Timestamp(args.date)
    if day not in df.index:
        raise SystemExit(f"{args.date} のデータがありません ({df.index.min():%Y-%m-%d}〜{df.index.max():%Y-%m-%d})")
    m = MapDrawer(df.columns, args, df.loc[day].values.astype(float))
    m.draw(df.loc[day][m.names].values.astype(float), f"海面水温 {day:%Y-%m-%d}")
    out = args.save or os.path.join(OUT, f"map_{day:%Y%m%d}.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    m.fig.savefig(out, dpi=130)
    print("保存:", out)
    if not args.save_only:
        plt.show()


def cmd_player(df, args):
    from matplotlib.animation import FuncAnimation
    setup_font()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end) if args.end else start + pd.offsets.YearEnd(0)
    sub = df.loc[start:end].iloc[::args.step]
    if sub.empty:
        raise SystemExit("指定期間にデータがありません")
    m = MapDrawer(df.columns, args, sub.values.astype(float))

    def frame(i):
        m.draw(sub.iloc[i][m.names].values.astype(float), f"海面水温 {sub.index[i]:%Y-%m-%d}")

    ani = FuncAnimation(m.fig, frame, frames=len(sub), interval=args.interval, repeat=True)
    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        ani.save(args.save, writer="pillow", fps=max(1, 1000 // args.interval))
        print("保存:", args.save)
    else:
        plt.show()


class Viewer:
    """年・月・日を Figure 上のボタン / スライダー / キーで操作する地図ビューア。
    データは日別なので「時間」の操作はありません。"""

    def __init__(self, df, args):
        self.df = df.dropna(how="all")
        self.avail = self.df.index
        self.m = MapDrawer(df.columns, args, self.df.values.astype(float))
        self.fig = self.m.fig
        self.fig.subplots_adjust(bottom=0.22)
        self.timer = None
        self._lock = False
        from matplotlib.widgets import Button, Slider
        specs = [("年 −", lambda: self.move_year(-1)), ("年 +", lambda: self.move_year(1)),
                 ("月 −", lambda: self.move_month(-1)), ("月 +", lambda: self.move_month(1)),
                 ("日 −", lambda: self.move_day(-1)), ("日 +", lambda: self.move_day(1)),
                 ("再生/停止", self.toggle_play)]
        self.buttons = []
        w, gap, x0 = 0.11, 0.015, 0.08
        for i, (text, fn) in enumerate(specs):
            b = Button(self.fig.add_axes([x0 + i * (w + gap), 0.095, w, 0.045]), text)
            b.on_clicked(lambda _e, fn=fn: fn())
            self.buttons.append(b)
        self.slider = Slider(self.fig.add_axes([0.08, 0.035, 0.74, 0.03]), "", 0, len(self.avail) - 1,
                             valinit=len(self.avail) - 1, valstep=1)
        self.slider.on_changed(lambda v: None if self._lock else self.show(int(v)))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.pos = len(self.avail) - 1
        self.show(self.pos)

    def show(self, pos):
        self.pos = int(min(max(pos, 0), len(self.avail) - 1))
        d = self.avail[self.pos]
        self.m.draw(self.df.iloc[self.pos][self.m.names].values.astype(float), f"海面水温 {d:%Y年%m月%d日}")
        self._lock = True
        self.slider.set_val(self.pos)
        self.slider.valtext.set_text(f"{d:%Y-%m-%d}")
        self._lock = False
        self.fig.canvas.draw_idle()

    def nearest(self, target):
        i = int(self.avail.searchsorted(target))
        cand = [j for j in (i - 1, i) if 0 <= j < len(self.avail)]
        return min(cand, key=lambda j: abs(self.avail[j] - target))

    def move_day(self, n):
        self.show(self.pos + n)

    def move_month(self, n):
        self.show(self.nearest(self.avail[self.pos] + pd.DateOffset(months=n)))

    def move_year(self, n):
        self.show(self.nearest(self.avail[self.pos] + pd.DateOffset(years=n)))

    def toggle_play(self):
        if self.timer is None:
            self.timer = self.fig.canvas.new_timer(interval=150)
            self.timer.add_callback(self.tick)
            self.timer.start()
        else:
            self.timer.stop()
            self.timer = None

    def tick(self):
        d = self.avail[self.pos]
        nxt = self.pos + 1
        if nxt >= len(self.avail) or self.avail[nxt].year != d.year:  # その年の夏の終わり -> 同じ年の頭に戻る
            nxt = self.nearest(pd.Timestamp(d.year, 6, 1))
        self.show(nxt)

    def on_key(self, e):
        {"right": lambda: self.move_day(1), "left": lambda: self.move_day(-1),
         "up": lambda: self.move_year(1), "down": lambda: self.move_year(-1),
         "pageup": lambda: self.move_month(1), "pagedown": lambda: self.move_month(-1),
         " ": self.toggle_play}.get(e.key, lambda: None)()


def cmd_viewer(df, args):
    setup_font()
    for k in ("keymap.back", "keymap.forward", "keymap.pan", "keymap.zoom", "keymap.save"):
        plt.rcParams[k] = []  # 矢印キー等をこのビューアの操作に使うため
    if not args.summer and not args.all_season:
        df = summer(df)
    v = Viewer(df, args)
    if args.date:
        v.show(v.nearest(pd.Timestamp(args.date)))
    plt.show()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=HERE, help="*.txt のあるフォルダ")
    p.add_argument("--summer", action="store_true", help="毎年の6/1〜8/31だけを使う (全コマンド共通。viewerは既定で夏季のみ)")
    sp = p.add_subparsers(dest="cmd", required=True)
    sp.add_parser("summary")
    pp = sp.add_parser("plot")
    pp.add_argument("--area", help="海域名 (例: 宮城県沿岸)。省略で全海域")
    mp = sp.add_parser("map", help="ある日の海水温を地図+等温線で表示")
    mp.add_argument("--date", default="2025-08-15")
    mp.add_argument("--save", help="PNG保存先")
    mp.add_argument("--save-only", action="store_true", help="画面表示せず保存だけ")
    mp.add_argument("--reach", type=float, default=1.5, help="観測点からこの度数以上遠い所は塗らない")
    mp.add_argument("--interval-deg", type=float, default=0.5, help="等温線の間隔(℃)")
    vw = sp.add_parser("viewer", help="年・月・日をFigure上のボタンで操作 (既定は夏季のみ)")
    vw.add_argument("--date", help="最初に表示する日 (省略で最新)")
    vw.add_argument("--all-season", action="store_true", help="夏季に限らず全期間を対象にする")
    vw.add_argument("--reach", type=float, default=1.5)
    vw.add_argument("--interval-deg", type=float, default=1.0, help="等温線の間隔(℃)")
    pl = sp.add_parser("player", help="期間を再生 (地図+等温線)")
    pl.add_argument("--reach", type=float, default=1.5)
    pl.add_argument("--interval-deg", type=float, default=1.0, help="等温線の間隔(℃)")
    pl.add_argument("--start", default="2020-01-01")
    pl.add_argument("--end")
    pl.add_argument("--step", type=int, default=1, help="何日ごとに再生するか")
    pl.add_argument("--interval", type=int, default=100, help="1コマのms")
    pl.add_argument("--save", help="GIF保存先 (指定時は画面表示しない)")
    args = p.parse_args()
    df = load(args.dir)
    if args.summer:
        df = summer(df)
    {"viewer": cmd_viewer, "summary": cmd_summary, "plot": cmd_plot, "map": cmd_map, "player": cmd_player}[args.cmd](df, args)


if __name__ == "__main__":
    main()
