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
import matplotlib.cm
import matplotlib.colors
import matplotlib.patches

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


# 東北の官署 (概算の緯度経度。位置がずれていたら直してください)
STATIONS = {
    "むつ": (41.28, 141.22), "青森": (40.82, 140.77), "深浦": (40.65, 139.93), "八戸": (40.53, 141.52),
    "秋田": (39.72, 140.10), "盛岡": (39.70, 141.17), "宮古": (39.65, 141.97), "大船渡": (39.07, 141.72),
    "石巻": (38.43, 141.30), "仙台": (38.26, 140.90), "酒田": (38.91, 139.84), "新庄": (38.76, 140.31),
    "山形": (38.25, 140.35), "福島": (37.75, 140.47), "若松": (37.49, 139.92), "白河": (37.13, 140.22),
    "小名浜": (36.95, 140.90),
}
COMPASS = ["北", "北北東", "北東", "東北東", "東", "東南東", "南東", "南南東",
           "南", "南南西", "南西", "西南西", "西", "西北西", "北西", "北北西"]
WIND_DEG = {n: i * 22.5 for i, n in enumerate(COMPASS)}
OBS_DIR = os.path.join(HERE, "東北地方官署アメダスデータ")
# 気象庁「時別値」CSV: 地点ごとに14列 = 気温(値,品質,均質) 風速(値,品質) 風向(値,品質,均質) 湿度(値,品質,均質) 視程(値,品質,均質)
OBS_COLS = {"temp": 0, "wind": 3, "dir": 5, "rh": 8, "vis": 11}


def _read_rows(path):
    import csv
    for enc in ("utf-8-sig", "cp932"):
        try:
            with open(path, encoding=enc, newline="") as f:
                return list(csv.reader(f))
        except UnicodeDecodeError:
            continue
    raise SystemExit("文字コードが判別できません: " + path)


def load_obs(obs_dir=OBS_DIR):
    """官署の時別値CSV(年ごと)を全部読み、{'temp','wind','dir','rh','vis': 時刻×地点のDataFrame} を返す。
    風向は度(北=0,東=90)。静穏は風向NaNかつ風速が数値。無ければ None。"""
    files = sorted(glob.glob(os.path.join(obs_dir, "*.csv")))
    if not files:
        return None
    parts = {k: [] for k in OBS_COLS}
    for path in files:
        rows = _read_rows(path)
        h = next((i for i, r in enumerate(rows) if r and r[0].startswith("年月日時")), None)
        if h is None:
            print("  読み飛ばし(形式が違う):", os.path.basename(path))
            continue
        names = rows[h - 1][1::14]
        data = [r for r in rows[h + 3:] if r and r[0]]
        t = pd.to_datetime([r[0] for r in data], format="%Y/%m/%d %H:%M:%S", errors="coerce")
        keep = ~t.isna()
        data = [r for r, k in zip(data, keep) if k]
        t = t[keep]
        for key, off in OBS_COLS.items():
            cols = {}
            for si, name in enumerate(names):
                if name not in STATIONS:
                    continue
                vals = [r[1 + 14 * si + off] if len(r) > 1 + 14 * si + off else "" for r in data]
                if key == "dir":
                    cols[name] = [WIND_DEG.get(v, np.nan) for v in vals]
                else:
                    cols[name] = pd.to_numeric(pd.Series(vals), errors="coerce").values
            parts[key].append(pd.DataFrame(cols, index=t))
    obs = {k: pd.concat(v).sort_index() for k, v in parts.items() if v}
    obs = {k: v[~v.index.duplicated()] for k, v in obs.items()}
    return obs


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
        gx, gy = np.meshgrid(np.linspace(lon0, lon1, nx), np.linspace(lat1, lat0, ny))
        # 等温線は粗い格子(K倍間引き)で計算して軽くする。陸地は最後に本来の精度で上から被せるので海岸線は粗くならない。
        K = 4
        cy, cx = ny // K, nx // K
        self.gx = gx[:cy * K, :cx * K].reshape(cy, K, cx, K).mean(axis=(1, 3))
        self.gy = gy[:cy * K, :cx * K].reshape(cy, K, cx, K).mean(axis=(1, 3))
        self.cland = (self.land[:cy * K, :cx * K].reshape(cy, K, cx, K).all(axis=(1, 3))
                      if self.land is not None else None)  # ブロックが全部陸地の所だけ陸扱い
        self.fig, self.ax = plt.subplots(figsize=(8, 9))
        self.fig.subplots_adjust(left=.08, right=.92, top=.94, bottom=.06)
        lo = math.floor(np.nanmin(values) / args.interval_deg) * args.interval_deg
        hi = math.ceil(np.nanmax(values) / args.interval_deg) * args.interval_deg
        self.levels = np.arange(lo - args.interval_deg, hi + args.interval_deg * 1.5, args.interval_deg)
        self.cmap = plt.get_cmap("turbo").copy()
        self.cmap.set_bad("w")
        self.norm = matplotlib.colors.BoundaryNorm(self.levels, self.cmap.N)
        self.flags = {"temp": True, "wind": True, "rh": True, "vis": True}  # 官署の表示項目
        self.st_names = [n for n in STATIONS]
        self.st_lat = np.array([STATIONS[n][0] for n in self.st_names])
        self.st_lon = np.array([STATIONS[n][1] for n in self.st_names])
        self.cbar = None
        self.sst_arts, self.arts = [], []
        self._sst_key = None
        self._static()

    def _static(self):
        """変わらない部分(陸地・海岸線)を1回だけ描く。"""
        ax = self.ax
        if self.land is not None:
            ax.imshow(np.where(self.land, 0.93, np.nan), extent=EXTENT, origin="upper",
                      cmap="gray", vmin=0, vmax=1, aspect="auto", zorder=2.5)
            lon0, lon1, lat0, lat1 = EXTENT
            ny, nx = self.land.shape
            ax.contour(np.linspace(lon0, lon1, nx), np.linspace(lat1, lat0, ny), self.land.astype(float),
                       levels=[.5], colors="#444", linewidths=.8, zorder=3)
        # imshow が aspect を 'auto' に変えてしまうので、その後に固定する (全画面でも縮尺が崩れない)
        ax.set(xlim=EXTENT[:2], ylim=EXTENT[2:], xlabel="経度", ylabel="緯度")
        ax.set_aspect(1 / np.cos(np.radians(39.5)), adjustable="box")

    @staticmethod
    def _remove(arts):
        for a in arts:
            if hasattr(a, "remove"):
                a.remove()
            else:  # 古いmatplotlibのContourSet
                for c in a.collections:
                    c.remove()
        arts.clear()

    def draw(self, row, title, obs=None, key=None):
        """row: 海域ごとの海面水温。obs: 官署の観測 {'temp','wind','dir','rh','vis': 官署順の配列} (省略可)
        key: 海面水温が同じ(=同じ日)なら等温線を再計算せず使い回すための識別子"""
        ax = self.ax
        self._remove(self.arts)
        if key is None or key != self._sst_key:
            self._remove(self.sst_arts)
            z = interpolate(self.lon, self.lat, row, self.gx, self.gy, self.cland, self.args.reach)
            z = np.ma.masked_invalid(z)
            cf = ax.contourf(self.gx, self.gy, z, levels=self.levels, cmap=self.cmap, norm=self.norm, zorder=1)
            cl = ax.contour(self.gx, self.gy, z, levels=self.levels, colors="k", linewidths=0.5, zorder=2)
            ax.clabel(cl, fmt="%g", fontsize=8)  # ラベルは cl を消すと一緒に消える
            self.sst_arts += [cf, cl]
            self._sst_key = key
            if not hasattr(self, "pts"):
                self.pts = ax.scatter(self.lon, self.lat, s=14, c="w", edgecolors="k", zorder=4)
                self.txt = [ax.text(x, y + .06, "", ha="center", fontsize=8, weight="bold", zorder=5,
                                    bbox=dict(fc="w", ec="none", alpha=.7, pad=.5))
                            for x, y in zip(self.lon, self.lat)]
                sm = matplotlib.cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
                self.cbar = self.fig.colorbar(sm, ax=ax, shrink=.6,
                                              label="海面水温(丸) / 気温(四角) ℃" if obs else "海面水温 ℃")
                if obs:
                    ax.text(.01, .01, "□気温(色)  →風の行き先  小さい○静穏  大きい円=視程(半径km)  紫=視程1km未満",
                            transform=ax.transAxes, fontsize=7, zorder=9,
                            bbox=dict(fc="w", ec="none", alpha=.75, pad=1.5))
            for t, v in zip(self.txt, row):
                t.set_text("" if np.isnan(v) else f"{v:.1f}")
        if obs is not None:
            self._draw_obs(obs)
        ax.set_title(title)

    def _draw_obs(self, obs):
        ax, f = self.ax, self.flags
        lon, lat = self.st_lon, self.st_lat
        t, w, d, rh, vis = (np.asarray(obs[k], float) for k in ("temp", "wind", "dir", "rh", "vis"))
        fog = f["vis"] & (vis < 1.0)  # 視程1km未満 = 低視程 (霧の目安だが、霧とは限らない)
        self.arts.append(ax.scatter(lon, lat, s=90, marker="s", c=np.ma.masked_invalid(t) if f["temp"] else "w",
                                    cmap=self.cmap, norm=self.norm, edgecolors=np.where(fog, "m", "k"),
                                    linewidths=np.where(fog, 2.5, 0.8), zorder=6))
        if f["wind"]:
            ok = ~np.isnan(w) & ~np.isnan(d)
            calm = ~np.isnan(w) & np.isnan(d)
            if ok.any():
                rad = np.radians(d[ok])  # 風向は風の吹いてくる向き -> 矢印は吹いていく向き
                self.arts.append(ax.quiver(lon[ok], lat[ok], -w[ok] * np.sin(rad), -w[ok] * np.cos(rad),
                                           angles="uv", scale=15, scale_units="inches", width=.006,
                                           color="k", zorder=7))
            if calm.any():
                self.arts.append(ax.scatter(lon[calm], lat[calm], s=40, facecolors="none", edgecolors="k",
                                            linewidths=1.2, zorder=7))
        if f["vis"]:  # 視程 = 観測点を中心とした半径(km)の円 (縦横とも実距離)
            for i in np.where(~np.isnan(vis))[0]:
                r = vis[i]
                self.arts.append(matplotlib.patches.Ellipse(
                    (lon[i], lat[i]), 2 * r / (111.0 * math.cos(math.radians(39.5))), 2 * r / 111.0,
                    fc=(1, 0, 1, .12) if fog[i] else (.2, .2, .2, .08),
                    ec="m" if fog[i] else "#555", lw=1.2, zorder=5))
                ax.add_patch(self.arts[-1])
        for i, name in enumerate(self.st_names):
            parts = []
            if f["temp"] and not np.isnan(t[i]):
                parts.append(f"{t[i]:.1f}℃")
            if f["rh"] and not np.isnan(rh[i]):
                parts.append(f"{rh[i]:.0f}%")
            if f["vis"] and not np.isnan(vis[i]):
                parts.append(f"{vis[i]:.1f}km")
            self.arts.append(ax.text(lon[i] + .07, lat[i] - .08, "\n".join([name] + parts),
                                     fontsize=7, va="top", zorder=8, color="#c00000" if fog[i] else "k",
                                     bbox=dict(fc="w", ec="none", alpha=.65, pad=.4)))


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


# ---- 低視程(視程<1km)の発生条件の解析: 沿岸官署の 気温・相対湿度・視程 と 近くの海域の海面水温 ----
COAST_DIR = os.path.join(HERE, "東北地方沿岸気象官署時別値")  # 気象庁の時別値(1時間ごと)
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
RH_BINS = [-np.inf] + list(range(90, 102))  # 湿度90%未満は1階級、90%以上は1%ごと (視程<1kmは90%未満ではほとんど無い)


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


def fog_conditions(v, rh_min=(90, 95, 97, 99, 100), dt_max=(1, 2, 3, 5, np.inf)):
    """「相対湿度≧A かつ |気温−海面水温|≦B」という条件ごとに、
    発生率(その条件の時に低視程だった割合%)と 捕捉率(低視程のうち、その条件に入っていた割合%)を返す。"""
    fog = v["fog"] == 1
    rate, cover, num = {}, {}, {}
    for a in rh_min:
        for b in dt_max:
            m = (v["RH"] >= a) & (v["dT"].abs() <= b)
            key = (f"RH≧{a}", f"|dT|≦{b:g}" if np.isfinite(b) else "dT問わず")
            rate[key] = fog[m].mean() * 100 if m.any() else np.nan
            cover[key] = m[fog].mean() * 100 if fog.any() else np.nan
            num[key] = int(m.sum())
    cols = [f"|dT|≦{b:g}" if np.isfinite(b) else "dT問わず" for b in dt_max]
    f = lambda d: pd.Series(d).unstack().reindex(index=[f"RH≧{a}" for a in rh_min], columns=cols)
    return f(rate), f(cover), f(num)


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


def cmd_fog(df, args):
    setup_font()
    d, v = fog_valid(df, args)
    pd.set_option("display.width", 220, "display.float_format", "{:.1f}".format)
    print(f"官署 {d['station'].nunique()}地点, {d['time'].min():%Y-%m-%d}〜{d['time'].max():%Y-%m-%d}")
    print(f"解析に使える時刻(視程・湿度・気温・海面水温がそろう): {len(v)}  うち低視程(視程<{FOG_KM:g}km): {int(v['fog'].sum())} ({v['fog'].mean() * 100:.2f}%)")
    print("\n[官署別] 解析数 / 低視程の数 / 割合% / 低視程の時の 気温-海水温(中央値) 湿度(中央値)")
    g = v.groupby("station")
    st = pd.DataFrame({"n": g.size(), "fog_n": g["fog"].sum().astype(int), "fog_%": g["fog"].mean() * 100,
                       "低視程時dT中央": v[v.fog == 1].groupby("station")["dT"].median(),
                       "低視程時RH中央": v[v.fog == 1].groupby("station")["RH"].median()})
    print(st.to_string())
    f, nf = v[v.fog == 1], v[v.fog == 0]
    qs = [.05, .1, .25, .5, .75, .9, .95]
    summ = pd.DataFrame({"低視程 dT(℃)": f["dT"].quantile(qs), "低視程 RH(%)": f["RH"].quantile(qs),
                         "視程1km以上 dT(℃)": nf["dT"].quantile(qs), "視程1km以上 RH(%)": nf["RH"].quantile(qs)})
    summ.index = [f"{int(q * 100)}%点" for q in qs]
    print("\n[分布] 低視程の時とそうでない時の 気温-海面水温(dT) と 相対湿度 (パーセント点)")
    print(summ.to_string())
    p, n, k = fog_table(v)
    print(f"\n[低視程の出現率%] 縦=相対湿度, 横=気温-海面水温(℃)  (観測数30未満は空欄)")
    print(p.to_string(na_rep="-"))
    rate, cover, num = fog_conditions(v)
    print("\n[条件別の低視程の出現率%] 行=相対湿度の下限, 列=|気温-海面水温|の上限 (その条件の時に低視程だった割合)")
    print(rate.to_string())
    print("[条件別の捕捉率%] (低視程だった時のうち、その条件に入っていた割合)")
    print(cover.to_string())
    os.makedirs(OUT, exist_ok=True)
    rate.to_csv(os.path.join(OUT, "fog_condition_rate.csv"), encoding="utf-8-sig")
    cover.to_csv(os.path.join(OUT, "fog_condition_cover.csv"), encoding="utf-8-sig")
    st.to_csv(os.path.join(OUT, "fog_by_station.csv"), encoding="utf-8-sig")
    summ.to_csv(os.path.join(OUT, "fog_distribution.csv"), encoding="utf-8-sig")
    p.to_csv(os.path.join(OUT, "fog_probability_dT_RH.csv"), encoding="utf-8-sig")
    n.to_csv(os.path.join(OUT, "fog_count_dT_RH.csv"), encoding="utf-8-sig")
    k.to_csv(os.path.join(OUT, "fog_fogcount_dT_RH.csv"), encoding="utf-8-sig")

    # --- 図 ---
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, bins, xl in ((axs[0], "dT", np.arange(-10, 10.1, 0.5), "気温 − 海面水温 (℃)"),
                              (axs[1], "RH", np.arange(40, 100.1, 1), "相対湿度 (%)")):
        ax.hist(nf[col].clip(bins[0], bins[-1]), bins, density=True, alpha=.5, color="#888", label=f"視程1km以上 (n={len(nf)})")
        ax.hist(f[col].clip(bins[0], bins[-1]), bins, density=True, alpha=.7, color="#d02090", label=f"低視程 視程<{FOG_KM:g}km (n={len(f)})")
        ax.set(xlabel=xl, ylabel="割合(密度)", title=f"{xl} の分布")
        ax.legend()
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fog_distribution.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    im = fog_heat(ax, p, n, k, "低視程(視程<1km)の出現率 % : 相対湿度 × (気温−海面水温)  [全官署]")
    fig.colorbar(im, ax=ax, label="低視程の出現率 %")
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.text(.01, .005, FOG_NOTE, fontsize=8)
    fig.savefig(os.path.join(OUT, "fog_probability.png"), dpi=120)
    plt.close(fig)

    fig = fog_station_figure(v)
    fig.savefig(os.path.join(OUT, "fog_probability_by_station.png"), dpi=110)
    plt.close(fig)
    print("\n保存先:", OUT, "(fog_*.png / fog_*.csv)")


class Viewer:
    """年・月・日・時を Figure 上のボタン / スライダー / キーで操作する地図ビューア。
    海面水温は日別、官署(気温・風・湿度・視程)は時別。obs が無ければ日別の海面水温だけ。"""

    def __init__(self, df, args, obs=None):
        self.sst = df
        self.obs = obs
        if obs:
            has = pd.concat([obs[k].notna().any(axis=1) for k in ("temp", "wind", "rh", "vis")], axis=1).any(axis=1)
            self.times = obs["temp"].index[has.reindex(obs["temp"].index).fillna(False).values]
            vals = np.concatenate([df.values.ravel(), obs["temp"].values.ravel()]).astype(float)
        else:
            self.times = df.dropna(how="all").index
            vals = df.values.astype(float)
        self.m = MapDrawer(df.columns, args, vals)
        if obs:  # 表示する時刻ぶんだけ、官署順のnumpy配列にしておく
            self.arr = {k: obs[k].reindex(index=self.times, columns=self.m.st_names).values.astype(float)
                        for k in obs}
        self.fig = self.m.fig
        self.fig.set_size_inches(8.4, 10.2)
        self.fig.subplots_adjust(bottom=0.27)
        self.timer = None
        self._lock = False
        from matplotlib.widgets import Button, Slider
        nav = [("年 −", lambda: self.move_year(-1)), ("月 −", lambda: self.move_month(-1)),
               ("日 −", lambda: self.move_day(-1)), ("時 −", lambda: self.move_hour(-1)),
               ("時 +", lambda: self.move_hour(1)), ("日 +", lambda: self.move_day(1)),
               ("月 +", lambda: self.move_month(1)), ("年 +", lambda: self.move_year(1)),
               ("再生/停止", self.toggle_play)]
        self.buttons = []
        w, gap, x0 = 0.09, 0.012, 0.06
        for i, (text, fn) in enumerate(nav):
            b = Button(self.fig.add_axes([x0 + i * (w + gap), 0.165, w, 0.04]), text)
            b.on_clicked(lambda _e, fn=fn: fn())
            self.buttons.append(b)
        self.toggles = {}
        if obs:
            for i, (key, text) in enumerate([("temp", "気温"), ("wind", "風"), ("rh", "湿度"), ("vis", "視程")]):
                b = Button(self.fig.add_axes([x0 + i * (w + gap), 0.11, w, 0.04]), text)
                b.on_clicked(lambda _e, k=key: self.toggle(k))
                self.toggles[key] = b
                self.buttons.append(b)
            self._paint_toggles()
        self.slider = Slider(self.fig.add_axes([0.08, 0.05, 0.70, 0.03]), "", 0, len(self.times) - 1,
                             valinit=0, valstep=1)
        self.slider.on_changed(lambda v: None if self._lock else self.show(int(v)))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.pos = 0
        self.show(self.pos)

    @staticmethod
    def label(ts, hourly):
        """気象庁の時別値は「その時刻までの1時間」なので 0時は前日の24時と表示する。"""
        if not hourly:
            return ts.normalize(), f"{ts:%Y年%m月%d日}"
        d, h = (ts - pd.Timedelta(days=1), 24) if ts.hour == 0 else (ts, ts.hour)
        return d.normalize(), f"{d:%Y年%m月%d日} {h}時"

    def _paint_toggles(self):
        for k, b in self.toggles.items():
            on = self.m.flags[k]
            b.ax.set_facecolor("#9fd6a0" if on else "#dddddd")
            b.hovercolor = "#7cc47d" if on else "#cccccc"
            b.color = "#9fd6a0" if on else "#dddddd"

    def toggle(self, key):
        self.m.flags[key] = not self.m.flags[key]
        self._paint_toggles()
        self.show(self.pos)

    def show(self, pos):
        self.pos = int(min(max(pos, 0), len(self.times) - 1))
        ts = self.times[self.pos]
        day, text = self.label(ts, self.obs is not None)
        row = (self.sst.loc[day] if day in self.sst.index else pd.Series(np.nan, index=self.sst.columns))
        o = None
        if self.obs:
            o = {k: a[self.pos] for k, a in self.arr.items()}
        self.m.draw(row[self.m.names].values.astype(float), "海面水温(日別)" + (" + 官署(時別) " if o else " ") + text,
                    o, key=day)
        self._lock = True
        self.slider.set_val(self.pos)
        self.slider.valtext.set_text(f"{ts:%Y-%m-%d %H:%M}")
        self._lock = False
        self.fig.canvas.draw_idle()

    def nearest(self, target):
        i = int(self.times.searchsorted(target))
        cand = [j for j in (i - 1, i) if 0 <= j < len(self.times)]
        return min(cand, key=lambda j: abs(self.times[j] - target))

    def move_hour(self, n):
        new = min(max(self.pos + n, 0), len(self.times) - 1)
        if self.times[new].year == self.times[self.pos].year:  # 別の年へは飛ばない
            self.show(new)

    def move_day(self, n):
        self.show(self.nearest(self.times[self.pos] + pd.DateOffset(days=n)))

    def move_month(self, n):
        self.show(self.nearest(self.times[self.pos] + pd.DateOffset(months=n)))

    def move_year(self, n):
        target = self.times[self.pos] + pd.DateOffset(years=n)
        new = self.nearest(target)
        if self.times[new].year == target.year:  # データの無い年へは動かない
            self.show(new)

    def toggle_play(self):
        if self.timer is None:
            self.timer = self.fig.canvas.new_timer(interval=150)
            self.timer.add_callback(self.tick)
            self.timer.start()
        else:
            self.timer.stop()
            self.timer = None

    def tick(self):
        cur = self.times[self.pos]
        nxt = self.pos + 1
        if nxt >= len(self.times) or self.times[nxt].year != cur.year:  # その年の夏の終わり -> 同じ年の頭に戻る
            nxt = self.nearest(pd.Timestamp(cur.year, 6, 1))
        self.show(nxt)

    def on_key(self, e):
        {"right": lambda: self.move_hour(1), "left": lambda: self.move_hour(-1),
         "up": lambda: self.move_day(1), "down": lambda: self.move_day(-1),
         "pageup": lambda: self.move_month(1), "pagedown": lambda: self.move_month(-1),
         "]": lambda: self.move_year(1), "[": lambda: self.move_year(-1),
         "1": lambda: self.toggle("temp"), "2": lambda: self.toggle("wind"),
         "3": lambda: self.toggle("rh"), "4": lambda: self.toggle("vis"),
         " ": self.toggle_play}.get(e.key, lambda: None)()


def cmd_viewer(df, args):
    setup_font()
    for k in ("keymap.back", "keymap.forward", "keymap.pan", "keymap.zoom", "keymap.save"):
        plt.rcParams[k] = []  # 矢印キー等をこのビューアの操作に使うため
    if not args.summer and not args.all_season:
        df = summer(df)
    obs = None
    if not args.no_obs:
        obs = load_obs(args.obs_dir)
        if obs is None:
            print(f"官署データが見つかりません ({args.obs_dir})。海面水温だけで表示します。--obs-dir で指定できます。")
        else:
            print(f"官署データ: {obs['temp'].index.min():%Y-%m-%d} 〜 {obs['temp'].index.max():%Y-%m-%d}, "
                  f"{obs['temp'].shape[1]}地点")
    v = Viewer(df, args, obs)
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
    vw.add_argument("--obs-dir", default=OBS_DIR, help="官署の時別値CSVのフォルダ")
    vw.add_argument("--no-obs", action="store_true", help="官署データを使わない")
    vw.add_argument("--all-season", action="store_true", help="夏季に限らず全期間を対象にする")
    vw.add_argument("--reach", type=float, default=1.5)
    vw.add_argument("--interval-deg", type=float, default=1.0, help="等温線の間隔(℃)")
    fg = sp.add_parser("fog", help="沿岸官署の 気温-海水温・相対湿度 と 低視程(視程<1km) の関係を解析")
    fg.add_argument("--coast-dir", default=COAST_DIR, help="沿岸官署の時別値CSVのフォルダ")
    fg.add_argument("--from-year", type=int, help="この年以降だけ使う (視程は2020年から毎時で、それ以前は間引き)")
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
    {"fog": cmd_fog, "viewer": cmd_viewer, "summary": cmd_summary, "plot": cmd_plot, "map": cmd_map, "player": cmd_player}[args.cmd](df, args)


if __name__ == "__main__":
    main()
