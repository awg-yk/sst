"""霧日数(棒)に、条件を満たす日数や、気温・海面水温・露点温度の年平均(折れ線)を重ねて、官署ごとに表示する。
fog_plot.py と同じフォルダに置いて実行します (fog_plot.py の読み込み処理を使います)。
霧日数は 東北地方気象官署月別値霧日数.csv (気象庁の月別値 「霧日数」。--kiri で場所を指定。既定は同じフォルダ) の6〜8月の合計。
露点温度は「東北地方沿岸気象官署時別値」フォルダの時別値CSV (1984年〜) にあります。

使い方: python fog_days.py 〔モード〕 〔オプション〕   (モードを省略すると rh)

    python fog_days.py rh                          # 相対湿度 と 気温−海面水温 の条件の日数 (黒線) + 先生の条件 (灰色の破線)
    python fog_days.py rh --rh-thr 98 --dt-lo -1 --dt-hi 1 --min-hours 2   # 条件を変える
    python fog_days.py cond                        # 海面水温−露点温度 ≦0℃ の時刻がある日の日数 (黒線)
    python fog_days.py cond --sat-max 2            # さらに 気温−露点温度 ≦2℃ (空気が飽和に近い) を加える
    python fog_days.py cond --sat-max 2 --dt-min -2 --dt-max 1   # さらに 気温−海面水温 が −2〜1℃ を加える
    python fog_days.py temps                       # 気温・海面水温・露点温度の年平均 (折れ線、右の軸)
    python fog_days.py gap                         # 海面水温−露点温度の年平均 (折れ線、右の軸。上ほど小さい向き)
    python fog_days.py chart                       # 官署ごとの大きな画像 (5年移動平均・回帰直線・傾き・欠測つき)。条件は湿度≧95% かつ |気温−海面水温|≦0℃
    python fog_days.py chart --rh-min 90 --dt-abs 1                     # 条件を変える
    python fog_days.py chart --cond td --gap-max 0 --sat-max 2          # 露点温度の条件 (cond と同じ条件)
    python fog_days.py chart --cond none --stations 八戸 宮古            # 霧日数だけ (条件の線なし)
    python fog_days.py temps --stations 八戸 宮古    # どのモードでも、官署を選べる
    python fog_days.py cond --save out/fog_days_cond.png   # PNGに保存

モード:
  rh     黒線: 6〜8月で、毎時の観測の「相対湿度 ≧ --rh-thr」かつ「--dt-lo ≦ 気温−その日の海面水温 ≦ --dt-hi」を
         満たす時刻が --min-hours 時間以上ある日の、年ごとの日数。既定: 湿度98%以上、気温−海面水温 −2℃以上(上限なし)、1時間以上。
         灰色の破線: 先生の条件 (日平均湿度≧80% かつ |日最低気温−海面水温|≦1℃) の日数 (--no-ref で消す)。
  cond   黒線: 次をすべて満たす時刻が --min-hours 時間以上ある日の、年ごとの日数。縦軸は全官署で同じ範囲。
           --gap-min ≦ 海面水温 − 露点温度 ≦ --gap-max (既定: 上限0℃、下限なし)
           気温 − 露点温度 ≦ --sat-max (既定: 使わない)
           --dt-min ≦ 気温 − 海面水温 ≦ --dt-max (既定: 使わない)
  temps  折れ線 (右の軸): 赤=気温、青=海面水温、緑=露点温度の、その年の6〜8月の平均 (3つがそろった時刻だけ)。縦軸は全官署で同じ範囲。
  gap    折れ線 (右の軸): その年の6〜8月の「海面水温 − 露点温度」の平均。上ほど差が小さい向きに反転してあり、
         棒と同じ向きに動けば「差が小さい年ほど霧日数が多い」。縦軸は全官署で同じ範囲。

  chart  Colab のノートブックと同じ図。官署ごとに1枚の画像を out/ (--outdir) に保存する (--show で画面にも表示)。
         --cond rh (既定): 毎時の相対湿度 ≧ --rh-min (既定95) かつ |気温−海面水温| ≦ --dt-abs (既定0) の時刻が1時間でもある日の日数。
         --cond td: 海面水温−露点温度 ≦ --gap-max (既定0)、必要なら --sat-max、--dt-min、--dt-max を加えた条件の日数。
         --cond none: 条件の線なし。 八戸の図には三沢 (三沢月別値霧日数.csv, --misawa) の線も重ねる。

棒: 気象庁の霧日数。均質番号 (観測環境) が変わった年に▲を付け、最新の区間を濃い緑にします。
タイトルの r: 霧日数と黒線 (または折れ線) の相関 (最新の区間 / そのうち --corr-from 年以降)。temps は相関を出しません。

注意:
  - 気温・湿度は1990年から毎時、1989年以前は3時間おき以下なので、1989年以前(薄い灰色の帯)は日数が少なめに出ます。
  - 相対湿度の記録は2013年ごろに変わっていて (縦の点線)、湿度や露点温度を使う条件・平均は、それ以前と比べられません。
    相関は --corr-from (既定2014) 以降だけのものも出します。
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fog_plot as fp

MODES = ("rh", "cond", "temps", "gap", "chart")
LINES = (("Ta", "気温", "#cc3311"), ("SST", "海面水温", "#0077bb"), ("Td", "露点温度", "#228833"))
XLABEL = "年  (灰色の帯: 観測が3時間おき以下の年 / 点線: 湿度の記録が変わった年 / ▲: 観測環境が変わった年)"


def load_kiri(path):
    """月別の霧日数CSV(cp932/utf-8) -> 官署, 年, 6〜8月合計, 均質番号"""
    rows = fp._read_rows(path)
    names = rows[2]
    out = []
    for st in dict.fromkeys(n for n in names[1:] if n):
        c = next(i for i in range(1, len(names)) if names[i] == st)
        for r in rows[5:]:
            if not r or "/" not in r[0] or len(r) <= c + 2:
                continue
            y, m = map(int, r[0].split("/"))
            val = float(r[c]) if r[c] != "" else np.nan  # 品質情報が5(資料不足値: 月の一部が欠測)でも値があれば使う
            out.append((st, y, m, val, int(r[c + 2]) if r[c + 2].isdigit() else np.nan))
    d = pd.DataFrame(out, columns=["station", "yr", "mon", "kiri", "homog"])
    d = d[d["mon"].isin([6, 7, 8])]
    return d.groupby(["station", "yr"]).agg(
        kiri=("kiri", lambda s: s.sum() if len(s) == 3 and s.notna().all() else np.nan),
        homog=("homog", "max")).reset_index()


def hourly_table(sst, coast):
    """毎時の観測に、その日の対応する海域の海面水温と、気温−海面水温(dT)を付ける (6〜8月)。"""
    c = coast[coast["station"].isin(fp.STATION_SST)].copy()
    c["day"] = (c["time"] - pd.Timedelta(hours=1)).dt.normalize()  # 0時は前日24時
    c["area"] = c["station"].map(fp.STATION_SST)
    long = sst.stack().rename("SST").rename_axis(["day", "area"]).reset_index()
    c = c.merge(long, on=["day", "area"], how="left")
    c["dT"] = c["Ta"] - c["SST"]
    c["yr"] = c["day"].dt.year
    return c[c["day"].dt.month.isin([6, 7, 8])]


def hourly_cond_days(h, rh_thr, lo, hi, min_hours):
    """毎時の条件を満たす時刻が min_hours 時間以上ある日の、(官署, 年)ごとの日数。"""
    hit = (h["RH"] >= rh_thr) & (h["dT"] >= lo) & (h["dT"] <= hi)
    n = hit.groupby([h["station"], h["day"]]).sum()
    ok = (n >= min_hours).reset_index(name="ok")
    ok["yr"] = ok["day"].dt.year
    return ok[ok["ok"]].groupby(["station", "yr"]).size()


def teacher_days(h):
    """先生の条件: 日平均湿度≧80% かつ |日最低気温−海面水温|≦1℃ の日数 (1日20時刻以上そろった日だけ)。"""
    g = h.groupby(["station", "day"])
    d = pd.DataFrame({"n_ta": g.Ta.count(), "n_rh": g.RH.count(), "rh_mean": g.RH.mean(),
                      "t_min": g.Ta.min(), "SST": g.SST.first()}).reset_index()
    d = d[(d.n_ta >= 20) & (d.n_rh >= 20) & d.SST.notna()]
    d = d[(d.rh_mean >= 80) & ((d.t_min - d.SST).abs() <= 1)]
    d["yr"] = d["day"].dt.year
    return d.groupby(["station", "yr"]).size()



def summer_hourly(sst, coast, cols):
    """時別値と海面水温を合わせ、夏季 (6〜8月) だけにして、列 cols (と time, station, day) の表を返す。0時は前日24時として日付を付ける。"""
    d = fp.fog_dataset(sst, coast)[["time", "station"] + cols]
    d = d.assign(day=(d["time"] - pd.Timedelta(hours=1)).dt.normalize())
    return d[d["day"].dt.month.isin([6, 7, 8])]


def draw_bars(ax, k):
    """霧日数の棒 (最新の均質区間を濃く)、▲ (観測環境が変わった年)、灰色の帯 (3時間おきの年)、点線 (湿度の記録が変わった年)。"""
    cur = k.homog == k.homog.max()
    ax.axvspan(1981.5, 1989.5, color="#eeeeee", zorder=0)
    ax.axvline(2013.5, color="gray", ls=":", lw=1.2)
    ax.bar(k.yr[~cur], k.kiri[~cur], color="#b9e6b9", edgecolor="#777", lw=.5, label="霧日数(以前の区間)")
    ax.bar(k.yr[cur], k.kiri[cur], color="#2ecc40", edgecolor="k", lw=.6, label="霧日数(最新の区間)")
    for y in k.yr[k.homog.diff().fillna(0) != 0]:
        ax.plot(y, 0, "^", color="darkorange", mec="k", ms=10, clip_on=False, zorder=5)
    return cur


def correlations(k, cur, years, values, corr_from, need_std=True):
    """霧日数 k (最新の区間 cur) と values (年 years ごと) の相関 (最新の区間, corr_from 年以降) と、その年数。"""
    both = pd.DataFrame({"k": k.set_index("yr").kiri, "c": pd.Series(np.asarray(values), index=years)}).dropna()
    both = both[both.index.isin(k.yr[cur])]
    ok = lambda x: len(x) > 4 and (not need_std or x.c.std() > 0)
    late = both[both.index >= corr_from]
    return both, late, (both.k.corr(both.c) if ok(both) else np.nan), (late.k.corr(late.c) if ok(late) else np.nan)


def r_title(st, both, late, r1, r2, corr_from, empty_plain=False):
    fmt = lambda r: f"{r:.2f}" if np.isfinite(r) else "-"
    if empty_plain and not len(both):
        return st, fmt(r1), fmt(r2)
    return (f"{st}   r = {fmt(r1)} (最新の区間 {both.index.min()}〜{both.index.max()}年, n={len(both)}) / "
            f"{corr_from}年以降 r = {fmt(r2)} (n={len(late)})"), fmt(r1), fmt(r2)


def finish(fig, axs, a):
    axs.ravel()[-1].set_xlabel(XLABEL)
    fig.tight_layout()
    if a.save:
        os.makedirs(os.path.dirname(os.path.abspath(a.save)), exist_ok=True)
        fig.savefig(a.save, dpi=110)
        print("保存:", a.save)
    plt.show()


def new_figure(stations):
    return plt.subplots(len(stations), 1, figsize=(14, 2.9 * len(stations)), squeeze=False)


# ---------------------------------------------------------------- rh
def run_rh(a, kiri, coast, stations):
    h = hourly_table(fp.summer(fp.load(a.dir)), coast)
    mine = hourly_cond_days(h, a.rh_thr, a.dt_lo, a.dt_hi, a.min_hours)
    ref = teacher_days(h)
    # 6〜8月の92日のうち、気温・湿度・海面水温のある日が60日以上ある年だけ線を引く
    ok_day = h.dropna(subset=["Ta", "RH", "SST"]).groupby(["station", "yr"])["day"].nunique()
    ok_day = ok_day[(ok_day >= 60) & (ok_day.index.get_level_values(1) >= a.from_year)]
    label = (f"毎時 湿度≧{a.rh_thr:g}% かつ 気温−SST≧{a.dt_lo:g}℃"
             + (f"・≦{a.dt_hi:g}℃" if np.isfinite(a.dt_hi) else "") + f" の時刻が{a.min_hours}時間以上ある日")
    stations = a.stations or [s for s in fp.STATION_SST if s in set(kiri["station"])]

    fig, axs = new_figure(stations)
    for ax, st in zip(axs.ravel(), stations):
        k = kiri[kiri.station == st].dropna(subset=["kiri"])
        cur = draw_bars(ax, k)
        years = sorted(ok_day[ok_day.index.get_level_values(0) == st].index.get_level_values(1))
        if not a.no_ref:
            r = ref.reindex(pd.MultiIndex.from_product([[st], years]), fill_value=0)
            ax.plot(years, r.values, "--", color="gray", marker="o", ms=3, lw=1,
                    label="先生の条件 (日平均湿度≧80% かつ |日最低気温−SST|≦1℃)")
        m = mine.reindex(pd.MultiIndex.from_product([[st], years]), fill_value=0)
        ax.plot(years, m.values, "-", color="k", marker="D", ms=4, lw=1.5, label=label)
        both = pd.DataFrame({"k": k.set_index("yr").kiri, "c": pd.Series(m.values, index=years)}).dropna()
        both = both[both.index.isin(k.yr[cur])]
        both = both[both.index >= a.corr_from]
        rr = both.k.corr(both.c) if len(both) > 4 and both.c.std() > 0 else np.nan
        ax.set_title(f"{st}   {both.index.min()}〜{both.index.max()}年の相関 r = {rr:.2f} (n={len(both)}年)" if np.isfinite(rr) else st,
                     loc="left", fontsize=11)
        ax.set_ylabel("日数")
        ax.set_xlim(1981, 2027)
        ax.grid(alpha=.3)
    axs.ravel()[0].legend(loc="upper left", fontsize=8, ncol=2)
    finish(fig, axs, a)


# ---------------------------------------------------------------- cond
def cond_days(sst, coast, lo, hi, min_hours, sat_max=np.inf, dt_min=-np.inf, dt_max=np.inf):
    """(官署, 年) ごとの、「海面水温 − 露点温度」が lo〜hi で、気温 − 露点温度 ≦ sat_max、気温 − 海面水温が dt_min〜dt_max の時刻が
    min_hours 時間以上ある日の日数 (Series)。
    あわせて、露点温度と海面水温がそろった日が60日以上ある (官署, 年) の目印も返す。"""
    d = summer_hourly(sst, coast, ["Ta", "SST", "Td"])
    d = d.dropna(subset=["SST", "Td"] + (["Ta"] if np.isfinite(sat_max) or np.isfinite(dt_min) or np.isfinite(dt_max) else []))
    g = d["SST"] - d["Td"]
    hit = (g >= lo) & (g <= hi)
    if np.isfinite(sat_max):
        hit &= (d["Ta"] - d["Td"]) <= sat_max
    if np.isfinite(dt_min) or np.isfinite(dt_max):
        hit &= ((d["Ta"] - d["SST"]) >= dt_min) & ((d["Ta"] - d["SST"]) <= dt_max)
    d = d.assign(hit=hit)
    n = d.groupby(["station", "day"])["hit"].sum()
    ok = (n >= min_hours).reset_index(name="ok")
    ok["yr"] = ok["day"].dt.year
    days = ok[ok["ok"]].groupby(["station", "yr"]).size()
    have = d.groupby(["station", d["day"].dt.year])["day"].nunique()
    return days, have[have >= 60]


def run_cond(a, kiri, coast, stations):
    days, have = cond_days(fp.summer(fp.load(a.dir)), coast, a.gap_min, a.gap_max, a.min_hours, a.sat_max, a.dt_min, a.dt_max)
    parts = [(f"{a.gap_min:g}℃ ≦ " if np.isfinite(a.gap_min) else "") + f"海面水温−露点温度 ≦ {a.gap_max:g}℃"]
    if np.isfinite(a.sat_max):
        parts.append(f"気温−露点温度 ≦ {a.sat_max:g}℃")
    if np.isfinite(a.dt_min) or np.isfinite(a.dt_max):
        parts.append((f"{a.dt_min:g}℃ ≦ " if np.isfinite(a.dt_min) else "") + "気温−海面水温" + (f" ≦ {a.dt_max:g}℃" if np.isfinite(a.dt_max) else ""))
    label = " かつ ".join(parts) + f" の時刻が{a.min_hours}時間以上ある日の日数"
    ymax = max(kiri[kiri.station.isin(stations)].kiri.max(), days[days.index.get_level_values(0).isin(stations)].max()) * 1.05

    fig, axs = new_figure(stations)
    print(f"条件: {label}")
    print("官署別: 霧日数と黒線の相関  最新の区間 / 最新の区間のうち corr-from 年以降")
    for ax, st in zip(axs.ravel(), stations):
        k = kiri[kiri.station == st].dropna(subset=["kiri"])
        cur = draw_bars(ax, k)
        years = sorted(have[have.index.get_level_values(0) == st].index.get_level_values(1))
        m = days.reindex(pd.MultiIndex.from_product([[st], years]), fill_value=0)
        ax.plot(years, m.values, "-", color="k", marker="D", ms=4, lw=1.5, label=label)
        both, late, r1, r2 = correlations(k, cur, years, m.values, a.corr_from)
        title, f1, f2 = r_title(st, both, late, r1, r2, a.corr_from, empty_plain=True)
        print(f"  {st}: {f1} (n={len(both)}) / {f2} (n={len(late)})")
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_ylabel("日数")
        ax.set_xlim(1981, 2027)
        ax.set_ylim(0, ymax)
        ax.grid(alpha=.3)
    axs.ravel()[0].legend(loc="upper left", fontsize=8, ncol=2)
    finish(fig, axs, a)


# ---------------------------------------------------------------- temps / gap
def yearly(sst, coast, cols, fn):
    """(官署, 年) ごとの 夏季の fn(表) の年平均。cols の列がすべてそろった時刻だけ。"""
    d = summer_hourly(sst, coast, cols).dropna(subset=cols)
    return fn(d).groupby([d["station"], d["day"].dt.year]).mean().rename_axis(["station", "yr"])


def run_temps(a, kiri, coast, stations):
    means = yearly(fp.summer(fp.load(a.dir)), coast, ["Ta", "SST", "Td"], lambda d: d[["Ta", "SST", "Td"]])
    ymax = kiri[kiri.station.isin(stations)].kiri.max() * 1.05
    mm = means[means.index.get_level_values(0).isin(stations)]
    tlo, thi = mm.min().min(), mm.max().max()
    pad = (thi - tlo) * .05

    fig, axs = new_figure(stations)
    twins = []
    for ax, st in zip(axs.ravel(), stations):
        draw_bars(ax, kiri[kiri.station == st].dropna(subset=["kiri"]))
        m = means[means.index.get_level_values(0) == st].droplevel(0)
        ax2 = ax.twinx()
        twins.append(ax2)
        for col, name, color in LINES:
            ax2.plot(m.index, m[col], "-", color=color, marker="o", ms=3, lw=1.3, label=f"{name} (右の軸)")
        ax2.set_ylabel("気温・水温 (℃)")
        ax2.set_ylim(tlo - pad, thi + pad)
        ax.set_title(st, loc="left", fontsize=11)
        ax.set_ylabel("霧日数 (日)")
        ax.set_xlim(1981, 2027)
        ax.set_ylim(0, ymax)
        ax.grid(alpha=.3)
    h1, l1 = axs.ravel()[0].get_legend_handles_labels()
    h2, l2 = twins[0].get_legend_handles_labels()
    axs.ravel()[0].legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, ncol=5)
    finish(fig, axs, a)


def run_gap(a, kiri, coast, stations):
    gap = yearly(fp.summer(fp.load(a.dir)), coast, ["SST", "Td"], lambda d: d["SST"] - d["Td"])
    fig, axs = new_figure(stations)
    twins = []
    print("官署別: 霧日数と「海面水温−露点温度」の相関  最新の区間 / 最新の区間のうち corr-from 年以降")
    for ax, st in zip(axs.ravel(), stations):
        k = kiri[kiri.station == st].dropna(subset=["kiri"])
        cur = draw_bars(ax, k)
        g = gap[gap.index.get_level_values(0) == st].droplevel(0)
        ax2 = ax.twinx()
        twins.append(ax2)
        ax2.plot(g.index, g.values, "-", color="#cc3311", marker="o", ms=3.5, lw=1.4, label="海面水温−露点温度 (右の軸、上ほど小さい)")
        ax2.invert_yaxis()   # 上が「差が小さい」: 差が小さい年ほど霧日数が多い関係なら、棒と折れ線が同じ向きに動く
        ax2.set_ylabel("海面水温−露点温度 (℃)\n(上ほど差が小さい=霧が出やすい)", color="#cc3311", fontsize=9)
        ax2.tick_params(axis="y", colors="#cc3311")
        both, late, r1, r2 = correlations(k, cur, g.index, g.values, a.corr_from, need_std=False)
        title, f1, f2 = r_title(st, both, late, r1, r2, a.corr_from)
        print(f"  {st}: {f1} (n={len(both)}) / {f2} (n={len(late)})")
        ax.set_title(title, loc="left", fontsize=10.5)
        ax.set_ylabel("霧日数 (日)")
        ax.set_xlim(1981, 2027)
        ax.grid(alpha=.3)
    # 縦軸の範囲を全官署で同じにする (左: 霧日数 0〜最大、右: 海面水温−露点温度 の全官署の最小〜最大。右は上ほど小さい向き)
    ymax = kiri[kiri.station.isin(stations)].kiri.max() * 1.05
    gg = gap[gap.index.get_level_values(0).isin(stations)]
    pad = (gg.max() - gg.min()) * .05
    for ax, ax2 in zip(axs.ravel(), twins):
        ax.set_ylim(0, ymax)
        ax2.set_ylim(gg.max() + pad, gg.min() - pad)
    h1, l1 = axs.ravel()[0].get_legend_handles_labels()
    h2, l2 = twins[0].get_legend_handles_labels()
    axs.ravel()[0].legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, ncol=3)
    finish(fig, axs, a)


# ---------------------------------------------------------------- chart
def run_chart(a, kiri, coast, stations):
    import fog_chart as fc
    hourly, daily = fc.build_tables(fp.summer(fp.load(a.dir)), coast)
    threshold, label, need = {}, "", ("n_Ta", "n_RH")
    if a.cond == "rh":
        mask = (hourly["RH"] >= a.rh_min) & (hourly["dT"].abs() <= a.dt_abs)
        label = f"RH>={a.rh_min:g}% & |T-SST|<={a.dt_abs:g}K"
    elif a.cond == "td":
        mask = hourly["gap"] <= a.gap_max
        parts = [f"SST-Td<={a.gap_max:g}K"]
        if np.isfinite(a.sat_max):
            mask &= hourly["sat"] <= a.sat_max
            parts.append(f"T-Td<={a.sat_max:g}K")
        mask &= (hourly["dT"] >= a.dt_min) & (hourly["dT"] <= a.dt_max)
        if np.isfinite(a.dt_min) or np.isfinite(a.dt_max):
            parts.append((f"{a.dt_min:g}K<=" if np.isfinite(a.dt_min) else "") + "T-SST" + (f"<={a.dt_max:g}K" if np.isfinite(a.dt_max) else ""))
        label, need = " & ".join(parts), ("n_Ta", "n_Td")
    if a.cond != "none":
        threshold = fc.count_days(mask, hourly, daily, 1, (6, 7, 8), need)
    fc.configure(a.kiri, a.misawa, outdir=a.outdir, show=a.show, threshold=threshold, label=label)
    names = [s for s in (a.stations or fp.STATION_SST) if s in fc.processed]
    if a.cond != "none":
        print("条件:", label)
        print("霧日数の最新の均質区間での比較:")
        print(fc.compare_table(threshold).to_string(index=False))
    fc.draw_all(names)


# ---------------------------------------------------------------- main
def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dir", default=fp.SST_DIR, help="海面水温 *.txt のあるフォルダ")
    common.add_argument("--coast-dir", default=fp.COAST_DIR, help="沿岸官署の時別値CSVのフォルダ")
    common.add_argument("--kiri", default=os.path.join(fp.HERE, "東北地方気象官署月別値霧日数.csv"), help="霧日数の月別値CSV")
    common.add_argument("--stations", nargs="*", help="官署名 (省略で全官署)")
    common.add_argument("--corr-from", type=int, default=2014, help="相関を取る最初の年 (湿度の記録が変わった後)")
    common.add_argument("--save", help="PNGの保存先 (省略で画面表示のみ)")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", metavar="モード")
    q = sub.add_parser("rh", parents=[common], help="相対湿度 と 気温−海面水温 の条件の日数")
    q.add_argument("--rh-thr", type=float, default=98, help="毎時の相対湿度がこれ以上")
    q.add_argument("--dt-lo", type=float, default=-2, help="気温−海面水温がこれ以上 (℃)")
    q.add_argument("--dt-hi", type=float, default=np.inf, help="気温−海面水温がこれ以下 (℃)")
    q.add_argument("--min-hours", type=int, default=1, help="1日にこの時間数以上 条件を満たせば、その日を数える")
    q.add_argument("--from-year", type=int, default=1982, help="黒線を引く最初の年 (海面水温は1982年から)")
    q.add_argument("--no-ref", action="store_true", help="先生の条件(灰色の破線)を重ねない")
    q = sub.add_parser("cond", parents=[common], help="海面水温−露点温度 などの条件の日数")
    q.add_argument("--gap-max", type=float, default=0, help="海面水温 − 露点温度 がこれ以下 (℃)")
    q.add_argument("--gap-min", type=float, default=-np.inf, help="海面水温 − 露点温度 がこれ以上 (℃)")
    q.add_argument("--sat-max", type=float, default=np.inf, help="気温 − 露点温度 がこれ以下 (℃)。省略で使わない")
    q.add_argument("--dt-min", type=float, default=-np.inf, help="気温 − 海面水温 がこれ以上 (℃)。省略で使わない")
    q.add_argument("--dt-max", type=float, default=np.inf, help="気温 − 海面水温 がこれ以下 (℃)。省略で使わない")
    q.add_argument("--min-hours", type=int, default=1, help="1日にこの時間数以上 条件を満たせば、その日を数える")
    q = sub.add_parser("chart", parents=[common], help="Colab のノートブックと同じ図 (官署ごとに1枚の画像)")
    q.add_argument("--cond", choices=("rh", "td", "none"), default="rh", help="黒線の条件: rh=湿度と気温−海面水温 / td=露点温度 / none=なし")
    q.add_argument("--rh-min", type=float, default=95, help="[rh] 相対湿度 ≧ この値 (%%)")
    q.add_argument("--dt-abs", type=float, default=0, help="[rh] |気温−海面水温| ≦ この値 (℃)")
    q.add_argument("--gap-max", type=float, default=0, help="[td] 海面水温 − 露点温度 ≦ この値 (℃)")
    q.add_argument("--sat-max", type=float, default=np.inf, help="[td] 気温 − 露点温度 ≦ この値 (℃)。省略で使わない")
    q.add_argument("--dt-min", type=float, default=-np.inf, help="[td] 気温 − 海面水温 ≧ この値 (℃)。省略で使わない")
    q.add_argument("--dt-max", type=float, default=np.inf, help="[td] 気温 − 海面水温 ≦ この値 (℃)。省略で使わない")
    q.add_argument("--misawa", default=os.path.join(fp.HERE, "三沢月別値霧日数.csv"), help="三沢の月別値CSV (八戸の図に重ねる。無ければ描かない)")
    q.add_argument("--outdir", default=os.path.join(fp.HERE, "out"), help="画像の保存先フォルダ")
    q.add_argument("--show", action="store_true", help="保存したあと、画面にも表示する")
    sub.add_parser("temps", parents=[common], help="気温・海面水温・露点温度の年平均")
    sub.add_parser("gap", parents=[common], help="海面水温−露点温度の年平均")
    argv = sys.argv[1:]
    if not argv or (argv[0] not in MODES and argv[0] not in ("-h", "--help")):
        argv = ["rh"] + argv        # モードを省略したら rh
    a = p.parse_args(argv)

    fp.setup_font()
    kiri = load_kiri(a.kiri)
    coast = fp.load_coast(a.coast_dir)
    if coast is None:
        raise SystemExit("沿岸官署のCSVが見つかりません: " + a.coast_dir)
    stations = [s for s in (a.stations or fp.STATION_SST) if s in set(kiri["station"])]
    {"rh": run_rh, "cond": run_cond, "temps": run_temps, "gap": run_gap, "chart": run_chart}[a.mode](a, kiri, coast, stations)


if __name__ == "__main__":
    main()
