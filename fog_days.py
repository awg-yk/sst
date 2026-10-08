"""霧日数(棒)に、「相対湿度」と「気温−海面水温」の条件を満たす時刻がある日の日数(黒線)を重ねて表示する。
fog_plot.py と同じフォルダに置いて実行します (fog_plot.py の読み込み処理を使います)。

    python fog_days.py                       # 既定の条件で全官署
    python fog_days.py --stations 八戸 宮古     # 官署を選ぶ
    python fog_days.py --rh-thr 98 --dt-lo -1 --dt-hi 1 --min-hours 2   # 条件を変える
    python fog_days.py --save out/fog_days.png

黒線: 6〜8月で、毎時の観測の「相対湿度 ≧ --rh-thr」かつ「--dt-lo ≦ 気温−その日の海面水温 ≦ --dt-hi」を
      満たす時刻が --min-hours 時間以上ある日の、年ごとの日数。
      既定: 湿度98%以上、気温−海面水温 −2℃以上(上限なし)、1時間以上。
灰色の破線: 先生の条件 (日平均湿度≧80% かつ |日最低気温−海面水温|≦1℃) の日数。
霧日数: 気象庁の月別値「霧日数」CSV (--kiri、既定は同じフォルダの tohoku.csv) の6〜8月の合計。
        均質番号(観測環境の変化)が変わった年に▲を付け、最新の区間を濃い緑にします。

注意:
  - 気温・湿度は1990年から毎時、1989年以前は3時間おき以下なので、1989年以前(薄い灰色の帯)は日数が少なめに出ます。
  - 相対湿度の記録は2013年ごろに変わっていて (湿度100%の記録が急に増える)、それ以前は湿度の条件の日数が
    他の年と比べられません (縦の点線)。相関は --corr-from (既定2014) 以降だけで出します。
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fog_plot as fp


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
            val = float(r[c]) if r[c] != "" and r[c + 1] == "8" else np.nan
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=fp.HERE, help="海面水温 *.txt のあるフォルダ")
    p.add_argument("--coast-dir", default=fp.COAST_DIR)
    p.add_argument("--kiri", default=os.path.join(fp.HERE, "tohoku.csv"), help="霧日数の月別値CSV")
    p.add_argument("--stations", nargs="*", help="官署名 (省略で全官署)")
    p.add_argument("--rh-thr", type=float, default=98, help="毎時の相対湿度がこれ以上")
    p.add_argument("--dt-lo", type=float, default=-2, help="気温−海面水温がこれ以上 (℃)")
    p.add_argument("--dt-hi", type=float, default=np.inf, help="気温−海面水温がこれ以下 (℃)")
    p.add_argument("--min-hours", type=int, default=1, help="1日にこの時間数以上 条件を満たせば、その日を数える")
    p.add_argument("--from-year", type=int, default=1982, help="黒線を引く最初の年 (海面水温は1982年から)")
    p.add_argument("--corr-from", type=int, default=2014, help="相関を取る最初の年 (湿度の記録が変わった後)")
    p.add_argument("--no-ref", action="store_true", help="先生の条件(灰色の破線)を重ねない")
    p.add_argument("--save", help="PNGの保存先 (省略で画面表示のみ)")
    a = p.parse_args()

    fp.setup_font()
    kiri = load_kiri(a.kiri)
    coast = fp.load_coast(a.coast_dir)
    if coast is None:
        raise SystemExit("沿岸官署のCSVが見つかりません: " + a.coast_dir)
    h = hourly_table(fp.summer(fp.load(a.dir)), coast)
    mine = hourly_cond_days(h, a.rh_thr, a.dt_lo, a.dt_hi, a.min_hours)
    ref = teacher_days(h)
    # 6〜8月の92日のうち、気温・湿度・海面水温のある日が60日以上ある年だけ線を引く
    ok_day = h.dropna(subset=["Ta", "RH", "SST"]).groupby(["station", "yr"])["day"].nunique()
    ok_day = ok_day[(ok_day >= 60) & (ok_day.index.get_level_values(1) >= a.from_year)]
    label = (f"毎時 湿度≧{a.rh_thr:g}% かつ 気温−SST≧{a.dt_lo:g}℃"
             + (f"・≦{a.dt_hi:g}℃" if np.isfinite(a.dt_hi) else "") + f" の時刻が{a.min_hours}時間以上ある日")
    stations = a.stations or [s for s in fp.STATION_SST if s in set(kiri["station"])]

    fig, axs = plt.subplots(len(stations), 1, figsize=(14, 2.9 * len(stations)), squeeze=False)
    for ax, st in zip(axs.ravel(), stations):
        k = kiri[kiri.station == st].dropna(subset=["kiri"])
        cur = k.homog == k.homog.max()
        ax.axvspan(1981.5, 1989.5, color="#eeeeee", zorder=0)
        ax.axvline(2013.5, color="gray", ls=":", lw=1.2)
        ax.bar(k.yr[~cur], k.kiri[~cur], color="#b9e6b9", edgecolor="#777", lw=.5, label="霧日数(以前の区間)")
        ax.bar(k.yr[cur], k.kiri[cur], color="#2ecc40", edgecolor="k", lw=.6, label="霧日数(最新の区間)")
        for y in k.yr[k.homog.diff().fillna(0) != 0]:
            ax.plot(y, 0, "^", color="darkorange", mec="k", ms=10, clip_on=False, zorder=5)
        years = sorted(ok_day[ok_day.index.get_level_values(0) == st].index.get_level_values(1))
        if not a.no_ref:
            r = ref.reindex(pd.MultiIndex.from_product([[st], years]), fill_value=0)
            ax.plot(years, r.values, "--", color="gray", marker="o", ms=3, lw=1,
                    label="先生の条件 (日平均湿度≧80% かつ |日最低気温−SST|≦1℃)")
        m = mine.reindex(pd.MultiIndex.from_product([[st], years]), fill_value=0)
        ax.plot(years, m.values, "-", color="k", marker="D", ms=4, lw=1.5, label=label)
        both = pd.DataFrame({"k": k.set_index("yr").kiri, "c": pd.Series(m.values, index=years)}).dropna()
        both = both[both.index.isin(k.yr[cur]) & (both.index >= a.corr_from)]
        rr = both.k.corr(both.c) if len(both) > 4 and both.c.std() > 0 else np.nan
        ax.set_title(f"{st}   {both.index.min()}〜{both.index.max()}年の相関 r = {rr:.2f} (n={len(both)}年)" if np.isfinite(rr) else st,
                     loc="left", fontsize=11)
        ax.set_ylabel("日数")
        ax.set_xlim(1981, 2027)
        ax.grid(alpha=.3)
    axs.ravel()[0].legend(loc="upper left", fontsize=8, ncol=2)
    axs.ravel()[-1].set_xlabel("年  (灰色の帯: 気温・湿度が3時間おき以下の年 / 点線: 湿度の記録が変わった年)")
    fig.tight_layout()
    if a.save:
        os.makedirs(os.path.dirname(os.path.abspath(a.save)), exist_ok=True)
        fig.savefig(a.save, dpi=110)
        print("保存:", a.save)
    plt.show()


if __name__ == "__main__":
    main()
