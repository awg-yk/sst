"""夏季 (6〜8月) の視程<1kmの割合 (%) の年ごとの推移を、官署ごとに折れ線で示す図。
観測が毎時になる前後で公平に比べられるよう、官署ごとに「1993年から全期間を通じて観測のある時刻」だけを使います。
fog_plot.py と同じフォルダに置いて実行します。

    python fog_plot_vis_rate.py                  # 官署ごとの図を画面に表示
    python fog_plot_vis_rate.py --from-year 1989 # この年から (既定は1993。視程の記録は1989年から)
    python fog_plot_vis_rate.py --save-dir out   # 画面に出さずPNGを保存

割合 = (使う時刻のうち視程<1kmだった時間) ÷ (使う時刻の観測時間)。使う時刻は各図の題に書きます。
▲ = 視程が毎時になった年 (観測が目視から視程計に自動化された翌夏。気象観測統計指針 表3.4－6、fog_plot.py の AUTOMATION)。
視程<1kmは霧とは限りません (降水・煙霧なども含む)。
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

import fog_plot as fp
from fog_plot_year import panels

MIN_SHARE = .5   # その時刻の観測が、夏の日数の何割以上ある年を「観測のある年」とするか


def fixed_hours(vis, from_year):
    """全期間 (from_year〜) を通じて、どの年も夏の日数の半分以上で視程の観測がある時刻の一覧。"""
    cnt = vis.groupby([vis["y"], vis["h"]]).size().unstack(fill_value=0)
    return [h for h in cnt.columns if (cnt[h] >= MIN_SHARE * 92).all()]


def rates(coast, station, from_year):
    """官署ひとつの、(年ごとの視程<1kmの割合%, 使う時刻)。使える時刻が無ければ (None, [])。"""
    d = coast[(coast["station"] == station) & coast["vis"].notna()].copy()
    day = (d["time"] - pd.Timedelta(hours=1)).dt.normalize()
    d = d[day.dt.month.isin([6, 7, 8]) & (day.dt.year >= from_year)]
    d["y"], d["h"] = (d["time"] - pd.Timedelta(hours=1)).dt.year, d["time"].dt.hour
    hrs = fixed_hours(d, from_year)
    if not hrs:
        return None, []
    d = d[d["h"].isin(hrs)]
    return (d["vis"] < fp.FOG_KM).groupby(d["y"]).mean() * 100, hrs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coast-dir", default=fp.COAST_DIR, help="沿岸官署の時別値CSVのフォルダ")
    p.add_argument("--from-year", type=int, default=1993, help="この年から使う (既定1993)")
    p.add_argument("--save-dir", help="画面に出さずPNGをこのフォルダに保存する")
    a = p.parse_args()

    fp.setup_font()
    coast = fp.load_coast(a.coast_dir)
    if coast is None:
        raise SystemExit(f"沿岸官署のCSVが見つかりません: {a.coast_dir} (--coast-dir で指定)")
    start = fp.hourly_start_years()
    names = [s for s in fp.STATION_SST if s in set(coast["station"])]
    res = {s: rates(coast, s, a.from_year) for s in names}

    fig, axs = panels(names)
    print("官署別: 使う時刻 / 視程<1kmの割合(%) 自動化の前→後の平均")
    for ax, s in zip(axs.ravel(), names):
        r, hrs = res[s]
        if r is None:
            ax.set_title(f"{s}: 全期間を通じた観測時刻なし")
            continue
        ax.plot(r.index, r.values, "o-", ms=3, lw=.8, color="#4477aa")
        y0 = start.get(s)
        if y0 and r.index.min() < y0 <= r.index.max():
            ax.plot([y0], [0], "^", color="#cc3311", ms=8, clip_on=False, zorder=5)
            ax.axvline(y0, color="#cc3311", ls=":", lw=.8)
            print(f"  {s}: {hrs} / {r[r.index < y0].mean():.2f} → {r[r.index >= y0].mean():.2f}")
        else:
            print(f"  {s}: {hrs} / 自動化の年 {y0} は期間外 (平均 {r.mean():.2f})")
        ax.set(title=f"{s} (使う時刻: {','.join(str(h) for h in hrs)}時)")
        ax.grid(alpha=.3)
    top = max(r.max() for r, _ in res.values() if r is not None)
    axs[0, 0].set_ylim(0, top * 1.08)   # 全官署で縦軸をそろえる
    for ax in axs[:, 0]:
        ax.set_ylabel("視程<1kmの割合 (%)")
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.text(.01, .004, "割合 = 使う時刻のうち視程<1kmだった時間 ÷ 観測時間 (時刻は題に記載。1993年から全期間を通じて観測のある時刻だけ)。▲・点線 = 視程が毎時になった年 "
             "(観測の自動化の翌夏。気象観測統計指針 表3.4－6)。\n視程<1kmは霧とは限らない (降水・煙霧なども含む)。", fontsize=8)
    if a.save_dir:
        os.makedirs(a.save_dir, exist_ok=True)
        path = os.path.join(a.save_dir, "fog_vis_rate.png")
        fig.savefig(path, dpi=130)
        print("保存:", path)
    else:
        plt.show()


if __name__ == "__main__":
    main()
