"""露点温度(横軸) × 海面水温(縦軸) を1℃ごとに区切って、視程<1kmの出現率を示す図 (fog_plot_ta_sst.py の横軸を気温から露点温度に変えたもの)。
fog_plot.py と fog_plot_ta_sst.py と同じフォルダに置いて実行します。
露点温度は「東北地方沿岸気象官署時別値露点温度」フォルダの時別値CSV (1984年〜) を使います。

    python fog_plot_td_sst.py                   # 沿岸10官署を並べた図を画面に表示
    python fog_plot_td_sst.py --each            # 官署ごとに1枚ずつ表示 (マス内に回数も表示)
    python fog_plot_td_sst.py --each --save-dir out   # 官署ごとのPNGを保存
    python fog_plot_td_sst.py --from-year 2014  # この年以降だけ使う

色: 視程<1kmの出現率(%) = 視程<1kmだった回数 ÷ 該当した回数。全官署で同じ色の基準。
灰色: 視程<1kmが0回、または該当した回数が40回未満。空白: 該当なし。
破線: 露点温度 = 海面水温。 マス内の数字 (--each のとき): 上=視程<1kmだった回数 / 下=該当した回数。
区切りは1℃ごとで、目盛りの数字はマスの境目です (各マスは例えば16.0℃以上17.0℃未満)。
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fog_plot as fp
import fog_plot_ta_sst as ts

DEW_DIR = os.path.join(fp.HERE, "東北地方沿岸気象官署時別値露点温度")


def load_dew(dew_dir=DEW_DIR):
    """露点温度の時別値CSV(年ごと)を読み、時刻×官署の縦長DataFrame(time, station, Td)を返す。品質情報が8(正常)の値だけを使う。"""
    import glob
    files = sorted(glob.glob(os.path.join(dew_dir, "*.csv")))
    if not files:
        raise SystemExit(f"露点温度のCSVが見つかりません: {dew_dir} (--dew-dir で指定)")
    out = []
    for path in files:
        rows = fp._read_rows(path)
        h = next((i for i, r in enumerate(rows) if r and r[0].startswith("年月日時")), None)
        if h is None:
            print("  読み飛ばし(形式が違う):", os.path.basename(path))
            continue
        names = rows[h - 1]
        data = [r for r in rows[h + 1:] if r and r[0] and r[0][0].isdigit()]
        t = pd.to_datetime([r[0] for r in data], format="%Y/%m/%d %H:%M:%S", errors="coerce")
        ok = ~t.isna()
        data, t = [r for r, k in zip(data, ok) if k], t[ok]
        width = max(len(r) for r in data)
        arr = np.array([r + [""] * (width - len(r)) for r in data], dtype=object)
        for st in dict.fromkeys(n for n in names[1:] if n):
            c = next(i for i in range(1, len(names)) if names[i] == st)
            v = pd.to_numeric(pd.Series(arr[:, c]), errors="coerce").values
            q = arr[:, c + 1] if c + 1 < width else np.full(len(t), "8")
            out.append(pd.DataFrame({"time": t, "station": st, "Td": np.where(q == "8", v, np.nan)}))
    return pd.concat(out, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=fp.HERE, help="海面水温 *.txt のあるフォルダ")
    p.add_argument("--coast-dir", default=fp.COAST_DIR, help="沿岸官署の時別値CSVのフォルダ (視程を使う)")
    p.add_argument("--dew-dir", default=DEW_DIR, help="露点温度の時別値CSVのフォルダ")
    p.add_argument("--from-year", type=int, help="この年以降だけ使う")
    p.add_argument("--each", action="store_true", help="官署ごとに1枚ずつ表示する")
    p.add_argument("--stations", nargs="*", help="官署名 (省略で沿岸10官署)")
    p.add_argument("--save-dir", help="--each のとき、画面に出さずPNGをこのフォルダに保存する")
    a = p.parse_args()

    fp.setup_font()
    coast = fp.load_coast(a.coast_dir)
    if coast is None:
        raise SystemExit(f"沿岸官署のCSVが見つかりません: {a.coast_dir} (--coast-dir で指定)")
    d = fp.fog_dataset(fp.summer(fp.load(a.dir)), coast)
    d = d.merge(load_dew(a.dew_dir), on=["time", "station"], how="left")
    if a.from_year:
        d = d[d["time"].dt.year >= a.from_year]
    d = d[d["vis"].notna() & d["Td"].notna() & d["SST"].notna()].copy()
    d["Ta"] = d["Td"]          # fog_plot_ta_sst の関数をそのまま使うため、横軸の列 Ta に露点温度を入れる
    allst = [s for s in fp.STATION_SST if s in set(d["station"])]
    lo, hi = ts.temp_range(d)
    tabs = {s: ts.ta_sst_table(d[d["station"] == s], lo, hi) for s in allst}
    norm = fp.fog_norm(list(tabs.values()))          # 全官署で共通の色の基準
    cmap = plt.get_cmap("magma_r")
    names = [s for s in (a.stations or allst) if s in tabs]
    title = lambda s: f"{s} (海域: {fp.STATION_SST[s]})"
    note = ("色=視程<1kmの出現率 (上÷下)。灰色=視程<1kmが0回、または該当した回数が40回未満。空白=該当なし。"
            "\n破線=露点温度と海面水温が等しい。目盛りはマスの境目 (各マスは例えば16.0℃以上17.0℃未満)")

    if a.each:
        for s in names:
            fig, ax = plt.subplots(figsize=(10, 8.5))
            sm = ts.heat(ax, *tabs[s], title(s), norm, cmap, numbers=True)
            ax.set_xlabel("露点温度 (℃)")
            fig.colorbar(sm, ax=ax, shrink=.85).set_label("視程<1kmの出現率 (%)")
            fig.tight_layout(rect=(0, .05, 1, 1))
            fig.text(.01, .004, "マス内の数字: 上=視程<1kmだった回数 / 下=該当した回数。" + note, fontsize=7)
            if a.save_dir:
                os.makedirs(a.save_dir, exist_ok=True)
                path = os.path.join(a.save_dir, f"fog_td_sst_{s}.png")
                fig.savefig(path, dpi=130)
                print("保存:", path)
                plt.close(fig)
            else:
                plt.show()
        return

    ncol = (len(names) + 1) // 2
    fig, axs = plt.subplots(2, ncol, figsize=(3.9 * ncol + 1, 8), squeeze=False)
    for ax, s in zip(axs.ravel(), names):
        sm = ts.heat(ax, *tabs[s], title(s), norm, cmap, numbers=False)
        ax.set_xlabel("露点温度 (℃)")
    for ax in axs.ravel()[len(names):]:
        ax.axis("off")
    fig.tight_layout(rect=(0, .06, .93, 1))
    fig.text(.01, .004, note, fontsize=9)
    cax = fig.add_axes([.945, .2, .015, .6])
    fig.colorbar(sm, cax=cax).set_label("視程<1kmの出現率 (%)", fontsize=9)
    plt.show()


if __name__ == "__main__":
    main()
