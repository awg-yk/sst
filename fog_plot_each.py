"""fog_plot.py の図を、官署ごとに1枚ずつ表示する (fog_plot.py と同じフォルダに置いて実行します)。

    python fog_plot_each.py                    # 沿岸10官署を1枚ずつ画面に表示 (ウィンドウを閉じると次の官署)
    python fog_plot_each.py --stations 八戸 宮古   # 官署を選ぶ
    python fog_plot_each.py --save-dir out     # 画面に出さず、官署ごとのPNGを out/ に保存
    python fog_plot_each.py --from-year 2014   # この年以降だけ使う (fog_plot.py と同じオプション)

図: 相対湿度 × (気温−海面水温) の階級ごとの、視程<1kmの出現率 (色)。マス内の数字は 上=視程<1kmだった回数 / 下=該当した回数。
    色の濃さは fog_plot.py の10官署並べた図と同じ基準 (全官署で共通。最大の官署に合わせる) です。
"""
import argparse
import os

import matplotlib.pyplot as plt

import fog_plot as fp


def common_norm(v):
    """fog_plot.py の図と同じ、全官署で共通の色の基準"""
    stations = [s for s in fp.STATION_SST if s in set(v["station"])]
    return fp.fog_norm([fp.fog_table(v[v["station"] == s]) for s in stations])


def station_figure(v, name, norm):
    """1官署の図 (Figure)"""
    n, k = fp.fog_table(v[v["station"] == name])
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sm = fp.fog_heat(ax, n, k, f"{name} (海域: {fp.STATION_SST[name]})", norm, plt.get_cmap("magma_r"))
    fig.colorbar(sm, ax=ax).set_label("視程<1kmの出現率 (%)")
    fig.text(.01, .005, fp.FOG_NOTE, fontsize=7)
    fig.tight_layout(rect=(0, .02, 1, 1))
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=fp.HERE, help="海面水温 *.txt のあるフォルダ")
    p.add_argument("--coast-dir", default=fp.COAST_DIR, help="沿岸官署の時別値CSVのフォルダ")
    p.add_argument("--from-year", type=int, help="この年以降だけ使う")
    p.add_argument("--hourly-only", action="store_true", help="官署ごとに、視程が毎時になった年以降だけを使う")
    p.add_argument("--stations", nargs="*", help="官署名 (省略で沿岸10官署)")
    p.add_argument("--save-dir", help="指定すると、画面に出さずPNGをこのフォルダに保存する")
    a = p.parse_args()

    fp.setup_font()
    _, v = fp.fog_valid(fp.summer(fp.load(a.dir)), a)
    names = a.stations or [s for s in fp.STATION_SST if s in set(v["station"])]
    norm = common_norm(v)
    for name in names:
        if name not in set(v["station"]):
            print("データがありません:", name)
            continue
        fig = station_figure(v, name, norm)
        if a.save_dir:
            os.makedirs(a.save_dir, exist_ok=True)
            path = os.path.join(a.save_dir, f"fog_{name}.png")
            fig.savefig(path, dpi=130)
            print("保存:", path)
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    main()
