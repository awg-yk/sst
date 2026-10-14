"""三沢飛行場(METAR)の6〜8月の風向頻度(風配図)。ポスター用の日本語版。
使い方: py fog_plot_wind_rose.py METAR_RJSM.csv --groups 2015-2019 2020-2024 --save out.png
        py fog_plot_wind_rose.py METAR_RJSM.csv --years 2015 2020 2024
縦軸は有効な風向(10〜360度)の観測に対する割合(%)。"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fog_plot as fp

ap = argparse.ArgumentParser()
ap.add_argument('csv')
ap.add_argument('--years', nargs='+', type=int)
ap.add_argument('--groups', nargs='+', help='例: 2015-2019 2020-2024')
ap.add_argument('--size-cm', nargs=2, type=float, default=[20, 16])
ap.add_argument('--fontsize', type=float, default=24)
ap.add_argument('--save')
a = ap.parse_args()

df = pd.read_csv(a.csv, usecols=['DTI', 'DDD'], parse_dates=['DTI'])
df = df[df.DTI.dt.month.isin([6, 7, 8])]
if a.years:
    sets = [(f'{y}年', [y]) for y in a.years]
else:
    gs = a.groups or ['2015-2019', '2020-2024']
    sets = []
    for g in gs:
        s, e = map(int, g.split('-'))
        sets.append((f'{s}〜{e}年', list(range(s, e + 1))))

fp.setup_font()
fs = a.fontsize
plt.rcParams.update({'font.size': fs, 'axes.unicode_minus': False})
fig = plt.figure(figsize=(a.size_cm[0] / 2.54, a.size_cm[1] / 2.54))
ax = fig.add_axes([.22, .07, .72, .72], projection='polar')
dirs = np.arange(10, 361, 10)
cols = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
for i, (lab, ys) in enumerate(sets):
    d = df[df.DTI.dt.year.isin(ys)].DDD.dropna()
    d = d[(d >= 10) & (d <= 360) & (d % 10 == 0)]
    p = d.value_counts().reindex(dirs, fill_value=0).values / len(d) * 100
    th = np.deg2rad(np.append(dirs, dirs[0]))
    ax.plot(th, np.append(p, p[0]), color=cols[i], lw=2.5, label=lab)
    ax.fill(th, np.append(p, p[0]), color=cols[i], alpha=.08)
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_xticks(np.deg2rad(np.arange(0, 360, 45)))
ax.set_xticklabels(['北', '北東', '東', '南東', '南', '南西', '西', '北西'], fontsize=fs * 1.2, fontweight='bold')
ax.tick_params(axis='x', pad=8)
ax.set_rlabel_position(-22.5)
ax.tick_params(axis='y', labelsize=fs * .8)
ax.grid(True, ls='--', alpha=.7)
fig.suptitle('三沢飛行場 6〜8月の風向頻度(%)', fontsize=fs * 1.15, fontweight='bold', y=.985)
fig.legend(loc='upper left', bbox_to_anchor=(0, .9), fontsize=fs * .85, prop={'weight': 'bold', 'size': fs * .85}, handlelength=1.5)
if a.save:
    fig.savefig(a.save, dpi=300)
else:
    plt.show()
