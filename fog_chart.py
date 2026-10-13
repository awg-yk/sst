"""霧日数 (気象庁の月別値) の棒グラフに、条件を満たす日数 (黒線) を重ねて、官署ごとに1枚の画像にする (fog_days.py の chart モードが使う)。
区間ごとの5年移動平均と回帰直線 (傾きは Day/Decade)、観測環境が変わった年の▲、欠測の表示、三沢 (八戸の図だけ) の線も描く。

fog_days.py と同じフォルダに置いて使います (単独では動きません)。
霧日数: 東北地方気象官署月別値霧日数.csv、三沢: 三沢月別値霧日数.csv (どちらも Python ファイルと同じフォルダ)。
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fog_plot as fp

QUALITY_OK = {"8"}
MIN_DAYS_FRAC = 0.65        # 対象の月の日数のうち、この割合以上の日がそろっていない年は黒線を描かない
COND_FROM_YEAR = 1982       # 黒線を描く最初の年 (海面水温は1982年から)
SAVE_FIG = True             # 地点ごとの画像を OUTDIR に保存する
OUTDIR = os.path.join(fp.HERE, "out")   # 地点ごとの画像の保存先
SAVE_DPI = 150
SHOW = False                # True なら、保存したあと画面にも表示する

# 以下は configure() で設定する
processed = {}                  # {官署: 年ごとの霧日数 DataFrame}
threshold_data = {}             # {官署: 条件を満たす日数 DataFrame(year, total_fog_days)}
THRESHOLD_LABEL = ""            # 黒線の凡例
df_misawa_metar = pd.DataFrame(columns=["year", "total_fog_days"])
df_misawa_combined = pd.DataFrame(columns=["year", "total_fog_days"])


def linregress(x, y):
    """最小二乗の直線 (傾き, 切片)。scipy を使わないための簡単な版。"""
    slope, intercept = np.polyfit(np.asarray(x, dtype=float), np.asarray(y, dtype=float), 1)
    return slope, intercept, None


SUMMER_MONTHS = [6, 7, 8]
MISAWA_MIN_VALID_MONTHS = 3     # 三沢：夏季の有効月数（月別データ）がこれ未満の年は除外
TRIM_LEADING_ZEROS = True       # 先頭の「霧日数0の年」を除外するか
ENV_LABEL = 'Change in observation environment'   # ▲の凡例表記
MIN_SEGMENT_YEARS = 5           # 区間ごとの5年移動平均・トレンドを描く最小年数

# トレンド線の個別調整（見た目の指摘を受けての手動調整。5年移動平均・棒グラフ・▲・欠測表示には影響しない）
#   TREND_TRIM_START: そのトレンド区間の開始年をここで指定した年以降に切り詰める
#     （むつは1940年代のMissing dataの前にごく短いデータがあり、そこから引くと違和感があるため）
TREND_TRIM_START = {
    'むつ': 1947,
}
#   TREND_EXCLUDE_START_YEARS: 開始年がここに挙げた年と一致するトレンド区間を描かない
#     （霧がほとんど出ていない短い区間に無理に回帰直線を引くと誤解を招くため）
TREND_EXCLUDE_START_YEARS = {
    'むつ': [1981, 1992],
    '新庄': [1992],
    '石巻': [1995],
    '白河': [1992],
}


def adjust_trend_segments(df, loc_name, trend_segments):
    """TREND_TRIM_START / TREND_EXCLUDE_START_YEARS を反映する。"""
    trim_year = TREND_TRIM_START.get(loc_name)
    exclude_years = set(TREND_EXCLUDE_START_YEARS.get(loc_name, []))
    result = []
    for s, e in trend_segments:
        start_year = int(df.loc[s, 'year'])
        if trim_year is not None and start_year < trim_year:
            candidates = df.loc[s:e]
            new_start = candidates[candidates['year'] >= trim_year].index.min()
            if pd.isna(new_start):
                continue
            s = int(new_start)
            start_year = int(df.loc[s, 'year'])
        if start_year in exclude_years:
            continue
        result.append((s, e))
    return result

# 区切り（▲）は tohoku.csv 自体の情報から決める:
#   ・均質番号が変わる所（観測環境の変化）
#   ・品質情報が 1 の月（欠測）で年が欠ける所

def adjust_ma_segments(ma_segments, trend_segments_adjusted):
    """5年移動平均の区間を、調整後のトレンド区間に重なる範囲だけに絞る。
    トレンド線を消した/切り詰めた箇所では、5年移動平均も同じ範囲だけ消える。"""
    result = []
    for ms, me in ma_segments:
        for ts, te in trend_segments_adjusted:
            lo, hi = max(ms, ts), min(me, te)
            if lo <= hi:
                result.append((lo, hi))
    return result


def load_tohoku(path):
    """霧日数の月別値CSV (気象庁) を読み、{官署: 夏季(6〜8月)の年ごとの霧日数 DataFrame} を返す。"""
    def detect_encoding(p, candidates=('cp932', 'utf-8')):
        for enc in candidates:
            try:
                with open(p, 'r', encoding=enc) as f:
                    f.read()
                return enc
            except UnicodeDecodeError:
                continue
        return 'utf-8'

    enc = detect_encoding(path)
    df = pd.read_csv(path, encoding=enc, skiprows=4, header=None)
    header_lines = []
    with open(path, 'r', encoding=enc, errors='ignore') as f:
        for _ in range(4):
            line = f.readline()
            if not line:
                break
            header_lines.append(line.strip().split(','))

    columns = [i for i in range(1, df.shape[1]) if (i - 1) % 3 == 0]
    names = {}
    for col_idx in columns:
        name = None
        for row_cells in header_lines:
            if col_idx < len(row_cells):
                val = row_cells[col_idx].strip().replace('"', '')
                if val != "" and "ダウンロード" not in val and "品質" not in val and "均質" not in val:
                    name = val
                    break
        if not name:
            for row_cells in header_lines:
                if (col_idx - 1) < len(row_cells):
                    val = row_cells[col_idx - 1].strip().replace('"', '')
                    if val != "" and "ダウンロード" not in val:
                        name = val
                        break
        names[col_idx] = name if name else f"Location_Col_{col_idx}"
    return {name: get_processed_annual_data(df, 0, col_idx) for col_idx, name in names.items()}


def load_misawa(path):
    """三沢の月別値CSV (METAR と METAR+SPECIAL の霧日数) を読み、夏季の年ごとの合計 (METAR, METAR+SPECIAL) を返す。
    ファイルが無ければ空の表を返す (三沢は描かない)。"""
    empty = pd.DataFrame(columns=['year', 'total_fog_days'])
    if not os.path.exists(path):
        print('三沢のファイルが無いので描きません:', path)
        return empty, empty
    raw = pd.read_csv(path, encoding='utf-8', header=None).iloc[1:, :3].copy()   # 先頭はヘッダー行
    raw.columns = ['datetime', 'metar_fog', 'combined_fog']
    raw['date'] = pd.to_datetime(raw['datetime'], format='%Y/%m/%d', errors='coerce')  # 月初日付
    raw = raw.dropna(subset=['date'])
    raw['year'] = raw['date'].dt.year
    raw['month'] = raw['date'].dt.month
    summer = raw[raw['month'].isin(SUMMER_MONTHS)].copy()
    # 欠測を 0（霧なし）にしない。NaN のまま扱い、有効月数を数える
    summer['metar_fog'] = pd.to_numeric(summer['metar_fog'], errors='coerce')
    summer['combined_fog'] = pd.to_numeric(summer['combined_fog'], errors='coerce')

    def annual(col):
        g = summer.groupby('year')[col]
        out = pd.DataFrame({'total_fog_days': g.sum(), 'n_valid': g.count()})
        dropped = out[out['n_valid'] < MISAWA_MIN_VALID_MONTHS]
        if not dropped.empty:
            print(f"三沢 {col}: 有効月数不足で除外した年 ->", {int(y): int(n) for y, n in dropped['n_valid'].items()})
        out = out[out['n_valid'] >= MISAWA_MIN_VALID_MONTHS]
        # 年の上限は設けない。METAR+SPECIAL は 2025年以降が空欄のため、有効月数の条件で自動的に除外される
        return out[['total_fog_days']].reset_index()

    return annual('metar_fog'), annual('combined_fog')


def configure(tohoku_path, misawa_path, outdir=None, show=False, threshold=None, label=""):
    """霧日数と三沢を読み、条件の日数 threshold ({官署: DataFrame}) と凡例 label を設定する。"""
    global processed, df_misawa_metar, df_misawa_combined, threshold_data, THRESHOLD_LABEL, OUTDIR, SHOW
    processed = load_tohoku(tohoku_path)
    df_misawa_metar, df_misawa_combined = load_misawa(misawa_path)
    threshold_data = threshold or {}
    THRESHOLD_LABEL = label
    SHOW = show
    if outdir:
        OUTDIR = outdir


def get_processed_annual_data(df_source, year_month_col_idx, fog_days_col_idx):
    """夏季(6-8月)の年合計。3か月そろった年のみ採用する（品質情報 1 = 欠測の月を含む年は除外）。
    均質番号は、その年の夏3か月のうち最も多い番号を採用し、
    夏の途中で番号が変わる年は mixed=True とする。"""
    df_loc = df_source.iloc[:, [year_month_col_idx, fog_days_col_idx, fog_days_col_idx + 2]].copy()
    df_loc.columns = ['year_month', 'fog_days', 'homog']
    df_loc.dropna(subset=['year_month'], inplace=True)

    df_loc['fog_days'] = pd.to_numeric(df_loc['fog_days'], errors='coerce')
    df_loc['homog'] = pd.to_numeric(df_loc['homog'], errors='coerce')
    df_loc.dropna(subset=['fog_days'], inplace=True)

    df_loc['date'] = pd.to_datetime(df_loc['year_month'], format='%Y/%m', errors='coerce')
    df_loc.dropna(subset=['date'], inplace=True)

    df_loc['year'] = df_loc['date'].dt.year
    df_loc['month'] = df_loc['date'].dt.month

    df_summer = df_loc[df_loc['month'].isin(SUMMER_MONTHS)]
    g = df_summer.groupby('year')
    annual = pd.DataFrame({
        'total_fog_days': g['fog_days'].sum(),
        'n_months': g['fog_days'].count(),
        'homog': g['homog'].agg(lambda x: x.mode().iloc[0] if x.notna().any() else float('nan')),
        'mixed': g['homog'].agg(lambda x: x.nunique() > 1),
    })
    annual = annual[annual['n_months'] == len(SUMMER_MONTHS)].drop(columns='n_months')
    return annual.reset_index()




def build_tables(sst, coast):
    """時別値と海面水温から、毎時の表 hourly と、日の表 daily (使える日の数を数えるのに使う) を作る。夏季 (6〜8月) だけ。"""
    h = fp.fog_dataset(sst, coast)
    h['day'] = (h['time'] - pd.Timedelta(hours=1)).dt.normalize()   # 0時は前日24時
    h = h[h['day'].dt.month.isin(SUMMER_MONTHS)].copy()
    h['yr'] = h['day'].dt.year
    h['gap'] = h['SST'] - h['Td']        # 海面水温 − 露点温度
    h['sat'] = h['Ta'] - h['Td']         # 気温 − 露点温度
    g = h.groupby(['station', 'day'])
    d = pd.DataFrame({'SST': g['SST'].first()})
    for c in ('Ta', 'RH', 'vis', 'Td'):
        d['n_' + c] = g[c].count()
    d = d.reset_index()
    d['yr'] = d['day'].dt.year
    d['month'] = d['day'].dt.month
    return h, d



def trim_to_first_nonzero(df):
    if not TRIM_LEADING_ZEROS:
        return df.reset_index(drop=True)
    first_idx = df[df['total_fog_days'] > 0].index.min()
    if pd.isna(first_idx):
        return df.reset_index(drop=True)
    return df.loc[first_idx:].reset_index(drop=True)


def fill_missing_years(df):
    """欠年を NaN 行として補い、行番号 = 年 - 開始年 になるようにする。"""
    years = range(int(df['year'].min()), int(df['year'].max()) + 1)
    return df.set_index('year').reindex(years).rename_axis('year').reset_index()


def build_segments(df):
    """系列を区切る。
      ・環境の変化（均質番号の変化）: 移動平均もトレンドも区切る → ▲
      ・欠測（品質情報 1 = 欠年）: 移動平均だけ切る。トレンドは同じ均質番号の間は続ける → |←欠測→|

    Returns:
        ma_segments:    移動平均用の区間 [(開始行, 終了行), ...]（欠年・環境変化で分割）
        trend_segments: トレンド用の区間（環境変化でのみ分割。欠年は含んだまま）
        env_markers:    ▲を置く x 位置（行番号ベース）
        gaps:           系列の途中にある欠年の範囲 [(最初の欠年行, 最後の欠年行), ...]
    """
    n = len(df)
    valid = df['total_fog_days'].notna().tolist()
    homog = df['homog'].tolist()
    vidx = [i for i in range(n) if valid[i]]

    # 環境変化で分けた区間（欠年をまたぐ）
    trend_segments, env_markers = [], []
    if vidx:
        start = vidx[0]
        for a_, b_ in zip(vidx[:-1], vidx[1:]):
            if homog[b_] != homog[a_]:
                trend_segments.append((start, a_))
                env_markers.append((a_ + b_) / 2)
                start = b_
        trend_segments.append((start, vidx[-1]))

    # 欠年の範囲（系列の途中のみ）
    gaps, i = [], 0
    while i < n:
        if not valid[i]:
            j = i
            while j + 1 < n and not valid[j + 1]:
                j += 1
            if i > 0 and j < n - 1:
                gaps.append((i, j))
            i = j + 1
        else:
            i += 1

    # 移動平均用: 環境変化の区間をさらに欠年で分割
    ma_segments = []
    for s_, e_ in trend_segments:
        run = None
        for k in range(s_, e_ + 1):
            if valid[k]:
                run = k if run is None else run
            elif run is not None:
                ma_segments.append((run, k - 1))
                run = None
        if run is not None:
            ma_segments.append((run, e_))
    return ma_segments, trend_segments, env_markers, gaps


def add_segment_moving_average(ax, df, segments, window=5, color='blue',
                               label='5-Year Moving Average'):
    first = True
    for s, e in segments:
        if e - s + 1 < max(window, MIN_SEGMENT_YEARS):
            continue
        seg = df.loc[s:e, 'total_fog_days']
        ma = seg.rolling(window=window, center=True, min_periods=window).mean()
        ax.plot(seg.index, ma, color=color, linewidth=2.0, label=label if first else None)
        first = False


def add_segment_trend_lines(ax, df, segments, color='red'):
    """区間ごとに年を説明変数として回帰して描く。
    返り値: [(開始行, 終了行, 開始年, 終了年, 傾き[日/年]), ...]"""
    fits = []
    for s, e in segments:
        seg = df.loc[s:e].dropna(subset=['total_fog_days'])
        if len(seg) < MIN_SEGMENT_YEARS:
            continue
        slope, intercept, *_ = linregress(seg['year'], seg['total_fog_days'])
        fits.append((s, e, int(seg['year'].iloc[0]), int(seg['year'].iloc[-1]), slope, intercept))

    for k, (s, e, y0, y1, slope, intercept) in enumerate(fits):
        idx = list(range(s, e + 1))
        line = [slope * df.loc[i, 'year'] + intercept for i in idx]
        label = 'Linear Trend' if k == 0 else None
        ax.plot(idx, line, color=color, linestyle='-', linewidth=2.5, label=label)
    return [(s, e, y0, y1, slope) for (s, e, y0, y1, slope, _) in fits]


def compute_segment_slope_labels(df, fits, extra_dfs=()):
    """各区間の傾き（Day/Decade）のラベル位置と文字列を計算する（描画はしない）。
    基本は1行。隣の数字や隣の区間の棒に重なりそうな狭い所だけ2行にする。
    戻り値: [{'x': 中心x, 'y': 文字の下端のy, 'text': 表示文字列}, ...]"""
    n = len(df)
    # 1行表示の横幅を「年」の単位で見積もる（図幅22in、軸の実効幅約20.5in、文字約2.1in）
    width_1line = 2.1 * n / 20.5
    centers = [(s + e) / 2 for s, e, *_ in fits]

    ymax_all = df['total_fog_days'].max()
    offset = 0.03 * ymax_all + 1.0
    labels = []
    for k, (s, e, y0, y1, slope) in enumerate(fits):
        c = centers[k]
        neighbors = [abs(c - centers[j]) for j in range(len(fits)) if j != k]
        too_close = any(d < width_1line * 1.05 for d in neighbors)
        out_of_axes = (c - width_1line / 2 < -0.5) or (c + width_1line / 2 > n - 0.5)
        too_wide = (e - s + 1) < width_1line      # 1行だと区間より横に広がって隣の棒に重なる
        sep = '\n' if (too_close or out_of_axes or too_wide) else ' '

        ymax = df.loc[s:e, 'total_fog_days'].max()
        for d in extra_dfs:
            if d is not None and not d.empty:
                m = d[(d['x_idx'] >= s) & (d['x_idx'] <= e)]['total_fog_days']
                if not m.empty:
                    ymax = max(ymax, m.max())
        labels.append({'x': c, 'y': ymax + offset, 'text': f'{slope * 10:+.2f}{sep}Day/Decade',
                       'two_line': '\n' in sep})
    return labels


def draw_segment_slope_labels(ax, labels, color='red'):
    for lb in labels:
        ax.text(lb['x'], lb['y'], lb['text'],
                color=color, fontsize=16, linespacing=1.1,
                ha='center', va='bottom', zorder=11,
                bbox=dict(facecolor='white', alpha=0.85, edgecolor=color, boxstyle='round,pad=0.15'))


def resolve_legend_overlap(ax, fig, labels, y_limit, pad_frac=0.025):
    """傾きラベルが凡例と重なりそうな場合、重ならなくなるまで縦軸の上限を広げる。
    戻り値: 最終的な y_limit（重なりがなければ元のまま）"""
    legend = ax.get_legend()
    if legend is None or not labels:
        return y_limit
    fig.canvas.draw()  # テキスト・凡例のサイズを確定させる
    renderer = fig.canvas.get_renderer()
    inv = ax.transAxes.inverted()
    lbbox = legend.get_window_extent(renderer)
    (lx0, ly0), (lx1, _) = inv.transform((lbbox.x0, lbbox.y0)), inv.transform((lbbox.x1, lbbox.y1))

    x0, x1 = ax.get_xlim()

    new_y_limit = y_limit
    for lb in labels:
        x_frac = (lb['x'] - x0) / (x1 - x0)
        label_half_h = 0.085 if lb['two_line'] else 0.05   # ラベル自体の高さ（軸割合の見積もり）
        if x_frac < lx0 - 0.03 or x_frac > lx1 + 0.03:
            continue  # 凡例のx範囲外なら重ならない
        y_frac = lb['y'] / y_limit
        if y_frac + label_half_h < ly0 - pad_frac:
            continue  # すでに凡例の下にある
        needed = lb['y'] / (ly0 - pad_frac - label_half_h)
        if needed > new_y_limit:
            new_y_limit = needed
    return new_y_limit


def apply_y_axis(ax, y_limit):
    step = 10 if y_limit > 30 else 5
    ax.set_ylim(0, y_limit)
    ax.set_yticks(range(0, int(y_limit + step), step))


def add_break_markers(ax, markers):
    """環境の変化（均質番号の変化）を横軸上の▲で示す。"""
    for k, x in enumerate(markers):
        ax.plot([x], [0], marker='^', color='darkorange', markeredgecolor='black',
                markersize=17, linestyle='none', clip_on=False, zorder=10,
                label=ENV_LABEL if k == 0 else None)      # 凡例には1つだけ載せる


def add_gap_spans(ax, gaps, df, color='black'):
    """欠測期間を  |←Missing data→|  で示す。y は軸の割合で指定。
    欠測の幅が文字より狭いときは、文字を棒の上に出して引き出し線でつなぐ。"""
    tr = ax.get_xaxis_transform()          # x: データ座標, y: 軸の割合
    n = len(df)
    text_width_years = 1.65 * n / 20.5     # 文字の横幅の見積もり（年）
    vals = df['total_fog_days']
    ymax_all = vals.max()
    for i, j in gaps:
        x0, x1 = i - 0.5, j + 0.5
        y = 0.07
        ax.annotate('', xy=(x1, y), xytext=(x0, y), xycoords=tr, textcoords=tr,
                    arrowprops=dict(arrowstyle='<->', color=color, lw=2.2,
                                    shrinkA=0, shrinkB=0), zorder=10)
        ax.plot([x0, x1], [y, y], transform=tr, marker='|', markersize=20,
                markeredgewidth=2.2, color=color, linestyle='none', zorder=10)
        mid = (x0 + x1) / 2
        if (j - i + 1) >= text_width_years:
            ax.text(mid, y + 0.03, 'Missing data', transform=tr, ha='center', va='bottom',
                    fontsize=18, color=color, zorder=11,
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1.5))
        else:
            # 欠測の年は棒がないので、その真上に引き出し線を通せる
            lo, hi = max(0, i - int(text_width_years / 2) - 1), min(n, j + int(text_width_years / 2) + 2)
            local_max = vals.iloc[lo:hi].max()
            ty = local_max + 0.05 * ymax_all + 1.0
            ax.annotate('Missing data', xy=(mid, y + 0.02), xycoords=tr, xytext=(mid, ty),
                        textcoords='data', ha='center', va='bottom', fontsize=18, color=color,
                        zorder=11,
                        bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1.5),
                        arrowprops=dict(arrowstyle='-', color=color, lw=1.2, shrinkA=0, shrinkB=0))


def build_legend(ax, priority=None, ncol=1):
    # ncol=2 のとき matplotlib は列優先で埋める。
    # 八戸の [MA, Trend, ▲, Combined, METAR] は 左列: MA・Trend・▲ / 右列: METAR+SPECIAL(実線)・METAR(破線) になる。
    handles, labels = ax.get_legend_handles_labels()
    if priority:
        order = []
        for p in priority:
            idx = next((i for i, l in enumerate(labels) if l.startswith(p)), None)
            if idx is not None:
                order.append(idx)
        order += [i for i in range(len(labels)) if i not in order]
    else:
        order = list(range(len(labels)))

    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              fontsize=18, loc='upper right', framealpha=0.9, ncol=ncol, markerscale=0.7)


def decorate_axis(ax, panel_label, years_to_plot, max_val, legend_priority=None, ncol=1):
    ax.text(0.015, 0.93, panel_label, transform=ax.transAxes, fontsize=32,
            ha='left', va='top', color='black')

    ax.set_xlabel('Year', fontsize=28)
    ax.set_ylabel('Days', fontsize=28)

    y_limit = max(max_val * 1.4, max_val + 15)
    apply_y_axis(ax, y_limit)
    ax.tick_params(axis='y', labelsize=26)
    ax.set_xlim(-0.5, len(years_to_plot) - 0.5)

    # 10年ごとの目盛り + 最初/最後の年（近すぎて重なる場合は端の年を省略）
    n = len(years_to_plot)
    tick_positions = [i for i, y in enumerate(years_to_plot) if y % 10 == 0]
    for edge in (0, n - 1):
        if all(abs(edge - p) >= 4 for p in tick_positions):
            tick_positions.append(edge)
    tick_positions = sorted(set(tick_positions))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(years_to_plot[i]) for i in tick_positions],
                       rotation=0, ha='center', fontsize=25)
    ax.grid(True, alpha=0.3)
    build_legend(ax, priority=legend_priority, ncol=ncol)
    return y_limit


def overlay_misawa(ax, df_misawa, year_to_idx, color='black', **plot_kwargs):
    d = df_misawa.copy()
    d['x_idx'] = d['year'].map(year_to_idx)
    d = d.dropna(subset=['x_idx']).sort_values('year')
    if not d.empty:
        ax.plot(d['x_idx'], d['total_fog_days'], color=color, markersize=6, linewidth=2, **plot_kwargs)
    return d


trend_results = {}


def draw_location(loc_name):
    df_loc = trim_to_first_nonzero(processed[loc_name])
    if df_loc.empty:
        print(f"{loc_name}: データなし")
        return
    df_loc = fill_missing_years(df_loc)          # index == year - 開始年
    years_to_plot = df_loc['year'].tolist()
    year_to_idx = {year: idx for idx, year in enumerate(years_to_plot)}

    ma_segments, trend_segments, env_markers, gaps = build_segments(df_loc)
    trend_segments = adjust_trend_segments(df_loc, loc_name, trend_segments)
    ma_segments = adjust_ma_segments(ma_segments, trend_segments)
    mixed_years = df_loc.loc[df_loc['mixed'] == True, 'year'].astype(int).tolist()
    if mixed_years:
        print(f"{loc_name}: 夏の途中で均質番号が切り替わる年（多数派の区間に含める）->", mixed_years)

    fig, ax = plt.subplots(figsize=(22, 6.2))

    ax.bar(df_loc.index, df_loc['total_fog_days'], color='limegreen', edgecolor='black', width=0.9)

    add_segment_moving_average(ax, df_loc, ma_segments, window=5, color='blue')
    fits = add_segment_trend_lines(ax, df_loc, trend_segments, color='red')
    trend_results[loc_name] = [(y0, y1, sl) for (_, _, y0, y1, sl) in fits]
    add_break_markers(ax, env_markers)
    add_gap_spans(ax, gaps, df_loc)

    max_val = df_loc['total_fog_days'].max()
    overlays = []
    priority_labels = ['5-Year Moving Average', 'Linear Trend']   # ENV_LABEL は最後に追加する（線の凡例をまとめるため）

    # 凡例の並び: 八戸は2列になるので、左列に MA・Trend・しきい値日数、右列に三沢（Misawa）2本 + ▲ が来るようにする
    if loc_name in threshold_data:
        th = overlay_misawa(ax, threshold_data[loc_name], year_to_idx,
                            marker='D', linestyle='-', label=THRESHOLD_LABEL)
        overlays.append(th)
        priority_labels.append(THRESHOLD_LABEL)

    if loc_name == '八戸':
        m_metar = overlay_misawa(ax, df_misawa_metar, year_to_idx, color='darkmagenta',
                                 marker='o', linestyle='--', label='Misawa (METAR)')
        m_comb = overlay_misawa(ax, df_misawa_combined, year_to_idx, color='darkmagenta',
                                marker='s', linestyle='-', label='Misawa (METAR+SPECIAL)')
        overlays += [m_metar, m_comb]
        priority_labels += ['Misawa (METAR+SPECIAL)', 'Misawa (METAR)']

    priority_labels.append(ENV_LABEL)

    if overlays:
        max_val = max([max_val] + [d['total_fog_days'].max() for d in overlays if not d.empty])

    ncol = 2 if loc_name == '八戸' else 1
    y_limit = decorate_axis(ax, f'{loc_name}', years_to_plot, max_val,
                            legend_priority=priority_labels, ncol=ncol)

    # 傾きラベルが凡例と重なる場合は、重ならなくなるまで縦軸の上限を広げてから描く
    slope_labels = compute_segment_slope_labels(df_loc, fits, extra_dfs=overlays)
    y_limit_adjusted = resolve_legend_overlap(ax, fig, slope_labels, y_limit)
    if y_limit_adjusted > y_limit:
        apply_y_axis(ax, y_limit_adjusted)
    draw_segment_slope_labels(ax, slope_labels, color='red')

    plt.tight_layout()

    if SAVE_FIG:
        out_path = os.path.join(OUTDIR, f"fog_days_{loc_name}.png")
        fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches='tight')
        print("保存:", out_path)
    if SHOW:
        plt.show()
    plt.close(fig)


def draw_all(locations):
    """locations の地点を1枚ずつ描き (画像は OUTDIR に保存)、トレンドを表示する"""
    os.makedirs(OUTDIR, exist_ok=True)
    trend_results.clear()
    for loc_name in locations:
        draw_location(loc_name)

    # =========================================================
    # 5. 変化量の結果表示（区間ごと・符号付き）
    # =========================================================
    print("\n--- 分析結果（日/10年）---")
    for loc_name in locations:
        fits = trend_results.get(loc_name)
        if not fits:
            print(f"{loc_name}: 計算不可（データ不足）")
        else:
            parts = [f"{y0}-{y1}: {s * 10:+.2f}" for (y0, y1, s) in fits]
            print(f"{loc_name}: " + " / ".join(parts))


def count_days(mask, hourly, daily, min_hours=1, months=(6, 7, 8), need=('n_Ta', 'n_RH')):
    """mask: 毎時の表 hourly の各行が条件を満たすなら True。
    条件を満たす時刻が min_hours 時間以上ある日を「該当日」とし、months の月だけを年ごとに数える。
    返り値: {地点: DataFrame(year, total_fog_days)}"""
    mask = mask.fillna(False).astype(bool)
    n = mask.groupby([hourly['station'], hourly['day']]).sum()
    hit = n[n >= min_hours].reset_index()[['station', 'day']]
    hit = hit[hit['day'].dt.month.isin(months)]
    hit['yr'] = hit['day'].dt.year
    usable = daily[daily['month'].isin(months)].dropna(subset=['SST'])
    usable = usable[usable[[c for c in need if c in usable.columns]].min(axis=1) > 0]
    n_days = usable.groupby(['station', 'yr']).size()          # その年に使える日数
    need = MIN_DAYS_FRAC * sum(pd.Period(f'2001-{m:02d}').days_in_month for m in months)
    cnt = hit.groupby(['station', 'yr']).size().reindex(n_days.index, fill_value=0)
    cnt = cnt[(n_days >= need) & (cnt.index.get_level_values('yr') >= COND_FROM_YEAR)]
    return {s: v.reset_index(level=0, drop=True).rename_axis('year').rename('total_fog_days').reset_index()
            for s, v in cnt.groupby(level=0)}


def compare_table(threshold_data):
    """霧日数(最新の均質区間)と黒線の比較表"""
    rows = []
    for s, th in threshold_data.items():
        if s not in processed:
            continue
        fog = processed[s].dropna(subset=['total_fog_days'])
        if fog.empty:
            continue
        fog = fog[fog['homog'] == fog['homog'].max()]
        m = fog.merge(th, on='year', suffixes=('_fog', '_cond'))
        if len(m) < 3:
            continue
        diff = m['total_fog_days_cond'] - m['total_fog_days_fog']
        rows.append({'地点': s, '期間': f"{m['year'].min()}-{m['year'].max()}", '年数': len(m),
                     '霧日数平均': round(m['total_fog_days_fog'].mean(), 1),
                     '黒線平均': round(m['total_fog_days_cond'].mean(), 1),
                     '差の大きさ(RMSE)': round(float(np.sqrt((diff ** 2).mean())), 1),
                     '相関r': round(m['total_fog_days_fog'].corr(m['total_fog_days_cond']), 2)})
    return pd.DataFrame(rows)
