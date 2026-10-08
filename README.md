# 東北〜津軽海峡沿岸 海面水温の推移

12海域の日別海面水温(1982〜、気象庁形式 `*.txt`)を解析するツール。
参考: [kanoto-amedas](https://github.com/awg-yk/kanoto-amedas)

```
pip install numpy pandas matplotlib pillow
python sst_tool.py summary   # 海域別の年平均・トレンド(℃/10年) → out/*.csv
python sst_tool.py plot      # 年平均推移・平年偏差・年×日ヒートマップ → out/*.png
python sst_tool.py viewer                     # 夏季の地図を年・月・日ボタンで操作
python sst_tool.py map --date 2025-08-15      # ある日の海水温を東北の地図+等温線で表示 (out/にPNGも保存)
python sst_tool.py player --start 2025-06-01 --end 2025-09-30   # 期間を再生
python sst_tool.py player --start 2025-06-01 --end 2025-09-30 --step 3 --save out/player.gif
```

- **夏季(6/1〜8/31)だけ**: `python sst_tool.py --summer summary`(plot / map / player にも付けられます)。
- **`viewer`**: 夏季のみの地図を、年・月・日・時のボタン(年−月−日−時−時+日+月+年+の順)、再生/停止、スライダー(最初は左端=最も古い時刻)で操作。
  海面水温(日別)の等温線に、陸上の官署(時別)の気温(四角の色)・風(矢印)・湿度・視程を重ねます。
  官署のCSV(気象庁「時別値」、年ごと)は `東北地方官署アメダスデータ` フォルダ(`--obs-dir` で変更)から全部読みます。
  海と陸の気温は同じ色スケールです。風が静穏の官署は小さな○。視程は観測点を中心にした半径=視程(km)の円で表示(実距離)。1km未満は紫の枠+赤字(霧の目安)。
  キー操作: ←→=時、↑↓=日、PageUp/PageDown=月、[ ]=年、1〜4=気温/風/湿度/視程の表示切替、スペース=再生。
  古い年は3時間おきなど観測のある時刻だけを順にたどります。

- **`fog`**: 沿岸10官署(`東北地方沿岸気象官署時別値`)の 気温・相対湿度・視程 と、近い海域の海面水温から、
  霧(視程<1km)が出る時の「気温−海面水温」「相対湿度」を調べます。`python sst_tool.py --summer fog [--from-year 2020]`
  → 画面に集計表、`out/fog_*.png` と `out/fog_*.csv`(`fog_plot.py` は単独で動き、官署別の図だけを保存せず画面に表示)。官署と海域の対応は `STATION_SST`。視程は2020年から毎時で、
  それ以前は間引かれている(観測時刻が偏る)ので、`--from-year 2020` との比較を勧めます。

- 平年値は 1991〜2020 年(日別・7日平滑)。年平均は欠測30日超の年と今年(途中)を除外。
- flag `P`(速報値)も含めて使用。
- `map`/`player` の海岸線は国土地理院の標高タイルから作ります(初回のみ通信、`cache/` に保存)。通信できない時は海岸線なしで描画。
- 等温線は12海域の代表値を逆距離加重で補間したものです。海域の座標は概算値(`AREAS`)。`--reach` で塗る範囲、`--interval-deg` で等温線の間隔を変えられます。
