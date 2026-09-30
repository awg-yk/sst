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
- **`viewer`**: 夏季のみの地図を、年・月・日のボタン(年−/+、月−/+、日−/+)、再生/停止、日付スライダーで操作。
  キー操作: ←→=日、↑↓=年、PageUp/PageDown=月、スペース=再生。データは日別なので時間の操作はありません。

- 平年値は 1991〜2020 年(日別・7日平滑)。年平均は欠測30日超の年と今年(途中)を除外。
- flag `P`(速報値)も含めて使用。
- `map`/`player` の海岸線は国土地理院の標高タイルから作ります(初回のみ通信、`cache/` に保存)。通信できない時は海岸線なしで描画。
- 等温線は12海域の代表値を逆距離加重で補間したものです。海域の座標は概算値(`AREAS`)。`--reach` で塗る範囲、`--interval-deg` で等温線の間隔を変えられます。
