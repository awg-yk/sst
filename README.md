# 東北〜津軽海峡沿岸 海面水温の推移

12海域の日別海面水温(1982〜、気象庁形式 `*.txt`)を解析するツール。
参考: [kanoto-amedas](https://github.com/awg-yk/kanoto-amedas)

```
pip install numpy pandas matplotlib pillow
python sst_tool.py summary   # 海域別の年平均・トレンド(℃/10年) → out/*.csv
python sst_tool.py plot      # 年平均推移・平年偏差・年×日ヒートマップ → out/*.png
python sst_tool.py player --start 2020-01-01 --end 2020-12-31 --save out/player.gif
```

- 平年値は 1991〜2020 年(日別・7日平滑)。年平均は欠測30日超の年と今年(途中)を除外。
- flag `P`(速報値)も含めて使用。
- `player` の海域座標は概算値(`sst_tool.py` の `AREAS`)。
