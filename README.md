## リポジトリ内のファイルの構成
```
リポジトリTOP
│
├ academic .. 学会用
│　├ poster.tex
│　└ fig
│
├ FlowShop_ver1 .. 手動でパラメータ調整を行ったコード
│  ├ FlowShop_ver1.py .. あるパラメータの時のデータを取得するコード
│  ├ draw.py .. FlowShop_ver1.pyを実行した後にplotするコード
│  ├ FlowShop_ver1.ipynb .. FlowShop_ver1.pyとdraw.pyを一つにまとめた
│  └ search .. データ取得したやつを日付毎にまとめてる
│    ├ exp.md ハイパーパラメータのパターンの詳細
│    ├ t_12 .. time=12としたときのデータ
│    ├ t_30 .. time=30としたときのデータ
│    └ plots .. plotした図
├ FlowShop_ver2 .. ver1をupdateしたやつ(改良中)
```

---

**注意事項**

- pyquboはpython 3.6から3.10までしかポートしてない
- openjijはpython 3.8以上

必要だったら[pyenv](https://qiita.com/koooooo/items/b21d87ffe2b56d0c589b)からlocalの方使って設定して