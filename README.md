## リポジトリ内のファイルの構成
```
リポジトリTOP
│
├ academic .. 学会用
│  ├ poster.tex
│  └ fig
│
├ FlowShop_ver1 .. 手動でパラメータ調整を行ったコード
│  ├ FlowShop_ver1.py .. あるパラメータの時のデータを取得するコード
│  ├ draw.py .. FlowShop_ver1.pyを実行した後にplotするコード
│  ├ FlowShop_ver1.ipynb .. FlowShop_ver1.pyとdraw.pyを一つにまとめた
│  └ search .. データ取得したやつ
│    ├ exp.md ハイパーパラメータのパターンの詳細
│    ├ plots plotした日付毎にまとめてる
│    ├ t_12 timeを12で設定した時の取得したデータ
│    └ t_30 timeを30で設定した時の取得したデータ(ここがメイン)
├ 
```

---

**注意事項**

- pyquboはpython 3.6から3.10までしかポートしてない
- openjijはpython 3.8以上

必要だったら[pyenv](https://qiita.com/koooooo/items/b21d87ffe2b56d0c589b)からlocalの方使って設定して
