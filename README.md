## リポジトリ内のファイルの構成
```
リポジトリTOP
│
├ academic .. 学会用
│　├ poster.tex
│　└ fig
│
├ FlowShop_ver1 .. モデルの分割前のコード
│  ├ FlowShop_ver1.ipynb
│  └ search .. データ取得したやつを日付毎にまとめてる
│    ├ exp.md ハイパーパラメータのパターンの詳細
│    ├ t_12 .. time=12としたときのデータ
│    ├ t_30 .. time=30としたときのデータ
│    └ plots .. plotした図
│
└  FlowShop_ver2 .. モデルの分割後のコード
   ├ FSP.cpn .. cpntoolsでモデル表現したファイル
   ├ FSP.xml .. cpntoolsでモデル表現したxmlファイル
   ├ FlowShop_ver2.ipynb
   └ search .. データ取得したやつを日付毎にまとめてる

```

---

**注意事項**

- pyquboはpython 3.6から3.10までしかポートしてない
- openjijはpython 3.8以上

必要だったら[pyenv](https://qiita.com/koooooo/items/b21d87ffe2b56d0c589b)からlocalの方使って設定して