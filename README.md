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
│    ├ job5 .. jobが5つの時のデータ
│    ├ job6 .. jobが6つの時のデータ
│    └ plots .. plotした図
│
└ FlowShop_ver2 .. モデルの分割後のコード
   ├ FSP.cpn .. cpntoolsでモデル表現したファイル
   ├ FSP.xml .. cpntoolsでモデル表現したxmlファイル
   ├ xml.ipynb .. xmlファイルから必要なデータを取得するコード
   ├ FlowShop_ver2.ipynb
   └ search .. データ取得したやつを日付毎にまとめてる
     ├ exp.md ハイパーパラメータのパターンの詳細
     ├ job5 .. jobが5つの時のデータ
     ├ job6 .. jobが6つの時のデータ
     ├ job7 .. jobが7つの時のデータ
     └ plots .. plotした図

```

---

**注意事項**

- pyquboはpython 3.6から3.10までしかポートしてない
- openjijはpython 3.8以上

必要だったら[pyenv](https://qiita.com/koooooo/items/b21d87ffe2b56d0c589b)からlocalの方使って設定して