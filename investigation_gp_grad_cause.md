# 調査メモ: `solver=gp`でも`grad routine`が表示される件

## 結論
原因は主に2つです。

1. **ログ文言が固定で`[grad ...]`になっている**
   - `optBySearch()`内の進行ログが、`SEARCH_METHOD`に関係なく`[grad routine]`/`[grad budget]`を出力する実装になっています。
   - そのため、`gp`を選んでもログ上は`grad routine`と表示されます。

2. **`optBySearch()`が期待する戻り値形式と`lib_gp.GaussianProcess.search()`の戻り値形式が不一致**
   - `optBySearch()`は `info['evaluated_rows']` を参照して新規評価結果を履歴に追加する前提です。
   - しかし `lib_gp.GaussianProcess.search()` は `{"acq": ..., "length_scale": ...}` を返し、`evaluated_rows` を返しません。
   - 結果として `evaluated_rows` が空配列扱いになり、`"optimizer produced no new evaluations"` 分岐に入りやすい構造です。

## 補足
- `SEARCH_METHOD` の初期値はノートブック内で `"random"` に設定されています。
- ノートブック実行順によっては、`SEARCH_METHOD`を書き換えたつもりでも、`optimizer = build_optimizer(...)` を再実行していないと古いoptimizerが残る可能性があります。

