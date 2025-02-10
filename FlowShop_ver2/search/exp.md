- スケジューリング問題の実験結果
   - taskの処理時間優先の場合
    - job5のとき
        A = trial.suggest_float("A", 650, 950)
        B = trial.suggest_float("B", 150, 250)
        C = trial.suggest_float("C", 200, 300)
        E = trial.suggest_float("E", 5, 20)

    - job6のとき
        A = trial.suggest_float("A", 500, 1500)
        B = trial.suggest_float("B", 150, 300)
        C = trial.suggest_float("C", 150, 300)
        E = trial.suggest_float("E", 2, 20)

    - job7のとき
        A = trial.suggest_float("A", 500, 1500)
        B = trial.suggest_float("B", 150, 300)
        C = trial.suggest_float("C", 150, 300)
        E = trial.suggest_float("E", 2, 20)

   - リソースコスト優先の場合
    - job5のとき
        A = trial.suggest_float("A", 700, 900)
        B = trial.suggest_float("B", 150, 300)
        C = trial.suggest_float("C", 100, 300)
        E = trial.suggest_float("E", 15, 25)

    - job6のとき
        A = trial.suggest_float("A", 500, 1500)
        B = trial.suggest_float("B", 150, 300)
        C = trial.suggest_float("C", 150, 300)
        E = trial.suggest_float("E", 2, 20)

    - job7のとき
        A = trial.suggest_float("A", 500, 1500)
        B = trial.suggest_float("B", 150, 300)
        C = trial.suggest_float("C", 150, 300)
        E = trial.suggest_float("E", 2, 20)

規模に応じてパラメータの範囲を広げる(ずらす、大きくするなど)する必要があった