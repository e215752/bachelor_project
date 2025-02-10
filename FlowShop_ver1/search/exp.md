- スケジューリング問題の実験結果
   - job5のとき
      A = trial.suggest_float("A", 1000, 2000)
      B = trial.suggest_float("B", 180, 220)
      C = trial.suggest_float("C", 250, 350)
      D = trial.suggest_float("D", 1, 10)
      E = trial.suggest_float("E", 1, 10)

      param_initial_values = {
         "A": 1500,
         "B": 200,
         "C": 300,
         "D": 1,
         "E": 1
      }
   - job6のとき
      A = trial.suggest_float("A", 1000, 2000)
      B = trial.suggest_float("B", 180, 220)
      C = trial.suggest_float("C", 250, 350)
      D = trial.suggest_float("D", 1, 10)
      E = trial.suggest_float("E", 1, 10)

      param_initial_values = {
         "A": 1008.83,
         "B": 180.95,
         "C": 331.45,
         "D": 4.94,
         "E": 1.04
      }
