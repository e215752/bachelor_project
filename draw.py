#平均と分散をplot
import pandas as pd
import numpy as np
import re
import glob
import matplotlib.pyplot as plt

# パターン1から7までのファイルパスを取得
file_paths = glob.glob("./search/2024-08-22_patt*_job15.csv")

# 結果を保存するリスト
all_results = []

# 各ファイルに対して処理を適用
for file_path in file_paths:
    # ファイル名からパターン番号を抽出
    patt_number = int(re.search(r'patt(\d+)', file_path).group(1))
    
    # CSVファイルを読み込む
    df = pd.read_csv(file_path)
    
    # feasibleが0の行のみを選択（文字列'0'も含む）
    df_feasible_0 = df[df['feasible'].astype(str) == '0']
    
    # resource_cost, wating_time, feasible列を数値に変換
    df_feasible_0['resource_cost'] = pd.to_numeric(df_feasible_0['resource_cost'], errors='coerce')
    df_feasible_0['wating_time'] = pd.to_numeric(df_feasible_0['wating_time'], errors='coerce')
    df_feasible_0['feasible'] = pd.to_numeric(df_feasible_0['feasible'], errors='coerce')
    
    # 数値に変換できなかったデータを除外
    df_feasible_0 = df_feasible_0.dropna(subset=['resource_cost', 'wating_time', 'feasible'])
    
    # iteration毎にresource_costとwating_timeの平均と分散を計算
    summary = df_feasible_0.groupby('iteration').agg(
        mean_resource_cost=('resource_cost', 'mean'),
        var_resource_cost=('resource_cost', 'var'),
        mean_wating_time=('wating_time', 'mean'),
        var_wating_time=('wating_time', 'var'),
    ).reset_index()
    
    # パターン番号を追加
    summary['patt'] = patt_number
    
    # パターン番号とiterationを結合して新しいラベルを作成
    summary['label'] = summary['patt'].astype(str) + "_" + summary['iteration'].astype(str)
    
    # リストに追加
    all_results.append(summary)
    
# 全パターンの結果を結合
final_df = pd.concat(all_results)

# パターンごとに結果をソート
final_df = final_df.sort_values(by=['patt', 'iteration'])

print(final_df)

# resource_costをプロット
plt.figure(figsize=(12, 6))

# 分散のエラーバーを計算
err_rc = np.sqrt(final_df['var_resource_cost'])

plt.errorbar(final_df['label'], final_df['mean_resource_cost'], 
             yerr=err_rc,
             fmt='o-', color='blue', capsize=5, label='Resource Cost')

plt.xlabel('Pattern_Iteration')
plt.ylabel('Resource Cost')
plt.title('Average and Variance of Resource Cost by Pattern and Iteration')
plt.xticks(rotation=45)  # ラベルを45度傾けて表示
plt.legend()
plt.grid(True)
plt.show()

# wating_timeをプロット
plt.figure(figsize=(12, 6))

# 分散のエラーバーを計算
err_wt = np.sqrt(final_df['var_wating_time'])

plt.errorbar(final_df['label'], final_df['mean_wating_time'], 
             yerr=err_wt,
             fmt='o-', color='orange', capsize=5, label='Wating Time')

plt.xlabel('Pattern_Iteration')
plt.ylabel('Wating Time')
plt.title('Average and Variance of Wating Time by Pattern and Iteration')
plt.xticks(rotation=45)  # ラベルを45度傾けて表示
plt.legend()
plt.grid(True)
plt.show()

#パレートフロントをplot
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

# パターン1から7までのファイルパスを取得
file_paths = glob.glob("./search/2024-08-23_patt*_job15.csv")

# プロットの設定
plt.figure(figsize=(10, 6))

def identify_pareto(waiting_times, resource_costs, epsilon=1e-6):
    pareto_indices = []
    for i in range(len(waiting_times)):
        is_pareto = True
        for j in range(len(waiting_times)):
            if i != j:
                if (waiting_times[i] > waiting_times[j] + epsilon and 
                    resource_costs[i] < resource_costs[j] + epsilon):
                    is_pareto = False
                    break
        if is_pareto:
            pareto_indices.append(i)
    return pareto_indices

# 各ファイルに対して処理を適用
all_resource_costs = []
all_waiting_times = []

for file_path in file_paths:
    df = pd.read_csv(file_path)
    df_feasible_0 = df[df['feasible'].astype(str) == '0']
    
    df_feasible_0['resource_cost'] = pd.to_numeric(df_feasible_0['resource_cost'], errors='coerce')
    df_feasible_0['wating_time'] = pd.to_numeric(df_feasible_0['wating_time'], errors='coerce')
    
    df_feasible_0 = df_feasible_0.dropna(subset=['resource_cost', 'wating_time'])
    
    # 重複を除去
    df_feasible_0 = df_feasible_0.drop_duplicates(subset=['resource_cost', 'wating_time'])
    
    all_resource_costs.extend(df_feasible_0['resource_cost'].tolist())
    all_waiting_times.extend(df_feasible_0['wating_time'].tolist())

# パレート効率解のインデックスを取得
pareto_indices = identify_pareto(all_waiting_times, all_resource_costs)

# パレート効率解とその他の解に分ける
pareto_waiting_times = [all_waiting_times[i] for i in pareto_indices]
pareto_resource_costs = [all_resource_costs[i] for i in pareto_indices]
non_pareto_waiting_times = [wt for i, wt in enumerate(all_waiting_times) if i not in pareto_indices]
non_pareto_resource_costs = [rc for i, rc in enumerate(all_resource_costs) if i not in pareto_indices]

# 全てのデータポイントをプロット
plt.scatter(non_pareto_waiting_times, non_pareto_resource_costs, color='blue', alpha=0.7, label='Non-Pareto')
plt.scatter(pareto_waiting_times, pareto_resource_costs, color='red', alpha=0.7, label='Pareto Efficient')

# グラフの設定
# plt.title('Waiting Time vs Resource Cost')
plt.xlabel('Waiting Time')
plt.ylabel('Resource Cost')
plt.grid(True)
plt.legend()

print(f"Total data points: {len(all_waiting_times)}")
print(f"Pareto efficient solutions: {len(pareto_waiting_times)}")

# グラフを表示
plt.show()

#パラメータDを固定しEの変えていき平均の値をplot
import pandas as pd
import numpy as np
import re
import glob
import matplotlib.pyplot as plt

# パターン7から12までのファイルパスを取得
file_paths = glob.glob("./search//t_30/0823/2024-08-23_patt*_job15.csv")
file_paths = [f for f in file_paths if re.search(r'patt[1-6]_job15\.csv$', f)]              #D=10
# file_paths = [f for f in file_paths if re.search(r'patt[7-9]|10|11|12_job15\.csv$', f)]     #D=8
# file_paths = [f for f in file_paths if re.search(r'patt13|14|15|16|17|18_job15\.csv$', f)]  #D=5

# 結果を保存するリスト
all_results = []

# 各ファイルに対して処理を適用
for file_path in file_paths:
    # ファイル名からパターン番号を抽出
    patt_number = int(re.search(r'patt(\d+)', file_path).group(1))
    
    # CSVファイルを読み込む
    df = pd.read_csv(file_path)
    
    # feasibleが0の行のみを選択（文字列'0'も含む）
    df_feasible_0 = df[df['feasible'].astype(str) == '0']
    
    # resource_cost, wating_time, feasible列を数値に変換
    df_feasible_0['resource_cost'] = pd.to_numeric(df_feasible_0['resource_cost'], errors='coerce')
    df_feasible_0['wating_time'] = pd.to_numeric(df_feasible_0['wating_time'], errors='coerce')
    df_feasible_0['feasible'] = pd.to_numeric(df_feasible_0['feasible'], errors='coerce')
    
    # 数値に変換できなかったデータを除外
    df_feasible_0 = df_feasible_0.dropna(subset=['resource_cost', 'wating_time', 'feasible'])
    
    # iteration毎にresource_costとwating_timeの平均を計算
    summary = df_feasible_0.groupby('iteration').agg(
        mean_resource_cost=('resource_cost', 'mean'),
        mean_wating_time=('wating_time', 'mean')
    ).reset_index()
    
    # パターン番号を追加
    summary['patt'] = patt_number
    
    # パターン番号とiterationを結合して新しいラベルを作成
    summary['label'] = summary['patt'].astype(str) + "_" + summary['iteration'].astype(str)
    
    # リストに追加
    all_results.append(summary)
    
# 全パターンの結果を結合
final_df = pd.concat(all_results)

# パターンごとに結果をソート
final_df = final_df.sort_values(by=['patt', 'iteration'])

# CSVファイルとして保存
final_df.to_csv('./search/final_summary_1_to_6.csv', index=False)
# final_df.to_csv('./search/final_summary_7_to_12.csv', index=False)
# final_df.to_csv('./search/final_summary_13_to_18.csv', index=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSVファイルを読み込む
final_df = pd.read_csv('./search/t_30/0823/final_summary_13_to_18.csv')

# プロットの作成
fig, ax1 = plt.subplots(figsize=(12, 6))

# 左の縦軸にresource_costをプロット
color = 'tab:blue'
ax1.set_xlabel('parameter E')
ax1.set_ylabel('Resource Cost', color=color)
ax1.plot(final_df['label'], final_df['mean_resource_cost'], marker='o', linestyle='-', color=color, label='Resource Cost')
ax1.tick_params(axis='y', labelcolor=color)

# 右の縦軸を作成
ax2 = ax1.twinx()  # 右の縦軸を作成
color = 'tab:orange'
ax2.set_ylabel('Wating Time', color=color)
ax2.plot(final_df['label'], final_df['mean_wating_time'], marker='x', linestyle='--', color=color, label='Wating Time')
ax2.tick_params(axis='y', labelcolor=color)

# X軸のラベルをカスタム設定
custom_xticks = ['50', '40', '30', '20', '10', '1']  # カスタムの横軸ラベル
plt.xticks(ticks=np.arange(len(custom_xticks)), labels=custom_xticks, rotation=45)

# プロットのタイトルとラベル
fig.suptitle('Average Resource Cost and Wating Time by Pattern and Iteration')

# 凡例の追加
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# グリッドの表示
ax1.grid(True)

plt.show()