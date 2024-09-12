#平均と分散をplot
import pandas as pd
import numpy as np
import re
import glob
import matplotlib.pyplot as plt

# 全パターンのファイルパスを取得
file_paths = glob.glob("./search/t_30/0822/2024-08-22_patt*_job15.csv") #任意のファイル名
# file_paths = glob.glob("./search/t_30/2024-08-22_patt*_job15.csv")

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
    df_feasible_0['waiting_time'] = pd.to_numeric(df_feasible_0['waiting_time'], errors='coerce')
    df_feasible_0['feasible'] = pd.to_numeric(df_feasible_0['feasible'], errors='coerce')
    
    # 数値に変換できなかったデータを除外
    df_feasible_0 = df_feasible_0.dropna(subset=['resource_cost', 'waiting_time', 'feasible'])
    
    # iteration毎にresource_costとwating_timeの平均と分散を計算
    summary = df_feasible_0.groupby('iteration').agg(
        mean_resource_cost=('resource_cost', 'mean'),
        var_resource_cost=('resource_cost', 'var'),
        mean_waiting_time=('waiting_time', 'mean'),
        var_waiting_time=('waiting_time', 'var'),
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
err_wt = np.sqrt(final_df['var_waiting_time'])

plt.errorbar(final_df['label'], final_df['mean_waiting_time'], 
             yerr=err_wt,
             fmt='o-', color='orange', capsize=5, label='Waiting Time')

plt.xlabel('Pattern_Iteration')
plt.ylabel('Waiting Time')
plt.title('Average and Variance of Waiting Time by Pattern and Iteration')
plt.xticks(rotation=45)  # ラベルを45度傾けて表示
plt.legend()
plt.grid(True)
plt.show()

#パレートフロントをplot
import pandas as pd
import re
import matplotlib.pyplot as plt
import glob

# ファイルパスのリストを取得
file_paths = glob.glob("./search/t_30/0901/2024-09-01_patt*_job15.csv")
# file_paths = [f for f in file_paths if re.search(r'patt([1-9]|1[0-9]|2[0-4])_job15\.csv$', f)]

# パレート効率解を識別する関数
def identify_pareto(waiting_times, resource_costs, epsilon=1e-6):
    pareto_indices = []
    for i in range(len(waiting_times)):
        is_pareto = True
        for j in range(len(waiting_times)):
            if i != j:
                if (waiting_times[i] >= waiting_times[j] - epsilon and 
                    resource_costs[i] >= resource_costs[j] - epsilon and 
                    (waiting_times[i] > waiting_times[j] + epsilon or 
                     resource_costs[i] > resource_costs[j] + epsilon)):
                    is_pareto = False
                    break
        if is_pareto:
            pareto_indices.append(i)
    return pareto_indices

# データフレームを格納するリスト
dfs = []

# 各ファイルを読み込み、データフレームリストに追加
for file_path in file_paths:
    df = pd.read_csv(file_path)
    
    # feasibleが0の行のみを選択（文字列'0'も含む）
    df_feasible_0 = df[df['feasible'].astype(str) == '0']
    
    # resource_cost, waiting_time, feasible列を数値に変換
    df_feasible_0['resource_cost'] = pd.to_numeric(df_feasible_0['resource_cost'], errors='coerce')
    df_feasible_0['waiting_time'] = pd.to_numeric(df_feasible_0['waiting_time'], errors='coerce')
    df_feasible_0['feasible'] = pd.to_numeric(df_feasible_0['feasible'], errors='coerce')
    
    # 数値に変換できなかったデータを除外
    df_feasible_0 = df_feasible_0.dropna(subset=['resource_cost', 'waiting_time', 'feasible'])
    
    # パラメータパターンを追加（ファイル名から抽出）
    pattern_match = re.search(r'patt(\d+)_job15', file_path)
    if pattern_match:
        pattern_number = int(pattern_match.group(1))
        df_feasible_0['pattern'] = pattern_number
    
    dfs.append(df_feasible_0)

# 全てのデータフレームを一つに結合
df_feasible_0 = pd.concat(dfs, ignore_index=True)

# 結果を保存するリスト
pareto_results = []

# 各パラメータパターンごとにデータをフィルタリング
for pattern_number in sorted(df_feasible_0['pattern'].unique()):  # 昇順にソート
    df_pattern = df_feasible_0[df_feasible_0['pattern'] == pattern_number]
    
    # スケーリングなしでデータを使用
    waiting_times, resource_costs = df_pattern['waiting_time'].values, df_pattern['resource_cost'].values
    
    # パレート効率解を識別
    pareto_indices = identify_pareto(waiting_times, resource_costs)
    
    # パレート効率解のデータを抽出
    pareto_solutions = df_pattern.iloc[pareto_indices]
    
    # 結果を保存
    pareto_results.append({
        'pattern_number': pattern_number,
        'pareto_count': len(pareto_indices),
        'total_count': len(df_pattern),
        'pareto_solutions': pareto_solutions
    })

# 全パターンのパレートフロントを求めるためのデータフレーム
all_pareto_solutions = pd.DataFrame()

# 各パラメータパターンごとにデータをフィルタリング
for result in pareto_results:
    pareto_solutions = result['pareto_solutions']
    all_pareto_solutions = pd.concat([all_pareto_solutions, pareto_solutions], ignore_index=True)

# スケーリングなしでデータを使用
waiting_times, resource_costs = all_pareto_solutions['waiting_time'].values, all_pareto_solutions['resource_cost'].values

# 全体のパレートフロントを求める
final_pareto_indices = identify_pareto(waiting_times, resource_costs)
final_pareto_solutions = all_pareto_solutions.iloc[final_pareto_indices]

# 各パターンのデータフレームにパターン番号を付与しておく
all_pareto_solutions_with_patterns = pd.merge(all_pareto_solutions, 
                                              df_feasible_0[['waiting_time', 'resource_cost', 'pattern']], 
                                              on=['waiting_time', 'resource_cost'],
                                              how='left')

# パレート効率解を可視化
plt.figure(figsize=(12, 8))

# グループごとのパレートフロントをプロット
for result in sorted(pareto_results, key=lambda x: x['pattern_number']):
    pareto_solutions = result['pareto_solutions']
    plt.scatter(pareto_solutions['waiting_time'], 
                pareto_solutions['resource_cost'], 
                color='blue', 
                alpha=0.6)

# 全体のパレートフロントをプロット
plt.scatter(final_pareto_solutions['waiting_time'], 
            final_pareto_solutions['resource_cost'], 
            color='black', 
            marker='X', s=100)

plt.xlabel('Waiting Time')
plt.ylabel('Resource Cost')
plt.title('Pareto Efficient Solutions by Pattern and Overall Pareto Front')
plt.grid(True)
plt.tight_layout()
plt.show()

# 各パラメータパターンのパレートフロント数を表示
for result in pareto_results:
    print(f"Pattern {result['pattern_number']} - Pareto Front Count: {result['pareto_count']} / {result['total_count']}")

# 全体のパレートフロント数を表示
print(f"Overall Pareto Front Count: {len(final_pareto_solutions)}")

# 全体パレートフロントのパターン情報を表示
print("Overall Pareto Front Pattern Information:")
for pattern in final_pareto_solutions['pattern'].unique():
    count = len(final_pareto_solutions[final_pareto_solutions['pattern'] == pattern])
    print(f"Pattern {pattern}: {count} solutions")

#パラメータDを固定しEの変えていき平均の値のデータ取得後plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# プロットの文字サイズを設定
plt.rcParams.update({
    'font.size': 14,          # 全体のフォントサイズ
    'axes.titlesize': 18,     # タイトルのフォントサイズ
    'axes.labelsize': 16,     # X軸、Y軸ラベルのフォントサイズ
    'xtick.labelsize': 12,    # X軸の目盛りラベルのフォントサイズ
    'ytick.labelsize': 12     # Y軸の目盛りラベルのフォントサイズ
})

# 読み込むCSVファイルのリスト
csv_files = [
    './search/t_30/0901/final_summary_1_to_6.csv',
    './search/t_30/0901/final_summary_7_to_12.csv',
    './search/t_30/0901/final_summary_13_to_18.csv',
    './search/t_30/0901/final_summary_19_to_24.csv',
    './search/t_30/0901/final_summary_25_to_30.csv'
]

# 凡例のラベルをDの値に対応させる
legend_labels = ['D=1', 'D=5', 'D=8', 'D=10', 'D=13']

# カラーマップを作成（viridisカラーマップを使用）
colors = plt.cm.viridis(np.linspace(0, 1, len(csv_files)))

# カスタムの横軸ラベル
custom_xticks = ['1', '10', '20', '30', '40', '50']

# プロットの作成 - 横幅を狭めるためfigsizeを小さく設定
fig, ax1 = plt.subplots(figsize=(6, 5))  # 横幅6インチ、縦幅5インチ

# 各CSVファイルについてループ
for i, file in enumerate(csv_files):
    final_df = pd.read_csv(file)
    x_indices = np.arange(len(final_df['label']))

    ax1.plot(
        x_indices, 
        final_df['mean_resource_cost'], 
        marker='o', linestyle='-', color=colors[i], 
        label=legend_labels[i]
    )

# 右のY軸を作成
ax2 = ax1.twinx()

# 各CSVファイルについてループ
for i, file in enumerate(csv_files):
    final_df = pd.read_csv(file)
    x_indices = np.arange(len(final_df['label']))

    ax2.plot(
        x_indices, 
        final_df['mean_waiting_time'], 
        marker='x', linestyle='--', color=colors[i], 
        label=legend_labels[i]
    )

# X軸のカスタムラベルを設定
plt.xticks(ticks=np.arange(len(custom_xticks)), labels=custom_xticks, fontsize=12, rotation=45)

# 軸ラベルの設定
ax1.set_xlabel('parameter E')
ax1.set_ylabel('resource cost')
ax2.set_ylabel('waiting time')

# X軸の範囲を調整
ax1.set_xlim(-0.5, len(custom_xticks) - 0.5)

# 余白を調整
ax1.margins(x=0)

# プロットのタイトルを追加
fig.suptitle('Average Resource Cost and Waiting Time by Pattern and Iteration=100')

# 凡例の追加
ax1.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=2, fontsize=10)  # 左側の凡例のフォントサイズ
ax2.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=2, fontsize=10)  # 右側の凡例のフォントサイズ

# グリッドを有効化
ax1.grid(True)

# プロットのレイアウトを自動調整し、余白を削除
plt.tight_layout()

# プロットの表示
plt.show()