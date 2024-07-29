# import pandas as pd
# import matplotlib.pyplot as plt
# import os

# # CSVファイルが存在するディレクトリのパスを指定
# csv_directory = './search'

# # CSVファイルを保存するディレクトリのパスを指定
# output_directory = './search/plots/machine_processing'

# # 指定されたディレクトリ内のすべてのCSVファイルを取得
# csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]

# # 各CSVファイルをプロットし、保存
# for csv_file in csv_files:
#     file_path = os.path.join(csv_directory, csv_file)
#     input_csv = pd.read_csv(file_path)

#     plt.figure()
#     plt.plot(input_csv['iteration'], input_csv['processing_time'], 'b-', label='Processing Time')
#     plt.xlabel('Iteration')
#     plt.ylabel('Processing Time')
#     plt.title(f'Processing Time variation ({csv_file})')
#     plt.legend()

#     # プロットを保存
#     output_path = os.path.join(output_directory, f'{os.path.splitext(csv_file)[0]}_plot.png')
#     plt.savefig(output_path)
#     plt.close()  # プロットを閉じてメモリを解放

# patt1からpatt6がある
# ハイパーパラメータを1から6段階に分けたやつである

import pandas as pd
import matplotlib.pyplot as plt
import os

# CSVファイルが存在するディレクトリのパスを指定
csv_directory = './search/t_12'

# 指定されたディレクトリ内の '(任意の名前)' がファイル名に含まれるすべてのCSVファイルを取得
csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv') and 'job4' in f]
csv_files.sort()

# プロットを作成
plt.figure()

# 各CSVファイルをプロット
for csv_file in csv_files:
    file_path = os.path.join(csv_directory, csv_file)
    input_csv = pd.read_csv(file_path)

    # ファイル名をラベルとしてプロット
    label = os.path.splitext(csv_file)[0]  # 拡張子を除いたファイル名をラベルに使用
    plt.scatter(input_csv['machine_cost'], input_csv['processing_time'], label=label)

# プロットの詳細設定
plt.xlabel('Machine Cost')
plt.ylabel('Processing Time')
plt.title('a')
plt.legend()

# # プロットをファイルに保存
# output_file = './search/plots/plot_job12.png'
# plt.savefig(output_file)

plt.show()