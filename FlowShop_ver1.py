#!/usr/bin/env python
# coding: utf-8

from pyqubo import Array, Placeholder, solve_ising, Constraint ,solve_qubo, SubH
import re

# resourceに番号を割り当てる
def assign_resource_num(resource):
    resource_num = {}
    cnt = 0
    for i in resource:
        for j in resource[i]:
            resource_num[j] = cnt
            cnt += 1
    return resource_num

# スケジューリング問題のjobリソースを生成
def generate_jobs_and_resources(num_jobs):
    resource_t = {'R0': [], 'R1': [], 'R2': []}
    place = []
    job = []

    for i in range(1, num_jobs + 1):
        t11 = f't{i}1'
        t12 = f't{i}2'
        t13 = f't{i}3'

        job.append([t11, t12, t13])

        resource_t['R0'].append(t11)
        resource_t['R1'].append(t12)
        resource_t['R2'].append(t13)

        place.append([f'p{i}1', f'p{i}2', f'p{i}3', f'p{i}4'])

    return resource_t, place, job

# ジョブ数を指定して生成
num_jobs = 16
resource_t, place, job = generate_jobs_and_resources(num_jobs)

resource_m = {'R0':['m11','m12'],'R1':['m21','m22','m23'],'R2':['m31','m32','m33','m34','m35','m36']}
machine_processing_time = {'m11':1,'m12':2,'m21':3,'m22':1,'m23':2,'m31':1,'m32':3,'m33':2,
                          'm34':2,'m35':1,'m36':2}

machine_cost = {'m11':10,'m12':14,'m21':13,'m22':10,'m23':10,'m31':12,'m32':10,'m33':9,
                          'm34':12,'m35':14,'m36':15}

resource_num = assign_resource_num(resource_m)

# トランジションに番号を割り当てる
transition_num = {}
cnt = 0
for i in job:
    for j in i:
        transition_num[j] = cnt
        cnt += 1

t_num = len(transition_num)
r_num = len(resource_num)
time = 12
x = Array.create('x', (time,t_num,r_num), 'BINARY') # binaryの宣言 


def cal_min_processing(resource_m, machine_processing_time):
    """
    各リソースマシーンのprocessing timeの最小を求める
    Args:
        resource_m : リソースマシーンの辞書型
        machine_processing_time : マシーンの処理時間の辞書型

    Returns:
        R0,R1,R2それぞれのprocessing timeの最小時間を求める
    """
    min_m_time = []
    for idx,v in enumerate(resource_m):
        _min = 10000
        for i in resource_m[v]:
            _min = min(_min,machine_processing_time[i])
        min_m_time.append(_min)
    return min_m_time

def cal_max_processing(resource_m, machine_processing_time):
    """
    各リソースマシーンのprocessing timeの最大を求める
    Args:
        resource_m : リソースマシーンの辞書型
        machine_processing_time : マシーンの処理時間の辞書型

    Returns:
        R0,R1,R2それぞれのprocessing timeの最大時間を求める
    """
    max_m_time = []
    for idx,v in enumerate(resource_m):
        _min = -1
        for i in resource_m[v]:
            _min = max(_min,machine_processing_time[i])
        max_m_time.append(_min)
    return max_m_time

#limit_timeは変数が増えてしまわないように制限を用いている
#ubはlimit_timeを満たすための変数
def calc_upper_limit(step, jobs, limit_time, _max):
    """
    現在のステップにおける処理時間の上限時間の計算
    Args:
        step : 現在のステップ
        jobs : 各jobのタスクのリスト
        limit_time : 制限時間
        _max : 各リソースマシーンのprocessing timeの最大値のリスト(max_list)

    Returns:
    現在のステップの上限時間を求める
    """
    p_time = 0
    for job in jobs: #jobはstr型
        if len(job) >= 4:
            if int(job[3]) >= step:
                p_time += _max[int(job[3])-1]
        elif int(job[2]) >= step:
                p_time += _max[int(job[2])-1]
    return limit_time - p_time


# 各トランジションの処理時間の最大・最小値を求める
range_trantision_ptime = {}
min_list = cal_min_processing(resource_m, machine_processing_time)
max_list = cal_max_processing(resource_m, machine_processing_time)


for idx, j in enumerate(job):
    for i in range(len(j)):
        range_time = []
        lb = sum(min_list[0:i]) if i != 0 else 0
        range_time.append(lb)
        ub = calc_upper_limit(i+1, j, time, max_list)
        range_time.append(ub)
        range_trantision_ptime[j[i]] = range_time

#制約
H_firing = 0.0
for idx, j in enumerate(job):
    for i in range(len(j)):
        sigma_h_firing = 0.0
        for r in resource_m['R'+str(i)]:
            lb = range_trantision_ptime[j[i]][0]
            ub = range_trantision_ptime[j[i]][1]
            for k in range(lb,ub):
                t = transition_num['t'+str(idx+1)+str(i+1)]
                sigma_h_firing += x[k,t,resource_num[r]]
        H_firing += Constraint((1-sigma_h_firing)**2,label="one_fired_t{}{}{}".format(idx+1,i+1,resource_num[r]))


#制約
H_conflict = 0.0

for i in range(len(resource_m)):
    for r in resource_m['R'+str(i)]:
        for j1 in range(len(job)):
            t1 = transition_num[job[j1][i]]
            for j2 in range(len(job)):
                t2 = transition_num[job[j2][i]]
                if j1 != j2:
                    lb = range_trantision_ptime[job[j1][i]][0]
                    ub = range_trantision_ptime[job[j1][i]][1]
                    for k1 in range(lb,ub):
                        max_limit = k1 + machine_processing_time[r]
                        if max_limit > 10:
                            max_limit = 10
                        for k2 in range(k1,max_limit):
                            if k2 < range_trantision_ptime[job[j2][i]][1]:
                                H_conflict += Constraint((x[k1,t1,resource_num[r]])*(x[k2,t2,resource_num[r]]), label="conflict{}".format(t1))

#制約
H_precedence = 0.0

for idx, j in enumerate(job):
    for i in range(len(j)):
        lb = range_trantision_ptime[j[i]][0]
        ub = range_trantision_ptime[j[i]][1]
        for r1 in resource_m['R'+str(i)]:
            if i+1 < len(j):
                for r2 in resource_m['R'+str(i+1)]:
                    t1 = transition_num['t'+str(idx+1)+str(i+1)]
                    t2 = transition_num['t'+str(idx+1)+str(i+2)]
                    for k1 in range(lb, ub):
                        fd = k1 + machine_processing_time[r1]
                        for k2 in range(fd, ub):
                            H_precedence += Constraint((x[k1, t1, resource_num[r1]]) * (x[k2, t2, resource_num[r2]]),
                                                        label="precedence{}{}{}{}".format(k1, t1, k2, t2))

#目的関数
H_resourceCost = 0.0

for i in range(len(resource_m)):
    for r in resource_m['R'+str(i)]:
        r_num = resource_num[r]
        rc = machine_cost[r]
        fd = machine_processing_time[r]
        for j in range(len(job)):
            lb = range_trantision_ptime['t'+str(j+1)+str(i+1)][0]
            ub = range_trantision_ptime['t'+str(j+1)+str(i+1)][1]
            t = transition_num['t'+str(j+1)+str(i+1)]
            for k in range(lb,ub):
                H_resourceCost += rc*fd*x[k,t,r_num]

#目的関数
H_waitingTime = 0.0

for j in range(len(job)):
    for i in range(len(job[j])-1):
        lb1 = range_trantision_ptime['t'+str(j+1)+str(i+2)][0]
        ub1 = range_trantision_ptime['t'+str(j+1)+str(i+2)][1]
        t1 = transition_num['t'+str(j+1)+str(i+2)]
        
        lb2 = range_trantision_ptime['t'+str(j+1)+str(i+1)][0]
        ub2 = range_trantision_ptime['t'+str(j+1)+str(i+1)][1]
        t2 = transition_num['t'+str(j+1)+str(i+1)]
        for k1 in range(lb1,ub1):
            for r1 in resource_m['R'+str(i+1)]:
                r_num1 = resource_num[r1]
                for k2 in range(lb2,ub2):
                    for r2 in resource_m['R'+str(i)]:
                        fd = k2 + machine_processing_time[r2]
                        r_num2 = resource_num[r2]
                        if r_num1 - fd >= 0:
                            H_waitingTime += r_num1*x[k1,t1,r_num1] - fd*x[k2,t2,r_num2]

from collections import OrderedDict

def calc_machine_cost(machine_processing_time,ans):
    ans_order = OrderedDict()
    _cnt = 0

    for idx in machine_processing_time.keys():
        ans_order[idx] = _cnt
        _cnt += 1

    ans_list = list(ans_order.keys())

    sum_cost = 0
    for i in ans:
        sum_cost += machine_cost[ans_list[int(i[3])]]
    return sum_cost

def calc_processing_time(machine_processing_time,ans):
    ans_order = OrderedDict()
    _cnt = 0

    for idx in machine_processing_time.keys():
        ans_order[idx] = _cnt
        _cnt += 1

    ans_list = list(ans_order.keys())

    max_time = 0
    for i in range(len(ans)):
        max_time = max(max_time,int(ans[i][1])+machine_processing_time[ans_list[int(ans[i][3])]])
    return max_time

# ## OpenJij

# ハミルトニアンを構築
A = Placeholder("A")
B = Placeholder("B")
C = Placeholder("C")
D = Placeholder("D")
E = Placeholder("E")

H = SubH(A*H_firing, "SubH1") + SubH(B*H_conflict, "SubH2") + SubH(C*H_precedence, "SubH3") + D*H_resourceCost + E*H_waitingTime

# モデルをコンパイル
model = H.compile()

feed_dict = {"A": 600.0, "B": 140.0, "C": 30.0, "D": 1, "E": 0.5} #job18個まで計算できる
# feed_dict = {"A": 600.0, "B": 160.0, "C": 50.0, "D": 1, "E": 0.5}

# QUBOを作成
bqm = model.to_bqm(feed_dict=feed_dict)

# QUBOの変数数（量子ビットの数）を確認
num_qubits = len(bqm.variables)
print(f"Number of qubits: {num_qubits}")

import openjij as oj
from pyqubo import Array, Placeholder, solve_ising, Constraint, SubH, Model
import numpy as np
import time 

# アニーリング回数
num_iterations = 100

# QUBOを作成
bqm = model.to_bqm(feed_dict=feed_dict)

# QUBOを辞書形式に変換
qubo_dict = bqm.to_qubo()[0]

# OpenJijのSamplerを使用してQUBOを解く
sampler = oj.SASampler()

response = sampler.sample_qubo(qubo_dict, num_reads=num_iterations)

# サンプリング結果をデコード
decoded_samples = model.decode_sampleset(response, feed_dict=feed_dict)

# 最良のサンプルを選択
best_sample = min(decoded_samples, key=lambda x: x.energy)
num_broken = len(best_sample.constraints(only_broken=True))

# 最良のサンプルの各サブハミルトニアンの値を計算する関数
def calculate_subh_energy(subh, sample, feed_dict):
    """
    各サブハミルトニアンの計算
    Args:
        subh: サブハミルトニアン
        sample: 最良のサンプル
        feed_dict: ハイパーパラメータ
    Returns:
        ハミルトニアンの各項のエネルギーを計算する
    """
    subh_model = subh.compile()
    subh_qubo, subh_offset = subh_model.to_qubo(feed_dict=feed_dict)
    subh_energy = sum(subh_qubo.get((v, v), 0) * sample[v] for v in sample) #一次項の計算
    for (v1, v2), coeff in subh_qubo.items(): #二次の項の計算
        if v1 != v2:
            subh_energy += coeff * sample[v1] * sample[v2]
    return subh_energy + subh_offset

resource_cost = calculate_subh_energy(D * H_resourceCost, best_sample.sample, feed_dict)
wating_time = calculate_subh_energy(E * H_waitingTime, best_sample.sample, feed_dict)

# デコードされた解を表示
print("Decoded Solution:")
for variable, value in best_sample.sample.items():
    print(f"{variable}: {value}")

print("\nEnergy:")
print(best_sample.energy)

print(best_sample.constraints(only_broken=True))
print("number of broken constarint = {}".format(num_broken))

print(best_sample.constraints(only_broken=True))


keys = [k for k, v in best_sample.sample.items() if v == 1]
#keys

# 正規表現
def extractVariable(s):  
    literal = re.split('[\[\]]', s)
    while '' in literal:
        literal.remove('')       
    return literal

ans = []

for k in keys:
    ans.append(extractVariable(k))

sort_ans = list(range(len(ans)))

for i in ans:
    sort_ans[int(i[2])] = i


machine_cost = calc_machine_cost(machine_processing_time, sort_ans)
processing_time = calc_processing_time(machine_processing_time, sort_ans)

print("-"*30)
print("Machine Cost : {}".format(machine_cost))
print("-"*30)
print("Processing Time : {}".format(processing_time))
print("-"*30)
print("Resource Cost : {}".format(resource_cost))
print("-"*30)
print("Wating Time : {}".format(wating_time))
print("-"*30)

subH_list = ['SubH1','SubH2','SubH3']
is_feasible = True

for i in subH_list:
    if best_sample.subh[i] != 0:
        is_feasible = False
if is_feasible:
    print('Feasible!!')
else:
    print(best_sample.subh)
    
import csv
import os
from datetime import datetime

now = datetime.now()
timestamp = now.strftime('%Y-%m-%d')

# 保存するファイル名
csv_filename = f'./search/{timestamp}_patt1_job{num_jobs}.csv'

# CSVファイルが存在するかどうかを確認
file_exists = os.path.isfile(csv_filename)

constraint_error = best_sample.constraints(only_broken=True)
feasible = 0 if is_feasible else constraint_error

# データをリストのリストとして準備
data = [[num_iterations, machine_cost, processing_time, resource_cost, wating_time, feasible]]

# ファイルが存在しない場合はヘッダーを追加
if not file_exists:
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['iteration', 'machine_cost', 'processing_time', 'resource_cost', 'wating_time','feasible'])
        writer.writerows(data)
else:
    # ファイルが存在する場合は追記
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)