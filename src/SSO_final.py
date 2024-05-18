import numpy as np
import time
import random
import torch
from utils import generate_random_numbers, random_select, find_min_key_value, SSO_2_csv
from train import Train_pipeline

def SSO_train_final(exp_name, Cg, Cp, Cw, Nsol, Ngen, random_number_range, initial_sol, train_detail):
    random.seed(42)
    torch.backends.cudnn.benchmark = True  # 開啟 CuDNN 自動調整
    start_time = time.time()
    print(f"start SSO train at {start_time}")

    model_name = f'F:/git_repo/WKN_SSO/result/pth/{exp_name}.pth'
    train_vali, hyper_parameter, wc = train_detail

    Nvar = len(random_number_range)
    X, FX, pX, pF, gBest = [initial_sol], [], [initial_sol], [], 0
    value_to_x_dict = {}

    _, initial_fit, _ = Train_pipeline(train_vali, hyper_parameter, initial_sol, wc)
    FX.append(initial_fit)
    pF.append(initial_fit)
    value_to_x_dict[initial_fit] = (tuple(initial_sol), (0, 0))
    print(0, 0, initial_sol, initial_fit)
    print("=====================================")

    for sol in range(1, Nsol):
        rX = generate_random_numbers(random_number_range)
        print(f"(gen, sol) = (0, {sol})")
        print(f"result X = {rX}")
        X.append(rX)
        pX.append(rX)
        _, fit, _ = Train_pipeline(train_vali, hyper_parameter, rX, wc)
        FX.append(fit)
        pF.append(fit)
        print(f"MSE fit = {fit}")
        value_to_x_dict[fit] = (tuple(rX), (0, sol))
        if fit < FX[gBest]:
            gBest = sol
            pX[gBest] = rX

    for gen in range(1, Ngen + 1):
        for sol in range(Nsol):
            for job in range(Nvar):
                rnd2 = np.random.rand()
                if rnd2 < Cg:
                    X[sol][job] = pX[gBest][job]
                elif rnd2 < Cp:
                    X[sol][job] = pX[sol][job]
                elif rnd2 < Cw:
                    X[sol][job] = X[sol - 1][job]
                else:
                    X[sol][job] = random_select(job, random_number_range)

                if not random_number_range[job][0] <= X[sol][job] <= random_number_range[job][1]:
                    X[sol][job] = int(random_select(job, random_number_range)) if X[sol][job] > 1 else random_select(job, random_number_range)
                
                X[sol][job] = int(X[sol][job])  # 確保所有值都是整數

            print(f"(gen, sol) = ({gen}, {sol})")
            print(f"result X = {X[sol]}")
            _, fit, train_result = Train_pipeline(train_vali, hyper_parameter, X[sol], wc)
            print(f"MSE fit = {fit}")
            value_to_x_dict[fit] = (tuple(X[sol]), (gen, sol))

            if fit < pF[sol]:
                pF[sol] = fit
                if fit < pF[gBest]:
                    gBest = sol
                    torch.save(train_result, model_name)
            if gen == 1 and sol == 0:
                torch.save(train_result, model_name)

    print(f"optimal sequence {value_to_x_dict[pF[gBest]][0]}")
    print(f"optimal value: {pF[gBest]}")
    print(f"optimal generation number: {value_to_x_dict[pF[gBest]][1][0]}")
    print(f"optimal sol number: {value_to_x_dict[pF[gBest]][1][1]}")
    print(f"sso_time: {time.time() - start_time}")

    return value_to_x_dict

if __name__ == "__main__":
    hyper_parameter = [32, 30]  # [batch_size, num_epochs]
    Learning_set = 'F:/git_repo/WKN_SSO/viberation_dataset/Learning_set/'
    Validation_set = 'F:/git_repo/WKN_SSO/viberation_dataset/Validation_set/'
    work_condition = [1, 2]
    exp_topic = 'Final_SSO_ML'

    random_number_range = [(1, 1000),
                           (1, 64),
                           (2, 128),
                           (1, 64),
                           (2, 32), 
                           (1, 64), 
                           (2, 32), 
                           (1, 10), 
                           (1, 99), 
                           (1, 5120), 
                           (1, 99), 
                           (50, 90), 
                           (50, 90)]
    
    iX = [100, 32, 64, 16, 32, 32, 3, 1, 50, 64, 30, 60, 60]

    # SSO setup
    Cg = [0.7, 0.1, 0.1, 0.1]
    Cp = [0.8, 0.8, 0.2, 0.2]
    Cw = [0.9, 0.9, 0.9, 0.3]
    Nsol = 6
    Ngen = 10
    exp_num = 30

    # Train
    start_time = time.time()
    train_vali = [Learning_set, Validation_set, 3]

    for term in range(exp_num):
        print(f"Current terms = {term + 1}")
        c = 0
        for wc in work_condition:
            train_detail = [train_vali, hyper_parameter, wc]
            exp_name = f"{exp_topic}_wc{wc}_{term + 1}st"
            print(f"start SSO with Cg = {Cg[c]}, Cp = {Cp[c]}, Cw = {Cw[c]}")
            all_result = SSO_train_final(exp_name, Cg[c], Cp[c], Cw[c], Nsol, Ngen, random_number_range, iX, train_detail)
            min_key, min_value = find_min_key_value(all_result)
            print(f"min_key = {min_key}, value of {min_value}")
            csv_name = f'F:/git_repo/WKN_SSO/result/SSO_result/{exp_name}.csv'
            SSO_2_csv(csv_name, all_result)

        print("SSO Train Finish!")
        print(f"SSO Train Time = {time.time() - start_time}")


