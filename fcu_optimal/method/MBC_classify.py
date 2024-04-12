import numpy as np
import pandas as pd
import math
import datetime


def load_data(path, name):
    df = pd.read_csv(path + name + '.csv')
    df = df[name]
    return df


def data_processed(data, win_len):
    df = pd.DataFrame()
    for i in range(0, len(data), win_len):
        df_new = (data[i:i+win_len]).reset_index(drop=True)
        df = pd.concat([df, df_new], axis=1, ignore_index=True)
    return df


def env(a_air, a_water):
    k = np.load('../system_data/env.npy')  # 这些数据包含了环境相关的系数
    p_f = np.load('../system_data/P_fan.npy')  # 风扇功率的系数
    p_p = np.load('../system_data/P_pump.npy')  # 泵功率的系数
    q_fcu = k[0] + k[1] * a_air + k[2] * a_water + k[3] * a_air ** 2 + k[4] * a_air * a_water + k[5] * a_water ** 2
    power_fan = (p_f[0]) * a_air ** 2 + (p_f[1]) * a_air + (p_f[2])  # 风流量-功率
    power_pump = (p_p[0]) * a_water ** 2 + (p_p[1]) * a_water + (p_p[2])  # 水流量-功率
    return q_fcu, power_fan, power_pump


def train(cl_data, save_path):

    df_reward = pd.DataFrame(index=range(0, len(cl_data) + 2), columns=range(0, len(cl_data.columns)))
    df_a_water = pd.DataFrame(index=range(0, len(cl_data)), columns=range(0, len(cl_data.columns)))
    df_a_air = pd.DataFrame(index=range(0, len(cl_data)), columns=range(0, len(cl_data.columns)))
    df_water_stability = pd.DataFrame(index=range(0, len(cl_data)), columns=range(0, len(cl_data.columns)))
    df_air_stability = pd.DataFrame(index=range(0, len(cl_data)), columns=range(0, len(cl_data.columns)))
    df_Pfan = pd.DataFrame(index=range(0, len(cl_data)), columns=range(0, len(cl_data.columns)))
    df_Ppump = pd.DataFrame(index=range(0, len(cl_data)), columns=range(0, len(cl_data.columns)))
    df_Q_fcu = pd.DataFrame(index=range(0, len(cl_data)), columns=range(0, len(cl_data.columns)))
    df_total_power = pd.DataFrame(index=range(0, len(cl_data) + 2), columns=range(0, len(cl_data.columns)))
    df_ccd_Q = pd.DataFrame(index=range(0, len(cl_data) + 2), columns=range(0, len(cl_data.columns)))

    a_water_list = pd.DataFrame(index=range(0, 14), columns=['water'])
    a_air_list = pd.DataFrame(index=range(0, 14), columns=['air'])
    a_air_list['air'] = 0
    a_water_list['water'] = 0

    for i_episode in range(len(cl_data.columns)):  # i_episode=0-10000+

        a_water_pre = 0
        a_air_pre = 0
        total_power = 0
        average_ccd_q = 0
        count_ccd = 0
        total_reward = 0

        for i in range(len(cl_data)):  # i = 0-13

            s = 1000 * cl_data.iloc[i, i_episode]
            diff = 0
            if i == len(cl_data) - 1:
                s_ = 0
                s_pre_0 = 0
                s_pre_1 = 0
            elif i == len(cl_data) - 2:
                s_ = 1000 * cl_data.iloc[i+1, i_episode]
                s_pre_0 = 1000 * cl_data.iloc[i+1, i_episode]
                s_pre_1 = 0
            else:
                s_ = 1000 * cl_data.iloc[i+1, i_episode]
                s_pre_0 = 1000 * cl_data.iloc[i+1, i_episode]
                s_pre_1 = 1000 * cl_data.iloc[i+2, i_episode]
            if s != 0:
                s_normalized = s / 5000  # 将这些变量的值缩放到0到1之间，标准化
                s__normalized = s_ / 5000
                s_pre_0_normalized = s_pre_0 / 5000
                s_pre_1_normalized = s_pre_1 / 5000

                if s < 2700:
                    a_water_range = range(80, 230)
                    a_air_range = range(100, 340)
                elif s < 3500:
                    a_water_range = range(230, 370)
                    a_air_range = range(340, 590)
                elif s < 4100:
                    a_water_range = range(370, 510)
                    a_air_range = range(590, 840)
                elif s < 4800:
                    a_water_range = range(510, 630)
                    a_air_range = range(840, 1060)
                else:
                    a_water_range = range(630, 700)
                    a_air_range = range(1060, 1200)

                def calculate_diff(a_water, a_air, a_water_pre, a_air_pre):
                    diff_water = abs(a_water - a_water_pre) / a_water
                    diff_air = abs(a_air - a_air_pre) / a_air
                    diff = 0.5 * (diff_water + diff_air)
                    return diff

                def calculate_r(s, q_fcu, power_fan, power_pump, diff):
                    k = 100
                    r = -(330 - 330 * math.exp(-((q_fcu - s) ** 2) / (2 * (s * 0.5) ** 2))) - (power_fan + power_pump) - k * diff
                    return r

                best_r = -float('inf')
                best_combination = (None, None)

                for a_water in a_water_range:
                    for a_air in a_air_range:
                        q_fcu, power_fan, power_pump = env(a_water=a_water, a_air=a_air)
                        diff = calculate_diff(a_water, a_air, a_water_pre, a_air_pre)
                        current_r = calculate_r(s, q_fcu, power_fan, power_pump, diff)
                        if current_r > best_r:
                            best_r = current_r
                            best_combination = (a_water, a_air)

                a_water_list.iloc[i, 0] = best_combination[0]
                a_air_list.iloc[i, 0] = best_combination[1]

                a_water_pre = best_combination[0]
                a_air_pre = best_combination[1]

                iteration = i_episode * len(cl_data) + i + 1

                current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                print('迭代 {}: {} - 状态s: {}, 最优动作: {}, r最大值: {}'.format(iteration, current_time, s, best_combination, best_r))

                df_reward.iloc[i, i_episode] = float(best_r)
                df_a_water.iloc[i, i_episode] = best_combination[0]
                df_a_air.iloc[i, i_episode] = best_combination[1]

                q_fcu, power_fan, power_pump = env(best_combination[0], best_combination[1])

                df_Q_fcu.iloc[i, i_episode] = q_fcu
                df_Pfan.iloc[i, i_episode] = power_fan
                df_Ppump.iloc[i, i_episode] = power_pump
                df_total_power.iloc[i, i_episode] = power_fan + power_pump
                df_ccd_Q.iloc[i, i_episode] = float(math.fabs(q_fcu - s) / s)

                total_power = total_power + power_fan + power_pump
                total_reward = total_reward + best_r
                average_ccd_q = average_ccd_q + float(math.fabs(q_fcu - s) / s)
                count_ccd += 1

        actions = pd.concat([a_water_list, a_air_list], axis=1, keys=['water_actions', 'air_actions'])
        actions_diff_out_for = actions.diff()
        actions_diff_out_for = actions_diff_out_for.abs()
        actions_diff_out_for['water_stability'] = actions_diff_out_for['water_actions'] / (actions['water_actions'])
        actions_diff_out_for['water_stability'] = actions_diff_out_for['water_stability'].astype(float)
        actions_diff_out_for['air_stability'] = actions_diff_out_for['air_actions'] / (actions['air_actions'])
        actions_diff_out_for['air_stability'] = actions_diff_out_for['air_stability'].astype(float)

        df_water_stability.iloc[:, i_episode] = actions_diff_out_for.iloc[:, 2]
        df_air_stability.iloc[:, i_episode] = actions_diff_out_for.iloc[:, 3]

        df_reward.iloc[len(cl_data) + 1, i_episode] = total_reward
        df_total_power.iloc[len(cl_data) + 1, i_episode] = total_power

    df_reward.to_csv(save_path + 'reward.csv', header=False, index=False)
    df_a_water.to_csv(save_path + 'a_water.csv', header=False, index=False)
    df_a_air.to_csv(save_path + 'a_air.csv', header=False, index=False)
    df_water_stability.to_csv(save_path + 'water_stability_diff.csv', header=False, index=False)
    df_air_stability.to_csv(save_path + 'air_stability_diff.csv', header=False, index=False)
    df_Pfan.to_csv(save_path + 'Pfan.csv', header=False, index=False)
    df_Ppump.to_csv(save_path + 'Ppump.csv', header=False, index=False)
    df_Q_fcu.to_csv(save_path + 'Q_fcu.csv', header=False, index=False)
    df_total_power.to_csv(save_path + 'total_power.csv', header=False, index=False)
    df_ccd_Q.to_csv(save_path + 'ccd_Q.csv', header=False, index=False)
    return 0


if __name__ == '__main__':
    data_path = '../../../../data/dest_out_data_processed_3/'
    save_path = '../contrast_experiment_result/MBC_classify_(2)/'
    room_name = '1-N-6'
    win_len = 14
    cool_load = load_data(data_path, room_name)
    cool_load = data_processed(cool_load, win_len)
    solution = train(cl_data=cool_load, save_path=save_path)
    print(solution)
