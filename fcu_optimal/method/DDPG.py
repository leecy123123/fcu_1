# r中没有diff
# 结果存储在code/chapter_4/method/contrast_experiment_result/DDPG
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pandas as pd


def load_data(PATH, NAME):
    DF = pd.read_csv(PATH + NAME + '.csv')
    DF = DF[NAME]  # 根据给定的NAME，从读取的数据中选择列，结果存放在DF中
    return DF


def data_processed(DATA, WIN_LEN):
    DF = pd.DataFrame()
    for I in range(0, len(DATA), WIN_LEN):
        DF_NEW = (DATA[I:I+WIN_LEN]).reset_index(drop=True)
        # (DATA[I:I+WIN_LEN]) 表示从 DATA 中切片出索引范围从 I 到 I+WIN_LEN-1 的数据段
        # reset_index(drop=True) 表示重置数据的索引,drop=True 参数表示不保留原索引，而是生成一个新的连续的整数索引
        DF = pd.concat([DF, DF_NEW], axis=1, ignore_index=True)
        # pd.concat 函数将 DF_NEW 与 DF 进行列拼接，即将数据段连接到 DF 的右侧，得到一个新的 DataFrame 对象
    return DF


def env(A_AIR, A_WATER):
    K = np.load('../system_data/env.npy')  # 这些数据包含了环境相关的系数
    P_F = np.load('../system_data/P_fan.npy')  # 风扇功率的系数
    P_P = np.load('../system_data/P_pump.npy')  # 泵功率的系数
    Q_FCU = K[0] + K[1] * A_AIR + K[2] * A_WATER + K[3] * A_AIR ** 2 + K[4] * A_AIR * A_WATER + K[5] * A_WATER ** 2
    POWER_FAN = (P_F[0]) * (A_AIR) ** 2 + (P_F[1]) * A_AIR + (P_F[2])  # 风流量-功率
    POWER_PUMP = (P_P[0]) * (A_WATER) ** 2 + (P_P[1]) * A_WATER + (P_P[2])  # 水流量-功率
    return Q_FCU, POWER_FAN, POWER_PUMP


class ActorNet(nn.Module):  # define the network structure for actor and critic
    def __init__(self, s_dim, a_dim):
        super(ActorNet, self).__init__()
        # self.fc1 = nn.Linear(s_dim, 30)
        # self.fc1.weight.data.normal_(0, 0.1)  # initialization of FC1
        # self.out = nn.Linear(30, a_dim)
        # self.out.weight.data.normal_(0, 0.1)  # initilizaiton of OUT

        self.lstm = nn.LSTM(input_size=s_dim, hidden_size=32)
        # 使用LSTM层来处理输入状态序列（时间步）的维度，其中input_size即为s_dim，hidden_size为LSTM层的隐藏单元数量
        self.fc1 = nn.Linear(16 * s_dim, 32)
        # nn.Linear用于创建线性层，将输入数据进行线性变换
        # 16*s_dim是输入特征的数量（输入维度），32是输出维度
        # 创建了一个线性层，将 16*s_dim 个输入特征映射到32个输出特征，然后将其赋值给模型的成员变量，用于进行特征变换和特征提取
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.out(x)
        # x = torch.tanh(x/5)
        lstm_out, _ = self.lstm(torch.unsqueeze(x, 1))
        # 用unsqueeze函数将张量x在维度1上进行扩展，x 的形状由 (batch_size, input_size)变为 (batch_size, 1, input_size)
        # 将扩展维度后的 x 作为输入传递给 LSTM 模型 self.lstm 进行计算，LSTM 模型会自动处理序列的时间步并返回输出结果
        # _ 用于占位符，表示忽略第二个返回值，即最终的隐藏状态
        # print(lstm_out.shape)
        x = self.fc1(lstm_out.view(lstm_out.shape[0], -1))
        # 将 LSTM 输出 lstm_out 进行形状变换，将其转化为形状为 (batch_size, -1)的张量，然后将其作为输入传递给全连接层 self.fc1计算
        x = torch.tanh(self.fc2(x))  # 使用了torch.tanh激活函数，将全连接层的输出 x 变为范围在 -1 到 1 之间的值
        actions = x * 2

        return actions


class CriticNet(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(CriticNet, self).__init__()
        self.fcs = nn.Linear(s_dim, 30)  # 该层表示一个完全连接（线性）层，该层将状态作为输入并生成 30 维输出，并使用正态分布初始化权重
        self.fcs.weight.data.normal_(0, 0.1)  # normal_() 是一个 PyTorch 张量的方法，用于按照指定的均值和标准差进行正态分布的随机初始化，均值为 0，标准差为 0.1
        # 将 self.fcs 层的权重参数初始化为一个正态分布，以均值为 0，标准差为 0.1 的随机值。
        # 这是为了在训练神经网络时将权重参数初始化为一些随机的初值，以便网络能够从不同的初始状态开始学习。
        self.fca = nn.Linear(a_dim, 30)  # 该层表示一个完全连接的层，该层将动作作为输入并生成 30 维输出，a_dim是动作的维度，表示输入层的大小，30是输出层的大小
        self.fca.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, 1)  # 该层是另一个完全连接的层，它采用前两层输出的串联并产生一维输出
        self.out.weight.data.normal_(0, 0.1)
        # 这些层的权重使用均值为 0 且标准差为 0.1 的正态分布进行初始化

    def forward(self, s, a):  # 该方法定义模型的正向传递。它接受（状态）和（动作）作为输入
        x = self.fcs(s)
        y = self.fca(a)
        actions_value = self.out(F.relu(x + y))  # 线性层的输出相加，通过整流线性单元（ReLU）激活函数，然后馈送到层中
        return actions_value


class DDPG(object):
    def __init__(self,
                 A_DIM,
                 S_DIM,
                 MEMORY_CAPACITY,
                 LR_ACTOR,
                 LR_CRITIC,
                 BATCH_SIZE,
                 GAMMA,
                 ):
        self.GAMMA = GAMMA
        self.BATCH_SIZE = BATCH_SIZE
        self.LR_CRITIC = LR_CRITIC
        self.LR_ACTOR = LR_ACTOR
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.a_dim, self.s_dim,  = A_DIM, S_DIM,
        self.memory = np.zeros((self.MEMORY_CAPACITY, self.s_dim * 2 + self.a_dim + 1), dtype=np.float32)
        # np.zeros()：创建一个用零填充的新 numpy 数组，np.float32:数组的数据类型为32位浮点数
        self.pointer = 0  # serves as updating the memory data
        # Create the 4 network objects
        self.actor_eval = ActorNet(self.s_dim, self.a_dim)  # 策略评估
        # 在训练过程中根据当前状态选择动作，并通过 Critic 网络（用于估计动作值函数）评估选择的动作的质量
        # 根据 Critic 网络的评估结果，可以计算梯度并更新 Actor 网络的参数，以提高策略的性能。
        self.actor_target = ActorNet(self.s_dim, self.a_dim)  # 策略改进
        self.critic_eval = CriticNet(self.s_dim, self.a_dim)
        self.critic_target = CriticNet(self.s_dim, self.a_dim)
        # create 2 optimizers for actor and critic
        self.actor_optimizer = torch.optim.Adam(self.actor_eval.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = torch.optim.Adam(self.critic_eval.parameters(), lr=LR_CRITIC)
        # Define the loss function for critic network update
        self.loss_func = nn.MSELoss()

    def store_transition(self, s, a, r, s_):  # how to store the episodic data to buffer
        transition = np.hstack((s, a, [r], s_))  # hstack 是np中水平堆叠数组的函数，生成的数组将具有相同的行数，但列将被连接起来
        index = self.pointer % self.MEMORY_CAPACITY  # replace the old data with new data
        # pointer用于跟踪下一个用于存储数据的可用位置
        # MEMORY_CAPACITY:内存缓冲区的容量
        self.memory[index, :] = transition  # 将变量中的数据赋值给中索引为的行的所有列
        self.pointer += 1

    def choose_action(self, s):
        # print(s)
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        # 将 s 转换为一个浮点型张量，并在维度0上进行扩展，从而得到一个形状为 (1, s_dim) 的张量
        return self.actor_eval(s)[0].detach()
        # 调用了名为actor_eval 的模型来对输入 s 进行评估
        # detach()用于将输出从计算图中分离出来
        # 该方法返回了模型对输入状态 s 的评估结果的第一个输出。在强化学习中，这个输出通常表示某个动作的概率或估计值，用于选择下一步要执行的动作

    def learn(self):
        # 更新 actor 和 critic 网络，以最大化 Q 值的收益并使其逼近目标 Q 值
        # softly update the target networks
        TAU = 0.01
        # 通过设置TAU的值来实现对目标网络的软更新。 TAU是一个小于 1 的超参数，用于控制每次更新时目标网络和评估网络的权重

        for x in self.actor_target.state_dict().keys():  # 遍历 actor 目标网络的所有权重参数
            eval('self.actor_target.' + x + '.data.mul_((1-TAU))')  # x 是权重参数
            eval('self.actor_target.' + x + '.data.add_(TAU*self.actor_eval.' + x + '.data)')
        for x in self.critic_target.state_dict().keys():
            eval('self.critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.critic_target.' + x + '.data.add_(TAU*self.critic_eval.' + x + '.data)')
            # sample from buffer a mini-batch data
        indices = np.random.choice(self.MEMORY_CAPACITY, size=self.BATCH_SIZE)  # 从经验池中选择随机选择一批索引
        batch_trans = self.memory[indices, :]  # 根据索引indices在经验池中获得数据，返回的是一个二维数组
        # ：表示提取所有的列数据
        # extract data from mini-batch of transitions including s, a, r, s_
        batch_s = torch.FloatTensor(batch_trans[:, :self.s_dim])  # 将batch_trans数据转化为PyTorch的浮点型张量
        # ：表示提取所有行的数据，:self.s_dim 表示提取第一列到s_dim列的数据
        batch_a = torch.FloatTensor(batch_trans[:, self.s_dim:self.s_dim + self.a_dim])
        batch_r = torch.FloatTensor(batch_trans[:, -self.s_dim - 1: -self.s_dim])
        batch_s_ = torch.FloatTensor(batch_trans[:, -self.s_dim:])

        # make action and evaluate its action values
        a = self.actor_eval(batch_s)  # 根据观测数据计算动作a
        q = self.critic_eval(batch_s, a)  # 根据观测数据和动作预测来计算相应Q值（值函数）
        actor_loss = -torch.mean(q)  # 方便在训练中最小化actor_loss, loss用于actor网络的训练

        # optimize the loss of actor network
        self.actor_optimizer.zero_grad()  # 清零优化器actor网络的梯度信息。
        # PyTorch 在进行反向传播时会累积梯度，所以在每次更新参数之前，需要将之前的梯度清零，以避免梯度叠加的影响
        actor_loss.backward()  # 自动计算梯度并在网络中进行反向传播
        self.actor_optimizer.step()  # 根据梯度信息更新actor网络的参数。这个方法会调用优化器，并根据梯度以一定的学习率来更新网络参数。
        # 这样，优化器会根据actor网络的损失减小的方向，对参数进行适当的调整。

        # compute the target Q value using the information of next state
        a_target = self.actor_target(batch_s_)  # 对下一个状态 batch_s_ 进行前向传播，得到对应的动作输出 a_target
        q_tmp = self.critic_target(batch_s_, a_target)
        # 对下一个状态 batch_s_ 和动作输出 a_target 进行前向传播，得到Q值输出q_tmp（Critic 网络对下一状态和目标动作的 Q 值估计）
        q_target = batch_r + self.GAMMA * q_tmp

        # compute the current q value and the loss
        q_eval = self.critic_eval(batch_s, batch_a)
        # 对当前状态 batch_s 和动作 batch_a 进行前向传播，得到对应的 Q 值输出 q_eval （Critic 网络对当前状态和实际动作的 Q 值估计）
        td_error = self.loss_func(q_target, q_eval)  # 根据目标Q值 q_target 和评估Q值 q_eval计算 TD error，这是 Critic 网络的损失函数

        # optimize the loss of critic network
        self.critic_optimizer.zero_grad()
        td_error.backward()
        self.critic_optimizer.step()



def Main_training_process(CL_DATA,
                          Learning_rate_actor,  # 用于更新 actor 模型的梯度下降算法的步长
                          Learning_rate_critic,
                          Gamma,
                          Memory_capactity,
                          Batch_size,  # 每次训练时从经验池中抽取的样本数量
                          State_dim,
                          Actor_dim,
                          Save_path,  # 模型保存的路径
                          var_0_num=3,
                          var_1_num=3,
                          clip_num=1.6,
                          a0_water_bound_low=80,
                          a0_water_bound_high=700,
                          a1_air_bound_low=100,
                          a1_air_bound_high=1200):

    ddpg = DDPG(A_DIM=Actor_dim,
                S_DIM=State_dim,
                MEMORY_CAPACITY=Memory_capactity,
                LR_ACTOR=Learning_rate_actor,
                LR_CRITIC=Learning_rate_critic,
                BATCH_SIZE=Batch_size,
                GAMMA=Gamma)

    df_reward = pd.DataFrame(index=range(0, len(CL_DATA)+2), columns=range(0, len(CL_DATA.columns)))
    df_a_water = pd.DataFrame(index=range(0, len(CL_DATA)), columns=range(0, len(CL_DATA.columns)))
    df_a_air = pd.DataFrame(index=range(0, len(CL_DATA)), columns=range(0, len(CL_DATA.columns)))
    df_Pfan = pd.DataFrame(index=range(0, len(CL_DATA)), columns=range(0, len(CL_DATA.columns)))
    df_Ppump = pd.DataFrame(index=range(0, len(CL_DATA)), columns=range(0, len(CL_DATA.columns)))
    df_Q_fcu = pd.DataFrame(index=range(0, len(CL_DATA)), columns=range(0, len(CL_DATA.columns)))
    df_total_power = pd.DataFrame(index=range(0, len(CL_DATA)+2), columns=range(0, len(CL_DATA.columns)))
    df_ccd_Q = pd.DataFrame(index=range(0, len(CL_DATA)+2), columns=range(0, len(CL_DATA.columns)))

    l_reward = list()
    a_water_list = pd.DataFrame(index=range(0, 14), columns=['water'])
    a_air_list = pd.DataFrame(index=range(0, 14), columns=['air'])
    a_water_list['water'] = 0
    a_air_list['air'] = 0

    for i_episode in range(len(CL_DATA.columns)):  # i_episode=0-10000+
        total_reward = 0
        total_power = 0
        average_ccd_Q = 0
        count_ccd = 0

        for i in range(len(CL_DATA)):  # i = 0-13

            s = 1000 * CL_DATA.iloc[i, i_episode]
            if i == len(CL_DATA) - 1:  # 已经遍历到最后一行，下一时刻状态为0，且下两时刻的状态为0
                s_ = 0
                s_pre_0 = 0
                s_pre_1 = 0
            elif i == len(CL_DATA) - 2:
                s_ = 1000 * CL_DATA.iloc[i+1, i_episode]  # s_ 被赋值为下一行位置的元素乘以1000
                s_pre_0 = 1000 * CL_DATA.iloc[i+1, i_episode]
                s_pre_1 = 0
            else:
                s_ = 1000 * CL_DATA.iloc[i+1, i_episode]
                s_pre_0 = 1000 * CL_DATA.iloc[i+1, i_episode]
                s_pre_1 = 1000 * CL_DATA.iloc[i+2, i_episode]

            if s != 0:
                s_normalized = s / 5000  # 将这些变量的值缩放到0到1之间，标准化
                s__normalized = s_ / 5000
                s_pre_0_normalized = s_pre_0 / 5000
                s_pre_1_normalized = s_pre_1 / 5000
                a = ddpg.choose_action([s_normalized, s__normalized])
                a = (a.numpy())

                a[0] = np.random.normal(a[0], var_0_num)
                a[1] = np.random.normal(a[1], var_0_num)
                a[0] = np.clip(np.random.normal(a[0], var_0_num), -clip_num, clip_num)
                a[1] = np.clip(np.random.normal(a[1], var_1_num), -clip_num, clip_num)

                a_water = a[0] * (a0_water_bound_high - a0_water_bound_low) / 3.2 + (a0_water_bound_high + a0_water_bound_low) / 2
                a_air = a[1] * (a1_air_bound_high - a1_air_bound_low) / 3.2 + (a1_air_bound_high + a1_air_bound_low) / 2

                Q_fcu, Power_fan, Power_Pump = env(A_AIR=a_air, A_WATER=a_water)
                punish_coefficient_1 = 330 - 330 * math.exp(-((Q_fcu - s) ** 2) / (2 * (s * 0.5) ** 2))

                r = -punish_coefficient_1 - (Power_fan + Power_Pump)

                ddpg.store_transition([s_normalized, s__normalized], a, r.squeeze() / 900,
                                      [s_pre_0_normalized, s_pre_1_normalized])

                if ddpg.pointer > Memory_capactity:
                    if var_0_num > 0.1:
                        var_0_num *= 0.9995  # decay the exploration controller factor
                        var_1_num *= 0.9995
                    else:  # 如果小于0.1，将他们重新设置为0.1，来限制最小的探索程度
                        var_0_num = 0.1  # decay the exploration controller factor
                        var_1_num = 0.1
                    ddpg.learn()

                df_reward.iloc[i, i_episode] = float(r)
                df_a_water.iloc[i, i_episode] = a_water
                df_a_air.iloc[i, i_episode] = a_air
                df_Pfan.iloc[i, i_episode] = Power_fan
                df_Ppump.iloc[i, i_episode] = Power_Pump
                df_Q_fcu.iloc[i, i_episode] = Q_fcu
                df_total_power.iloc[i, i_episode] = Power_fan + Power_Pump
                df_ccd_Q.iloc[i, i_episode] = float(math.fabs(Q_fcu - s) / s)

                total_reward = total_reward + r
                total_power = total_power + Power_fan + Power_Pump
                average_ccd_Q = average_ccd_Q + float(math.fabs(Q_fcu - s) / s)
                count_ccd += 1

        if i_episode % 10 == 0:
            print('Ep: ', i_episode, ' |', 'r:', total_reward)

        df_reward.iloc[len(CL_DATA)+1, i_episode] = total_reward
        df_total_power.iloc[len(CL_DATA)+1, i_episode] = total_power

        if count_ccd != 0:
            df_ccd_Q.iloc[len(CL_DATA)+1, i_episode] = average_ccd_Q/count_ccd
        l_reward.append(total_reward)

    df_reward.to_csv(Save_path + 'reward.csv',  header=False, index=False)
    df_a_water.to_csv(Save_path+'a_water.csv',  header=False, index=False)
    df_a_air.to_csv(Save_path+'a_air.csv',  header=False, index=False)
    df_Pfan.to_csv(Save_path+'Pfan.csv',  header=False, index=False)
    df_Ppump.to_csv(Save_path+'Ppump.csv', header=False, index=False)
    df_Q_fcu.to_csv(Save_path+'Q_fcu.csv', header=False, index=False)
    df_total_power.to_csv(Save_path+'total_power.csv', header=False, index=False)
    df_ccd_Q.to_csv(Save_path+'ccd_Q.csv', header=False, index=False)

    # 绘制奖励曲线图表
    base_ddpg_reward = pd.read_csv('../data/gamma/G090/reward.csv')
    base_ddpg_reward = (base_ddpg_reward.iloc[14:15, :]).T  # 对base_ddpg_reward进行处理，选取第14行至第15行的数据，并进行转置操作
    base_ddpg_reward.index = range(0, len(base_ddpg_reward))
    base_ddpg_reward.columns = ['base_ddpg']
    plt.plot(base_ddpg_reward, color='red')
    plt.plot(l_reward, color='blue')  # 绘制L_reward的折线图
    plt.ylabel('reward')
    plt.legend()
    plt.show()

    return 0


if __name__ == '__main__':
    data_path = '../../../../data/dest_out_data_processed_3/'
    gamma = 0.99

    room_name = '1-N-6'  # 1/2/3/4/5房间需要改布局
    win_len = 14
    learning_rate_actor = 0.001
    learning_rate_critic = 0.002
    memory_capacity = 4000
    batch_size = 32

    # save_path = '../contrast_experiment_result/DDPG/'
    save_path = '../contrast_experiment_result/DDPG_reward_bar_5/第五次/'
    state_dim = 2
    actor_dim = 2

    cool_load = load_data(data_path, room_name)
    cool_load = data_processed(cool_load, win_len)
    answer = Main_training_process(CL_DATA=cool_load,
                                   Learning_rate_actor=learning_rate_actor,
                                   Learning_rate_critic=learning_rate_critic,
                                   Gamma=gamma,
                                   Memory_capactity=memory_capacity,
                                   Batch_size=batch_size,
                                   State_dim=state_dim,
                                   Actor_dim=actor_dim,
                                   Save_path=save_path)
    print(answer)

