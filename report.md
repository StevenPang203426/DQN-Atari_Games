# RL实践 - MsPacman DQN实验报告

## 1. 实验目标

本实验基于 Gymnasium Atari 环境实现 Deep Q-Network (DQN)，训练智能体在 `ALE/MsPacman-v5` 中根据游戏画面选择动作。项目要求补全 `dqn_atari.py` 中的 `QNetwork` 卷积神经网络，并通过训练生成吃豆人游戏视频。

## 2. Gymnasium训练流程

Gymnasium 提供统一的强化学习环境接口。训练时先调用 `envs.reset(seed=args.seed)` 得到初始观测，然后智能体根据当前观测选择动作，调用 `envs.step(actions)` 与环境交互，获得 `next_obs`、`rewards`、`terminated`、`truncated` 和 `infos`。本项目使用 Atari wrapper 对图像做预处理，包括随机空操作、跳帧、生命结束处理、奖励裁剪、缩放到 `84x84`、灰度化和 4 帧堆叠，最终网络输入为 `4 x 84 x 84`。

## 3. DQN算法原理

DQN 使用深度神经网络近似动作价值函数 `Q(s,a)`。网络输入当前状态 `s`，输出每个离散动作的 Q 值，智能体通过 epsilon-greedy 策略在探索和利用之间切换。训练中将交互得到的 `(state, action, reward, next_state, done)` 存入 Replay Buffer，再随机采样 batch 更新网络，降低样本时间相关性。目标值使用 target network 计算：

```text
td_target = reward + gamma * max_a Q_target(next_state, a) * (1 - done)
```

当前网络通过最小化 `Q_current(state, action)` 与 `td_target` 的均方误差进行优化，并定期将当前网络参数同步到 target network，从而提升训练稳定性。

## 4. 网络结构与训练配置

按照课程 PDF 中的结构，`QNetwork` 采用经典 Atari DQN 卷积网络：

```text
Input(4x84x84)
-> Conv2d(4, 32, kernel=8, stride=4) + ReLU
-> Conv2d(32, 64, kernel=4, stride=2) + ReLU
-> Conv2d(64, 64, kernel=3, stride=1) + ReLU
-> Flatten
-> Linear(3136, 512) + ReLU
-> Linear(512, action_num)
```

正式训练命令如下：

```bash
python dqn_atari.py --exp-name MsPacman-v5 --capture-video --save-model --env-id ALE/MsPacman-v5 --total-timesteps 5000000 --buffer-size 400000
```

主要超参数包括 `learning-rate=1e-4`、`gamma=0.99`、`batch-size=32`、`learning-starts=80000`、`train-frequency=4`、`target-network-frequency=1000`。训练过程会保存模型到 `runs/{run_name}/MsPacman-v5.pth`，并将视频保存到 `videos/{run_name}/` 或 `videos/{run_name}-eval/`。

## 5. 实验结果

本地环境用于 smoke test，确认网络维度、训练循环和模型保存逻辑可运行；正式结果建议在云端 GPU 上完成 5,000,000 steps 长训。提交时选择 `videos/{run_name}-eval/` 中最后生成的评估视频作为吃豆人游戏视频，并在训练完成后记录最终 episode return。
