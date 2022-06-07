# Lane-Changing

## 主要环境依赖
- python 3.9.12
- pytorch 1.11.0

其他依赖在requirements.txt中已经列出。

## 关键目录说明
```
.
├── DQN_TCC_best_model.zip  # 所有训练结果中性能最优的权重。为DQN在TimeToCollision环境下的训练权重。平均reward约为47.3
├── agent.py                # 训练与测试的主要入口文件。
├── env.py                  # 环境封装
├── learners.py             # 多个强化学习算法的封装
├── log                     # tensorboard的日志目录
├── models.py               # 深度学习模型的编写处
├── plots                   # 绘制实验图表的模块
├── requirements.txt        # 环境依赖
└── weights                 # 包含各个agent，状态类型对应的训练好的权重。

```


## 使用方法

使用agent.py作为训练和测试的入口程序。
参数中的"|"符号表示可选项之一。
``` shell
python agent.py
    --mode train|test                  # train或test，分别代表训练和测试模式
    --obs_type 1|2|3                   # 1代表Kinematics状态，2代表TimeToCollision状态，3代表OccupancyGrid状态
    --learner_type PPO|A2C|DQN|EGO     # 分别代表不同类型的四种agent。其中EGO只支持Kinematics状态。
    --parallels 1                      # 整数，代表并行训练的环境个数
    --nb_steps 100000                  # 整数，代表训练的step数量。
    --eval_interval_steps 50           # 整数，代表并行训练的每个agent经过多少个step进行一次测试。
    --init_weight_path path/to/weights # 字符串，表示初始化agent的权重zip路径。mode为test时必须传该参数。
```

举例：若想训练PPO，使用Kinematics状态，使用20个并发环境，训练100000步，并且并发中的每个环境训练50步后进行一次测试：
``` shell
python agent.py --learner_type PPO --mode train --obs_type 1 --eval_interval_steps 50 --parallels 20 --nb_steps 100000
```

举例：若想测试DQN，使用TimeToCollision状态，权重路径为./DQN_TCC_best_model.zip，则使用：
``` shell
python agent.py --learner_type DQN --mode test --obs_type 1 --init_weight_path ./DQN_TCC_best_model.zip 
```

