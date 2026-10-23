# BadmintonLab-Game

Booster T1 羽毛球机器人模型回放，基于 Isaac Lab。`play` 分支加载用户提供的 TorchScript 模型，运行双机器人对打。

每个球场放置两台机器人，用户分别提供 player0 和 player1 的模型文件。每个模型只接收对应玩家的观测，独立推理确定性动作；观测分别转换到各自的标准球场坐标系。

## 安装

先准备能够运行 Isaac Lab 的 Isaac Sim、PyTorch 和 Python 环境，使用与所安装 Isaac Lab 版本匹配的依赖组合。本项目要求 Python >= 3.10，回放无需安装 RSL-RL 或 Isaac Lab 的 RSL-RL 支持。

在同一个环境中，从项目根目录执行：

```bash
python -m pip install -e source/badmintonlab_game
```

本项目以源码 checkout 加 editable install 的方式使用。机器人 URDF 和网格保留在 `robots/T1/`，运行时自动定位；请保留完整项目目录。生成的 USD 缓存写入项目的 `tmp/usd/`。

## 回放

```bash
python scripts/rsl_rl/play.py \
  --player0_checkpoint /absolute/path/to/player0.pt \
  --player1_checkpoint /absolute/path/to/player1.pt
```

默认任务为 `BadmintonLab-Game`，默认回放一个球场。必须分别通过 `--player0_checkpoint` 和 `--player1_checkpoint` 指定可由 `torch.jit.load` 加载的 TorchScript 模型；双方模型分别加载和调用。

每个模型的接口为 `forward(observations) -> actions`：输入形状为 `(N, observation_dim)`，输出为 `(N, action_dim)`，其中 `N` 为球场数。player0 模型接收 `player_0` 观测，player1 模型接收 `player_1` 观测；观测顺序和动作顺序必须与当前环境一致。两份动作按 player0、player1 的顺序沿关节维度拼接后交给环境。

模型需导出确定性策略，并包含训练时使用的观测归一化及其他输入预处理。当前接口按无状态策略调用，模型应支持不同批量大小。回放端直接调用模型，不重建网络或额外归一化观测。

| 参数 | 用途 |
| --- | --- |
| `--player0_checkpoint PATH` | 必填，指定玩家 0 的 TorchScript 模型 |
| `--player1_checkpoint PATH` | 必填，指定玩家 1 的 TorchScript 模型 |
| `--num_envs N` | 球场数，默认 1 |
| `--seed N` | 随机种子，默认 42，`-1` 表示随机采样 |
| `--device cuda:0` | 仿真与策略设备 |
| `--clip_actions LIMIT` | 可选动作裁剪，与训练时的裁剪设置一致；默认不裁剪 |

## 环境规则

- T1 使用关节位置目标控制，包含执行器延迟和随速度变化的力矩限制。
- 物理步长为 0.005 秒，每 4 个物理步更新一次动作，即 50 Hz 控制频率。
- 羽毛球用带碰撞的球体近似，每个物理步施加重力和二次空气阻力。
- 每回合随机选择先击球方，将球静止放置在其半场上方，由机器人完成开球。
- 球拍接触检测覆盖整个控制步的物理子步，只有轮到的玩家能记录有效击球。
- 落地、出界或下网结束回合并自动重置，不设置回合超时。摔倒本身不会直接判负。

## 代码结构

```text
scripts/rsl_rl/
  play.py                      # 双玩家模型回放
source/badmintonlab_game/
  config/extension.toml        # Isaac Lab 扩展元数据
  badmintonlab_game/
    inference.py               # 双玩家观测和动作适配
    assets/                    # 球场、T1 与执行器配置
    tasks/manager_based/badmintonlab_game/
      dual_agent_env.py        # 物理推进、击球、判定与重置
      dual_agent_env_cfg.py    # 场景与环境配置
      player0_cfg.py           # Player 0 观测和初始状态配置
      player1_cfg.py           # Player 1 观测和初始状态配置
      mdp/                     # 场景重置函数
robots/T1/                     # URDF 及其引用的网格
```

回合在羽毛球落地、出界或下网时自动重置，不设置回合超时，也不计算训练奖励。

## 开发检查

```bash
python -m pip install -e 'source/badmintonlab_game[dev]'
ruff check .
ruff format --check .
```

静态检查不启动 Isaac Sim，也不验证实际仿真或模型行为。

源码中已有的第三方版权和许可证声明予以保留。本分支未新增项目许可证。

评测方可分别替换 `player0_cfg.py` 和 `player1_cfg.py` 中的 `ObsCfg` 和 `EventsCfg`，定义各自的观测、root 初始位姿及关节初始状态。需要 Isaac Lab 未提供的 MDP 函数时，可直接在对应文件中定义并引用。
