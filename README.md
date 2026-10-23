# BadmintonLab-Game

Booster T1 羽毛球机器人共享策略自对弈训练，基于 Isaac Lab 和 RSL-RL PPO。

每个球场放置两台机器人，双方使用同一个策略，观测分别转换到各自的标准球场坐标系。训练时，`N` 个球场提供 `2N` 份玩家样本；双方奖励分别计算，回合同步结束和重置。

## 安装

先准备能够运行 Isaac Lab 的 Isaac Sim、PyTorch 和 Python 环境，并在该环境中安装 Isaac Lab 的 RSL-RL 支持。使用与所安装 Isaac Lab 版本匹配的依赖组合。本项目要求 Python >= 3.10、RSL-RL >= 3.0.1。

在同一个环境中，从项目根目录执行：

```bash
python -m pip install -e source/badmintonlab_game
```

本项目以源码 checkout 加 editable install 的方式使用。机器人 URDF 和网格保留在 `robots/T1/`，运行时自动定位；请保留完整项目目录。生成的 USD 缓存写入项目的 `tmp/usd/`。

## 训练

```bash
python scripts/rsl_rl/train.py --headless --num_envs 2048
```

默认任务为 `Booster-Badminton-Dual-v1`。可以先用较少球场进行冒烟检查：

```bash
python scripts/rsl_rl/train.py --headless --num_envs 4 --max_iterations 2
```

日志、模型和本次配置写入 `logs/rsl_rl/badmintonlab_game/<时间戳>/`。常用参数：

| 参数 | 用途 |
| --- | --- |
| `--num_envs` | 球场数，默认 2048 |
| `--max_iterations` | PPO 迭代数，默认 50000 |
| `--seed` | 随机种子，`-1` 表示随机采样 |
| `--device cuda:0` | 仿真与策略设备 |
| `--run_name NAME` | 日志目录后缀 |
| `--video` | 定期录制训练视频 |

支持 Hydra 覆盖环境和 PPO 参数，例如修改过程奖励的击球次数上限：

```bash
python scripts/rsl_rl/train.py --headless env.process_reward_cap=10
```

恢复训练时加载原生 RSL-RL checkpoint，包括优化器和迭代状态：

```bash
python scripts/rsl_rl/train.py --headless --resume \
  --checkpoint /absolute/path/to/model_1000.pt
```

也可以用 `--resume --load_run RUN_DIRECTORY` 在日志目录内查找模型。

## 回放与录制

```bash
python scripts/rsl_rl/play.py \
  --checkpoint /absolute/path/to/model_1000.pt --real-time
```

默认回放一个球场，双方使用同一个模型的确定性动作。省略 `--checkpoint` 时，从实验日志目录查找最近的 checkpoint。

录制视频前安装视频依赖：

```bash
python -m pip install -e 'source/badmintonlab_game[video]'
python scripts/rsl_rl/play.py --headless \
  --checkpoint /absolute/path/to/model_1000.pt \
  --video --video_length 500
```

回放视频保存到 checkpoint 所在目录的 `videos/play/`。

## 环境与奖励

- T1 使用关节位置目标控制，包含执行器延迟和随速度变化的力矩限制。
- 物理步长为 0.005 秒，每 4 个物理步更新一次动作，即 50 Hz 控制频率。
- 羽毛球用带碰撞的球体近似，每个物理步施加重力和二次空气阻力。
- 每回合随机选择先击球方，将球静止放置在其半场上方，由机器人完成开球。
- 球拍接触检测覆盖整个控制步的物理子步，只有轮到的玩家能记录有效击球。
- 落地、出界、下网结束回合；最长回合为 10 秒。摔倒本身不会直接判负。
- 结果奖励为胜负 ±10；获得正奖励需要本方至少完成一次有效击球。超时没有额外胜负奖励。
- 辅助代价包括接近来球、出球质量、身体高度和朝向、球拍撑地、关节限位、动作变化及脚部滑动。

## 代码结构

```text
scripts/rsl_rl/                 # PPO 训练、回放和共享命令行参数
source/badmintonlab_game/
  config/extension.toml        # Isaac Lab 扩展元数据
  badmintonlab_game/
    assets/                    # 球场、T1 与执行器配置
    tasks/manager_based/badmintonlab_game/
      dual_agent_env.py        # 物理推进、击球、判定与重置
      dual_agent_env_cfg.py    # 观测、动作、奖励与终止配置
      badminton_scene_cfg.py   # 双机器人仿真场景
      shared_policy_wrapper.py # N 个球场转换为 2N 份玩家样本
      agents/                  # PPO 配置
      mdp/                     # 观测、奖励、重置和终止函数
robots/T1/                     # URDF 及其引用的网格
tests/                         # 无需启动 Isaac Sim 的接口测试
```

## 开发检查

在已有 PyTorch 环境中安装开发依赖：

```bash
python -m pip install pytest ruff gymnasium tensordict packaging
python -m pytest
ruff check .
ruff format --check .
```

接口测试验证自对弈样本、动作、奖励和终止信号的对应关系，以及 checkpoint 加载 API 适配。测试用最小环境替身隔离仿真依赖，不验证 PhysX 碰撞、轨迹或训练收敛；完整运行需要上面的仿真冒烟检查。

整理过程中，包重命名前已在 Isaac Lab 0.46.3、RSL-RL 3.0.1、PyTorch 2.7.0 环境完成 2 个球场、1 次 PPO 迭代的冒烟检查。重命名后仅执行静态和打包检查，未重新启动仿真。

## 精简说明

当前版本只保留共享策略自对弈。联赛、固定对手、PFSP、exploiter、蒸馏和历史任务入口已移除。原来的 Cap 任务可用 `env.process_reward_cap=10` 替代。回放与恢复使用原生 RSL-RL checkpoint，不再加载联赛专用的跨版本拆分模型格式。

源码中已有的第三方版权和许可证声明予以保留。本轮未新增项目许可证。
