# BadmintonLab-Game

Booster T1 羽毛球机器人共享策略自对弈训练，基于 Isaac Lab 和 RSL-RL PPO。

## 安装
环境要求：

**务必使用 Isaac Lab 2.3.2 和 Isaac Sim 5.1.0。**

- Python 3.11
- Isaac Lab 2.3.2
- Isaac Sim 5.1.0
- PyTorch 2.7

```bash
python -m pip install -e source/badmintonlab_game
```

## 训练

```bash
python scripts/rsl_rl/train.py --headless
```

默认任务为 `BadmintonLab-Game`。可以先用较少球场进行冒烟检查：

```bash
python scripts/rsl_rl/train.py --headless --num_envs 4 --max_iterations 2
```
