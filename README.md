# BadmintonLab-Game-Play

`play` 分支加载学生提交的 TorchScript 编译模型，进行双机器人对战。策略须使用神经网络从观测端到端输出关节控制动作。

球场两侧各放置一台机器人，评测方统一配置环境及双方的 `player_cfg.py`，学生仅提交模型文件。每个模型只接收对应玩家的观测，执行确定性推理。

## 安装

环境要求：

本项目仅针对 Isaac Lab 2.3.2 和 Isaac Sim 5.1.0 开发与验证，不维护旧版本兼容逻辑。

- Python 3.11
- Isaac Lab 2.3.2
- Isaac Sim 5.1.0
- PyTorch 2.7

从项目根目录执行：

```bash
python -m pip install -e source/badmintonlab_game
```

## Play

默认以无窗口模式运行，不进行实时画面渲染。

```bash
python scripts/rsl_rl/play.py \
  --player0_checkpoint /absolute/path/to/player0.pt \
  --player1_checkpoint /absolute/path/to/player1.pt
```

默认任务为 `BadmintonLab-Game-Play`。双方均须提交可由 `torch.jit.load()` 加载的 TorchScript 模型（`.pt`），通过 `torch.jit.script()` 或 `torch.jit.trace()` 导出，再用 `torch.jit.save()` 保存。训练 checkpoint、`state_dict` 或普通 `torch.save(model)` 文件不能直接提交。

模型接口为 `forward(observations) -> actions`：输入形状为 `(N, observation_dim)`，输出为 `(N, action_dim)`，其中 `N` 为球场数。当前默认配置每方观测为 120 维、动作为 23 维；双方模型分别接收自身观测并独立推理。模型须支持不同批量大小，输出有限浮点数，并在模型内部包含训练时使用的固定观测归一化及必要的输入预处理。回放按无状态方式调用模型，不提供隐藏状态或回合重置信号。

控制动作必须由神经网络端到端产生，不允许使用手写规则、条件判断、状态机、轨迹规划或其他控制器生成或切换动作，包括将这些逻辑封装进 TorchScript 模型。提交物不包含额外的推理代码、回调或自定义算子。上述要求是作业规范；TorchScript 格式本身不能自动保证模型内部不含规则逻辑。

## 环境规则

- 策略控制频率为 50 Hz。
- 每回合随机选择先击球方，将球静止放置在其半场上方，自由下落，由机器人完成开球，开球击球高度不得超过1.15m。
- 落地、出界或下网结束回合并自动重置，不设置回合超时，摔倒本身不会直接判负。
- 不允许二次击球。

开球高度按球拍接触时球心相对球场地面的高度判定，超过 1.15 m 结束回合。同一次持续接触计为一次击球；分离后再次接触且未轮到该玩家时，结束回合。

评测方可分别替换 `player0_cfg.py` 和 `player1_cfg.py` 中的 `ObsCfg` 和 `EventsCfg`，定义各自的观测、root 初始位姿及关节初始状态。需要 Isaac Lab 未提供的 MDP 函数时，可直接在对应文件中定义并引用。
