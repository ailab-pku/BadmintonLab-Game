# BadmintonLab-Game-Play

`play` 分支加载学生提交的 TorchScript 编译模型，进行双机器人对战。策略须使用神经网络从观测端到端输出关节控制动作。

球场两侧各放置一台机器人，评测方统一配置对战环境，学生提交自己的 `player_cfg.py` 和 TorchScript 模型。每个模型只接收对应玩家配置所定义的观测，执行确定性推理。

## 安装

环境要求：

必须使用 **Isaac Lab 2.3.2** 和 **Isaac Sim 5.1.0**，不支持其他版本。

- Python 3.11
- Isaac Lab 2.3.2
- Isaac Sim 5.1.0
- PyTorch 2.7

从项目根目录执行：

```bash
python -m pip install -e source/badmintonlab_game
```

## 提交要求

每位同学需提交以下两个文件：

- `player_cfg.py`：包含 `ObsCfg` 和 `EventsCfg`，定义模型使用的观测以及自身机器人的 root 初始位姿、关节初始状态。可参考 `source/badmintonlab_game/badmintonlab_game/tasks/manager_based/badmintonlab_game/` 下的 `player0_cfg.py` 和 `player1_cfg.py`；自定义观测或初始化所需的 MDP 函数也应包含在该文件中。
- 模型文件（`.pt`）：可由 `torch.jit.load()` 加载的 TorchScript 模型，其输入观测的内容、顺序、维度和预处理须与提交的 `player_cfg.py` 一致。

评测方将双方提交的配置分别接入 `player0_cfg.py` 和 `player1_cfg.py`，按对应玩家侧设置资产引用及镜像坐标，再通过下述命令加载双方模型。`player_cfg.py` 仅用于观测及自身机器人初始化，不得修改公共物理参数、对手配置及比赛规则。

## Play

默认以无窗口模式运行，不进行实时画面渲染。

```bash
python scripts/rsl_rl/play.py \
  --player0_checkpoint /absolute/path/to/player0.pt \
  --player1_checkpoint /absolute/path/to/player1.pt
```

默认任务为 `BadmintonLab-Game-Play`。双方均须提交可由 `torch.jit.load()` 加载的 TorchScript 模型（`.pt`），通过 `torch.jit.script()` 或 `torch.jit.trace()` 导出，再用 `torch.jit.save()` 保存。训练 checkpoint、`state_dict` 或普通 `torch.save(model)` 文件不能直接提交。

模型接口为 `forward(observations) -> actions`：输入形状为 `(N, observation_dim)`，输出为 `(N, action_dim)`，其中 `N` 为球场数。当前默认配置每方观测为 120 维、动作为 23 维；双方模型分别接收自身观测并独立推理。模型须支持不同批量大小，输出有限浮点数，并在模型内部包含训练时使用的固定观测归一化及必要的输入预处理。


## 导出对战视频

`render.py` 专门用于无窗口渲染和导出 MP4，不影响 `play.py` 的纯物理回放。先安装视频编码依赖：

```bash
python -m pip install -e 'source/badmintonlab_game[video]'
```

```bash
python scripts/rsl_rl/render.py \
  --player0_checkpoint /absolute/path/to/player0.pt \
  --player1_checkpoint /absolute/path/to/player1.pt \
  --video_length 1000 \
  --output outputs/videos/match.mp4
```

脚本固定渲染一个球场，自动启用 headless 相机渲染。默认分辨率为 1280×720，每个控制步记录一帧；当前环境为 50 FPS，1000 帧对应 20 秒仿真时间，期间回合结束会按环境规则自动重置。


## 环境规则

`play.py` 和 `render.py` 均支持 `--num_rallies N`（默认 100），每个球场独立进行 N 个回合，每回合胜方得 1 分。打满后比较总得分，得分高者获胜，同分为平局。

场内落地时，球所在半场的玩家输；出界、下网时，最后击球方输；发球超高或二次击球时，违例方输。开球下落后无人击球则开球方输。

- 策略控制频率为 50 Hz。
- 每回合随机选择先击球方，将球静止放置在其半场上方，自由下落，由机器人完成开球，开球击球高度不得超过1.15m。
- 落地、出界或下网结束回合并自动重置，不设置回合超时，摔倒本身不会直接判负。
- 不允许二次击球。

开球高度按球拍接触时球心相对球场地面的高度判定，超过 1.15 m 结束回合。
