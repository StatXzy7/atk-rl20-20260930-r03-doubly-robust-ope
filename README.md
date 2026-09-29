# 有限时域离线策略评估（Finite-Horizon OPE）实验

从零实现并比较三个离线策略评估估计器：**per-decision importance sampling
(PDIS)**、**per-decision 自归一化 IS (SNIS)**、**序列 doubly robust (DR)**。
数据以**完整 episode** 为独立采样单位；目标策略固定；目标量是给定初始状态分布下的
折扣累计回报 $J(\pi)$。核心估计器、权重、交叉拟合与 bootstrap 全部自行实现（仅依赖
numpy）。

## 目录结构

```
fh_ope/
  mdp.py          # 3 个可解析 MDP + 已知核上的向后 DP 真值、目标占据度
  data.py         # 向量化 episode 采集、episode 折划分、可识别性诊断
  weights.py      # log 域累积权重、截断、log-sum-exp ESS 诊断
  models.py       # 表格 Q/V nuisance（正确 / action-blind 错误 / oracle）
  estimators.py   # PDIS、SNIS、序列 DR（含截断、交叉拟合封装）
  bootstrap.py    # episode 层级 bootstrap（DR 每个 draw 重新交叉拟合）
  experiments.py  # 多次重采数据：bias/RMSE/覆盖/MCSE
tests/test_core.py
docs/derivations.md  # 公式、递推、成立条件、偏差来源的完整推导
run_experiments.py   # 入口：results/results.{json,md}
```

## 运行

```bash
python run_experiments.py        # 约 10-20 分钟，结果写入 results/
python -m pytest tests/ -q       # 约 5 秒（全部向量化后）
```

## 三个环境与真值

- `good_coverage`：行为/目标全支撑、比值温和。
- `tail_risk`：行为几乎确定、目标偏稀有动作，逐步比值 16，累积权重爆炸、ESS≈1；
  用于展示权重退化、截断偏差（方向已知为向下）、SNIS 偏差。
- `no_coverage`：目标在状态 1 取行为**严格零概率**的动作。提供两个观测数据完全相同、
  目标真值不同（0.3 vs 1.2）的变体；程序**标记为不可识别，不输出点估计/窄区间**，
  不用 epsilon 修补。

## 关键纪律（详见 docs/derivations.md）

- PDIS 无偏需要**精确行为概率 + 覆盖**；SNIS 有限样本有偏但一致；DR 在
  props 精确时对任意固定 Q 无偏，Q 精确时对 props 不敏感——两条腿分开陈述。
- **截断有偏**（即使 props 精确），实验同时报告 raw/clip 的 bias 与 RMSE。
- **交叉拟合按整条 episode 划分**，评估折 episode 不进入其自身 Q 训练。
- bootstrap 单位是 episode；DR 在每个 bootstrap 样本内重新交叉拟合；区间是经验
  不确定性，覆盖率靠重采数据实测，**不**宣称无条件有限样本覆盖。
- 对照：action-blind 错误 Q；以及训练/评估泄漏（in-sample DR）导致区间过度乐观。
