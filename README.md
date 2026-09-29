# 有限时域离线策略评估：PDIS / SNIS / 序列 Doubly Robust

以**完整 episode 为独立采样单位**的有限时域（finite-horizon）离线策略评估（OPE）实验。
核心估计量全部自行实现（`src/ope/estimators.py`），真值由**已知转移模型上的后向动态规划**
独立给出（`src/ope/mdp.py: dp_evaluate`），不使用数据拟合值冒充真值。

- 估计量：per-decision importance sampling（PDIS）、轨迹级自归一化 IS（SNIS）、
  序列/加权 doubly robust（DR, WDR）
- 交叉拟合：按 **episode** 划分 K 折；评估折 k 的 Q/V 只用其余 K−1 折拟合；
  **同一 episode 的任何片段都不会同时出现在训练折和评估折**
- 诊断：累计权重的 log 空间稳定计算、每步 ESS、权重截断（显式记录）、
  episode 层级 bootstrap（经验不确定性，非有限样本覆盖保证）
- 场景：良好覆盖 / 行为概率极小导致权重退化 / 目标可达但行为零覆盖（标记为不可识别）
- 对照：故意错误的 Q 模型（myopic、Q≡0）；一次训练/评估泄漏反例

## 目录

```
src/ope/
  mdp.py         有限时域表格 MDP；后向 DP 真值（独立于数据）
  data.py        以 episode 为单位采集完整轨迹（含记录的行为概率 log_mu）
  weights.py     log 策略比率、累计权重、截断、支持度检验（不做 epsilon 修补）
  estimators.py  PDIS / SNIS / 序列DR / ESS（核心，手写，log-sum-exp 稳定）
  nuisance.py    表格 Q/V：经验拟合、oracle(DP)、myopic、零 四种
  crossfit.py    episode 级 K 折交叉拟合、统一评估、泄漏版 in-sample
  bootstrap.py   整段 episode 重采样的非参数 bootstrap
experiments/
  scenarios.py        三个（+泄漏用）解析 MDP
  run_experiments.py  多次重新采集：偏差/RMSE/覆盖/区间宽度/蒙特卡洛误差
  run_leakage.py      训练评估泄漏反例
  make_report.py      汇总 results/*.json 为 results/RESULTS.md
tests/test_core.py   真值、无偏恒等式、DR 条件、支持度、折划分、数值稳定
```

## 1. 问题设定与目标量

有限时域 H，折扣 γ，状态 S、动作 A。行为策略 μ 下采集轨迹

```
τ = (s0, a0, r0, s1, a1, r1, ..., s_{H-1}, a_{H-1}, r_{H-1}),  s0 ~ p0,
a_t ~ μ(·|s_t),  s_{t+1} ~ P(·|s_t,a_t),  r_t 的条件均值 R_t(s_t,a_t).
```

目标策略 π 固定。目标量为给定初始分布 p0 的折扣累计回报

```
J(π) = E_π[ Σ_{t=0}^{H-1} γ^t r_t ] = Σ_s p0(s) V^π_0(s).
```

前缀重要性比率与折扣回报：

```
ρ_{0:t}(τ) = ∏_{h=0}^{t} π(a_h|s_h)/μ(a_h|s_h),   ρ_{-1}=1,
g(τ)       = Σ_t γ^t r_t.
```

## 2. 三个估计量（实现即按此）

**PDIS（per-decision IS，Horvitz–Thompson 型）**

```
J_PDIS = (1/n) Σ_i Σ_t γ^t ρ^i_{0:t} r^i_t.
```

**轨迹级 SNIS（自归一化 IS）**

```
J_SNIS = Σ_i ρ^i_{0:H-1} g_i / Σ_i ρ^i_{0:H-1}.
```

**序列 / 加权 Doubly Robust（WDR, Thomas & Brunskill 2016）**

给定 Q/V nuisance 模型 `Q̂_t(s,a)`，令

```
V̂_t(s)   = Σ_a π(a|s) Q̂_t(s,a),     Q̂_H(·)=0,
δ^i_t    = r^i_t + γ Q̂_{t+1}(s^i_{t+1}) − Q̂_t(s^i_t,a^i_t)   (Bellman 残差),
J_DR     = (1/n) Σ_i [ V̂_0(s^i_0) + Σ_t γ^t ρ^i_{0:t} δ^i_t ].
```

完整代数推导、两类正确性条件与偏差分解见 [`DERIVATIONS.md`](DERIVATIONS.md)。这里给出实现所需的核心：

它可以做望远镜展开：把 `r_t = Q_t − γ V_{t+1} + δ_t` 代入 PDIS 后求和，
所有“纯权重×价值”项逐层相消，在 Q̂=Q^π、V̂=V^π 时恰好剩下 `V̂_0(s0)` 这一项，
这正是代码里 `dr()` 的形式（`src/ope/estimators.py`）。

### 2.1 无偏/一致性的成立条件（两类“正确性”必须区分）

**（A）行为概率 μ 准确（重要性采样这条腿成立）**

- 条件：记录的 μ(a|s) 真实，且满足**正支持（positivity）**：
  对所有 π(a|s)>0 的、π 可达的 (s,a) 有 μ(a|s)>0；nuisance 由**交叉拟合**给出
  （评估折的 Q̂ 独立于评估折的数据，从而 `E[ρ δ | 评估折]` 中 Q̂ 对该折为常数）。
- 结论：
  - PDIS **严格无偏**：`E_μ[ρ_{0:t} r_t]=E_π[r_t]`（逐决策的测度变换）。
  - 序列 DR **严格无偏**：因 `E_μ[ρ_{0:t} δ_t | P_t]=δ_t·π/μ` 的逐层期望为 0
    （`E_μ[δ_t|P_t]=0` 对**任意** Q̂ 都成立，见下），交叉拟合使该条件期望合法。
    即 **μ 准确时，DR 不需要 Q 模型对也无偏**（Q 错只增大方差）。
  - SNIS **不无偏但一致**：分子分母都无偏，比值是 O(1/n) 偏差；大权重时
    有限样本偏差可能很大。

**（B）价值模型 Q 准确（回归这条腿成立）**

- 条件：`Q̂_t = Q^π_t`（从而 `V̂=V^π`）；此时对每条轨迹 Bellman 残差
  `δ_t = r_t + γQ^π_{t+1}(s_{t+1}) − Q^π_t(s_t,a_t)` 满足
  `E[δ_t | s_t,a_t] = 0`（期望意义，随机奖励下单条轨迹的 δ 不为零）。
- 结论：
  - 权重项 `Σ_t γ^t ρ_{0:t} δ_t` 的期望为 0，于是 `E[J_DR]=E_{s0}[V^π_0(s0)]=J(π)`；
    当模型与转移都确定时，单条轨迹 δ 恒为 0，DR **逐样本精确**（测试
    `test_dr_exact_q_is_exact_deterministic`）。
  - 这条腿**不依赖 μ 估计是否精确**（只要比率良定义、有限），是 DR 的第二条命。
  - 若 μ 错误但 Q 准确，PDIS/SNIS 会偏，DR 仍准。

**“双重稳健”的准确说法**：上面两条腿满足其一时，DR 即无偏/一致（依赖交叉拟合
消除一阶拟合偏差）；两条都满足时方差通常更小。**两者都错时 DR 也会偏**，偏差
近似 `E_μ[ρ_{0:t} ε_t]` 一阶项，其中 ε_t 是 Q̂ 的 Bellman 误差；不能用“稳健”
掩盖它。截断、SNIS 归一化、模型外推都会带来各自的偏差，下面分别量化而不隐藏。

### 2.2 三类偏差来源（实验中显式暴露，不称作“稳健”）

- **权重截断** `ρ ↦ min(ρ, c)`：把大权重压到 c，减小方差但引入
  `E[(ρ−c)_+ r]` 方向的偏差；c 在结果中显式报告（`clip10`），不做默认隐藏截断。
- **自归一化**：SNIS 的 O(1/n) 比值偏差，权重退化时偏差占主导（场景2可见）。
- **模型拟合误差**：错误 Q（myopic/zero）仍无偏（μ 准确 + 交叉拟合），但 RMSE
  增大；未交叉拟合时还有额外的一阶拟合偏差与乐观方差（泄漏反例）。

## 3. Episode 级交叉拟合（禁止片段泄漏）

`episode_kfold` 把 **n 个 episode** 随机分到 K 折；`crossfit_q` 对折 k 只用
`folds != k` 的**整段 episode** 拟合 Q/V，再在折 k 的 episode 上评估。轨迹的
任意时间步都不可能一半在训练、一半在评估。测试 `test_fold_assignment_is_episode_level`
与 `test_crossfit_uses_disjoint_episodes` 固化该不变量。

## 4. 零覆盖：不可识别，拒绝修补

若存在 π 可达且 π(a|s)>0 但 μ(a|s)=0 的 (s,a)，则重要性比率为无穷、目标值
**不被行为分布识别**。`weights.population_support_check` 在模型层检出这些三元组，
`support_violations` 在数据层检出实际踩到的轨迹。评估器对 PDIS/SNIS/DR 一律
返回 **NaN**，`note` 明确写出“NOT identified”，**不做 epsilon 平滑、不输出貌似
可信的窄区间**（场景3，见结果）。任何在零支持单元上的 Q 插件值都是**不可检验的
模型外推**，代码里 `unseen_cells` 也会如实计数。

## 5. 数值稳定、ESS、bootstrap 的正确定位

- **累计权重**：全程存 `log ρ`，累加用 cumsum；`−inf/+inf` 显式处理（零目标动作
  使前缀乘积恒为 0；零行为动作为 +inf）。求和对可能为负的被加项做正/负部
  log-sum-exp（`signed_weighted_sum`），避免 1000 个极小权重相乘下溢成 0。
- **ESS**：`ESS_t = (Σ_i w_it)^2 / Σ_i w_it^2` 在 log 空间计算，并报告每步值与
  `ESS_t/n`；取所有时间步最小值。ESS 小说明方差/自归一化偏差风险高，
  **不是正确性或覆盖的保证**。
- **Bootstrap**：按 **episode** 有放回重采样，整条管线（权重 + 交叉拟合 Q + 估计）
  重跑，取 2.5/97.5 百分位。它是**经验抽样分布的不确定性估计**，对有偏/重尾估计量
  （场景2 的 PDIS/SNIS）覆盖可能远低于名义值；**不声称任何无条件有限样本覆盖保证**。

## 6. 三个解析 MDP

H=4、γ=0.9、S=A=2、确定性转移（动作1 使 s:0→1 并保持，动作0 停留0），
伯努利奖励均值 `R(s,a)`（s0 确定从状态0出发）。目标策略恒定动作1，
真值（DP）`J(π)=R(0,1)+γR(1,1)+γ²R(1,1)+γ³R(1,1)`。

- 场景1 `good_coverage`：μ(动作1)=0.8，比率温和，`min ESS/n≈0.4`。
- 场景2 `tiny_behavior`：μ(动作1)=ε（实验取 ε=0.12；另设工厂参数 1e-3 的极端退化点）。
  每步 ESS/n 中位数从约 0.12 塌缩到 ~0.013→0.003→0.0005；轨迹级权重约 (1/ε)^4≈4.8e3。
  在 n=300 下：SNIS 仅约 7.5% 的重复能采到全合规轨迹从而给出有限值，其余 NaN；
  PDIS 近似无偏但 RMSE≈3.18、覆盖仅约 0.45。
  - 截断对照 `clip=20`：把前缀权重压到 ≤20 后，PDIS 偏差变为约 −1.72、区间中位宽仅约 0.56、
    覆盖 0——**截断用偏差换方差，偏差没有被“稳健”掩盖**；DR 仍部分依赖 Q 这条腿（RMSE≈0.39）。
- 场景3 `no_coverage`：μ(动作1|状态0)=0，而目标在状态0以概率1选动作1；
  真值对模型存在但**数据不可识别**，仅报告检出率与缺失三元组。

（奖励均值与策略见 `experiments/scenarios.py`，真值 `python experiments/scenarios.py`。）

### 6.1 训练/评估泄漏反例（`run_leakage.py`）

稀疏目标单元（μ(动作1)=0.10、n=120；见 `run_leakage.py` 默认参数）下对比两条流水线：
诚实的 episode 交叉拟合，与“全量 episode 训练 Q、再在**同一批** episode 上评估”的泄漏版。
后者让每条轨迹的 `(s,a,r,s')` 进入用于评估它自己的单元均值，Bellman 残差被人为缩小，
估计量塌缩到有偏的模型插件值，bootstrap 区间随之变窄。实测（120 次重复、199 bootstrap）：

- 诚实交叉拟合：点估计近似无偏（偏差≈0 量级）、真实重复采样 SD 与 bootstrap SE 同量级、
  覆盖明显更高、区间更宽。
- 泄漏：偏差约 −0.5、区间中位宽约 1.27（诚实约 2.2）、覆盖仅约 0.35；
  `mean_bootstrap_se`≈0.46 而真实 `sd_of_estimates`≈0.87——**声称的精度约为真实的一半**，
  这就是“看起来更精确、实际过度乐观且欠覆盖”的反例。

注意即使诚实流水线，在如此稀疏/高方差数据下覆盖也低于 95%（约 0.6），这本身说明
bootstrap 百分位区间**不是**无条件有限样本覆盖保证；泄漏只是让欠覆盖进一步恶化。

## 7. 复现

```bash
pip install numpy pytest
python -m pytest tests/test_core.py -q

python experiments/scenarios.py                       # DP 真值
python experiments/run_experiments.py                 # 全量：120重复×199bootstrap×n300
python experiments/run_experiments.py --quick         # 快速冒烟
python experiments/run_leakage.py
python experiments/make_report.py                     # 生成 results/RESULTS.md
```

输出：`results/<scenario>.json` 含每估计器均值、偏差与 RMSE 的蒙特卡洛标准误、
覆盖频率与覆盖二项标准误、区间中位宽、有限值比例、ESS；`results/RESULTS.md` 汇总。
