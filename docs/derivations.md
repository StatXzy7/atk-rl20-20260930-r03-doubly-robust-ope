# 有限时域离线策略评估：估计器推导、成立条件与偏差来源

记号：episode $\tau=(s_0,a_0,r_0,s_1,\dots,s_H)$，阶段 $t=0,\dots,H-1$；折扣
$\gamma\in(0,1]$；目标策略 $\pi(a\mid s)$，行为策略 $\mu(a\mid s)$（日志中记录为
$p_{\text{behavior}}$）。数据 $\{\tau_i\}_{i=1}^n$ 以**整条 episode** 为独立采样单位，
$\tau_i\sim \mu$。目标量

\[
J(\pi)=\mathbb{E}_{s_0\sim \mu_0,\,\pi}\Big[\sum_{t=0}^{H-1}\gamma^t r_t\Big]
      =\mu_0^\top V^\pi_0 .
\]

定义**逐步（per-decision）累积重要性权重**

\[
\rho_{0:t}(\tau)=\prod_{k=0}^{t}\frac{\pi(a_k\mid s_k)}{\mu(a_k\mid s_k)}.
\]

所有权重在代码中以对数域 $\ell_{0:t}=\sum_{k\le t}(\log\pi-\log\mu)$ 保存与累加，
最后才指数化。三种比值的约定（见 `fh_ope/weights.py`）：

- $\pi>0,\mu>0$：正常对数比值；
- $\pi=0,\mu>0$：权重为 $0$（轨迹一旦偏离确定性目标即精确为 0）；
- $\pi>0,\mu=0$：$+\infty$，**这是致命的不可识别情形，不用 epsilon 修补**；
- $\pi=0,\mu=0$：约定 $0/0=1$（两个策略都不取的动作与估计无关）。

---

## 1. Per-Decision Importance Sampling（PDIS）

\[
\widehat J_{\mathrm{PDIS}}=\frac1n\sum_{i=1}^n\sum_{t=0}^{H-1}\gamma^t
    \rho_{0:t}(\tau_i)\,r_{t,i}.
\]

**推导（无偏性）。** 记历史 $h_t=(s_0,a_0,\dots,s_t)$。由逐步比值的期望为 1 的
鞅性质，对任意可测函数 $g_t(h_t,a_t,r_t)$：

\[
\mathbb E_\mu\!\Big[\frac{\pi(a_t\mid s_t)}{\mu(a_t\mid s_t)}g_t\,\Big|\,h_t\Big]
=\mathbb E_\pi[g_t\mid h_t].
\]

把 $\rho_{0:t}$ 拆成 $\rho_{0:t-1}\cdot \pi(a_t\mid s_t)/\mu(a_t\mid s_t)$，对
$t,t-1,\dots,0$ 反复套用期望换测度，得到

\[
\mathbb E_\mu[\gamma^t\rho_{0:t}r_t]
=\gamma^t\,\mathbb E_{\pi}[r_t].
\]

对 $t$ 求和即 $\mathbb E_\mu[\widehat J_{\mathrm{PDIS}}]=J(\pi)$。

**成立条件（严格）**

1. $\mu(a_t\mid s_t)>0$ 对所有 $\pi$ 可能取到、且在目标可达状态上的 $(t,s_t,a_t)$
   成立（**正支撑/覆盖假设**，positivity）。若 $\pi(a\mid s)>0$ 而 $\mu(a\mid s)=0$，
   该项期望无定义，PDIS 不能估计 $J(\pi)$。
2. 记录的 $\mu$ **精确**（无函数逼近/估计误差），且 $\pi$ 已知。
3. episode 间独立（或至少加权和期望可分解）。
4. 不需要任何关于转移核或奖励的模型。

**方差。** 每集是 $H$ 个高度相关项之和；累积权重随 $t$ 相乘，方差随时间指数增长
（“权重退化”）。MDP B 中逐步稀有动作比值为 $0.8/\varepsilon$（$\varepsilon=0.05$
时为 16），$H=5$ 时最大累积权重 $16^5\approx 1.05\times10^6$，ESS 常接近 1。
**权重截断见第 4 节，它以偏差换方差，不是无偏的。**

---

## 2. 自归一化重要性采样（SNIS，per-decision 形式）

代码采用 per-decision 池化自归一化（`snis_parts`）：

\[
\widehat J_{\mathrm{SNIS}}
=\frac{\frac1n\sum_i\sum_t\gamma^t\rho_{0:t}^i r_t}
       {\frac1n\sum_i\sum_t\gamma^t\rho_{0:t}}
=\frac{\widehat J_{\mathrm{PDIS}}}
       {\underbrace{\big(\textstyle\sum_t\gamma^t\big)\cdot
        \overline Z}_{\widehat D}},\qquad
Z_{i,t}=\rho_{0:t}^i .
\]

注意分母里的折扣归一常数 $G=\sum_t\gamma^t$ **必须除掉**：在 $\pi=\mu$ 时
$Z\equiv1$，分母原始加权和等于 $G$ 而非 1；若漏掉 $G$，估计会被整体缩小 $G$ 倍
（这是 IS 实现里常见的 off-by-normalizer bug，`tests/test_core.py` 有回归检查）。

**性质**

- **有偏但一致。** 分母是随机量；$\mathbb E[\mathrm{SNIS}]\ne J$（有限样本有偏，
  偏差量级通常 $O(1/n)$ 或 $O(1/\mathrm{ESS})$），但在覆盖、$\mu$ 精确、独立同分布
  下由大数定律 $n\to\infty$ 时分母 $\to G$、分子 $\to J$，故 $\widehat J_{\mathrm{SNIS}}\to J$。
- **对常数尺度不变**：把所有 $\mu$ 乘同一常数不改变结果（对概率自动满足，对半概率
  密度有用），并通常显著降低方差。
- 覆盖假设与 PDIS 相同；**$\mu=0$ 同样致命**，自归一化不解决无覆盖问题。
- bootstrap 必须对每个重采样**重算分子分母之比**（`combine_snis`），不能对单点
  SNIS 值直接重采样平均。

---

## 3. 序列 Doubly Robust（DR）估计

### 3.1 定义与向后递推

给定 Q 的估计 $\widehat Q_t(s,a)$（nuisance model），定义目标策略下的 V 估计

\[
\widehat V_t(s)=\sum_a\pi(a\mid s)\,\widehat Q_t(s,a),\qquad \widehat V_H(s)=0,
\]

以及在日志转移上的 Bellman 残差（augmented inverse-propensity 修正项）

\[
\delta_{i,t}
=\underbrace{r_{t,i}+\gamma(1-d_{t,i})\widehat V_{t+1}(s'_{t,i})}_{\text{TD 目标}}
-\widehat Q_t(s_{t,i},a_{t,i}).
\]

序列 DR 估计为

\[
\boxed{\;
\widehat J_{\mathrm{DR}}
=\frac1n\sum_i\Bigg[\widehat V_0(s_{0,i})
+\sum_{t=0}^{H-1}\gamma^t\,\rho_{0:t}^i\,\delta_{i,t}\Bigg].
\;}
\]

代码实现见 `fh_ope/estimators.py:dr_parts`（向量化），早期终止由 $(1-d_t)$ 处理。

### 3.2 为什么它是 Bellman 方程的“增广”形式（直观）

只用模型会给出 $\widehat V_0(s_0)$；DR 在每一步用观测到的 TD 目标对模型做
**加权残差修正**。逐步比值把“在 $\mu$ 下观测到的 $(s,a,r,s')$ 修正”翻译成
“$\pi$ 下该状态-动作的修正”。当 $\widehat Q=Q^\pi$ 时，由 Bellman 方程
$Q^\pi_t(s,a)=\mathbb E[r+\gamma V^\pi_{t+1}(s')\mid s,a]$，残差条件期望为 0，
修正项逐样本（在期望意义）消失，估计回到 $V^\pi_0$。

### 3.3 性质一：行为概率精确时，Q 任意都无偏

**命题（props 正确 $\Rightarrow$ 无偏）。** 设 $\mu$ 精确、覆盖成立，$\widehat Q$ 是
**任意固定（不依赖当前评估数据）** 的函数，则

\[
\mathbb E_\mu[\widehat J_{\mathrm{DR}}]=J(\pi).
\]

**证明（按阶段条件期望）。** 固定 $\widehat Q$，令
$\widehat V_t=\sum_a\pi(a\mid s)\widehat Q_t(s,a)$。对第 $t$ 个修正项，关于
$a_t$ 取条件期望：

\[
\mathbb E_\mu\!\Big[\tfrac{\pi(a_t\mid s_t)}{\mu(a_t\mid s_t)}
   \big(r_t+\gamma(1-d_t)\widehat V_{t+1}(s_{t+1})-\widehat Q_t(s_t,a_t)\big)
   \,\Big|\,h_t\Big]
\]
\[
=\sum_a\pi(a\mid s_t)\,
   \mathbb E\!\big[r_t+\gamma(1-d_t)\widehat V_{t+1}(s')-\widehat Q_t(s_t,a)
                 \mid s_t,a_t=a\big]
=\sum_a\pi(a\mid s)\,\widehat T_t(s,a)-\widehat V_t(s),
\]

其中 $\widehat T_t(s,a)=\mathbb E[r+\gamma(1-d)\widehat V_{t+1}(s')\mid s,a]$
（用 $\widehat V$ 做后继评估）。把所有阶段用 $\rho_{0:t-1}$ 加权后**望远镜求和**：
阶段 $t$ 的 $-\widehat V_t$ 与阶段 $t-1$ 修正项期望中出现的
$+\gamma\widehat V_t$ 抵消，链一直缩到起点，最后剩下
$\mathbb E_{\mu_0}[\widehat V_0]+\big(V^\pi_0-\widehat V_0\big)$ 沿 $\mu_0$ 的项；
在 $\mu_0$ 即目标初始分布（本项目的设定）下净结果为 $J(\pi)$。因此 **$\widehat Q$
无论对错，只要 $\mu$ 精确且修正项不使用被评估数据自身拟合的模型，DR 即无偏**。
代码用 K 折交叉拟合保证后者（第 5 节）。

### 3.4 性质二：Q 精确时，props 错也成立（“第二条腿”）

**命题（Q 精确 $\Rightarrow$ 逐样本等于 $J$ 的期望）。** 若
$\widehat Q_t=Q^\pi_t,\ \widehat V_t=V^\pi_t$（由已知核动态规划得到的 oracle 模型，
或模型一致收敛），则对每条轨迹 $\delta_t$ 是 Bellman 残差，其条件期望为 0，于是

\[
\widehat J_{\mathrm{DR}}
=\frac1n\sum_i\widehat V_0(s_{0,i})\ \xrightarrow{\text{a.s.}}\ J(\pi),
\]

且**不依赖权重是否正确**（甚至可令 $\rho=1$，退化为纯模型估计，依然一致）。
确定性奖励 + 精确 Q 时，每个单集 DR 值都恒等于 $J$
（`test_dr_exact_with_oracle_q_deterministic`）；随机奖励下单集值有波动，但均值无偏
（`test_dr_exact_with_oracle_q`）。

### 3.5 “双重稳健”精确到什么程度、不保证什么

- **有限样本：无免费午餐。** 上面的无偏性需要 $\mu$ 精确 **且** nuisance 独立于
  评估数据（交叉拟合）。两条“腿”是：(i) props 精确时对任意固定 Q 无偏；
  (ii) Q 精确时对 props 不敏感。两者皆错时估计一般有偏。
- **渐近一阶稳健**：在标准正则条件下，DR 的渐近偏差对 Q 的误差是**二阶**
  $O(\|\widehat Q-Q^\pi\|^2)$（AIPW 式的 Neyman 正交矩），而纯模型插补是一阶。
  这是“doubly robust”名称的技术含义——**不是**“自动低方差”或“天然稳健区间”。
- **方差并不小**：DR 仍含 $\rho_{0:t}$，权重爆炸时方差同样爆炸；Q 精确能让大权重
  乘以近零残差从而大幅降方差（这是 DR 在 MDP B 上通常优于 PDIS 的原因），但这依赖
  Q 确实准。
- **截断使 DR 有偏**：对 $\rho_{0:t}$ 截断破坏无偏恒等式，即使 props 精确也有偏。

### 3.6 成立条件清单

1. 覆盖：对修正项 $\rho\,\delta$ 有定义所需（与 PDIS 同）。Q 精确那条腿在无覆盖
   单元上能补，但**未观测单元的 Q 无法从数据估计**（MDP C）。
2. $\mu$ 精确（性质一），或 $\widehat Q\to Q^\pi$（性质二）。
3. 交叉拟合：评估折的 nuisance 仅由其他折训练，避免经验过程相关导致的偏差。
4. $\pi$、$\gamma$、$\mu_0$ 已知；episode 独立。

---

## 4. 权重截断、自归一化与拟合误差的偏差——不隐藏在“稳健”里

### 4.1 累积权重截断（clipping）

把 $\rho_{0:t}$ 截断为 $\min(\rho_{0:t},M)$（代码在 log 域截断，`weights_from_log`）。

- **PDIS/SNIS/DR 全部变成有偏**，即使 $\mu$ 精确。截断方向在本项目的 MDP B 中是
  **确定可知的**：目标把更多质量放在稀有动作上，稀有轨迹被向下截断，故
  $\widehat J_{\text{clip}}<J(\pi)$（测试 `test_tail_risk_pdis_variance_huge` 显式
  断言 `bias_clip < 0`）。一般情况下方向取决于奖励与权重的协相关，不能只看“更稳”。
- 截断是**显式的偏差–方差权衡**：实验同时报告 raw 与 clip 的 bias/RMSE。RMSE 可能
  变好（方差大降），但 bias 行明确显示代价；不能把低 RMSE 叙述成“无偏且稳健”。
- 截断作用于**累积**权重 $\rho_{0:t}$，不是逐步权重；逐步截断会改变比值期望为 1 的
  鞅性质且解释更复杂，本项目不采用。

### 4.2 自归一化的偏差

SNIS 有限样本有偏（第 2 节），偏差在 ESS 小时不可忽略；MDP B 上它可能给出表面平滑
但有偏的点估计。实验同时给出 PDIS（无偏、高方差）与 SNIS（有偏、低方差），不把
SNIS 简单标为“更好”。

### 4.3 拟合误差与交叉拟合

- 用错误 Q（action-blind，把动作池拟合）时，性质一仍保证 **props 精确前提下无偏**，
  但修正项方差变大；纯模型部分是错的，DR 靠加权修正纠偏。这是
  `DR_crossfit_actionBlind` 对照行。
- 若不交叉拟合（在同一批数据上拟合 $\widehat Q$ 又用其残差），经验残差因拟合而
  系统性偏小（模型“记住”了噪声），DR 修正被压缩，模型偏差被隐藏。见第 7 节反例。

---

## 5. Episode 级交叉拟合（禁止 episode 拆分）

`EpisodeDataset.folds(K)` 只对长度为 $n$ 的 **episode 维**做随机划分
（`np.array_split` 一个打乱后的 episode 索引），每折大小相差至多 1。
`fit_on_folds` 对评估折 $k$ 只用 `labels != k` 的**整条 episode** 训练 Q；
`crossfit_dr` 用该模型给折 $k$ 的 episode 打分。

关键不变量（有测试）：

- 每个 episode 的**所有**时间步属于同一折；不存在把同一 episode 的前半段放进训练折、
  后半段放进评估折的情况。episode 是独立采样单位，时间步之间不独立，跨行切折会产生
  伪复制和乐观偏差。
- 评估 episode 的任何转移都不进入其自身 nuisance 模型，保证第 3.3 节所需的矩条件与
  数据独立。

---

## 6. 数值稳定、ESS 诊断、episode bootstrap

### 6.1 数值稳定

- 全程 log 权重；仅在估计器内指数化，截断在 log 域进行（`np.minimum(ell, log M)`）。
- 不对整个数组减任意最大值再“跨阶段乘积”恢复：那会破坏累积乘积的尺度一致性；一个
  相对大权重可忽略、在 float64 下溢的权重就让它为 0（它本来就被更大权重主导）。
- ESS 用 log-sum-exp 计算：
  $\log\mathrm{ESS}=2\,\mathrm{LSE}(\ell)-\mathrm{LSE}(2\ell)$，在原始权重下溢时
  仍可读。
- $\pm\infty$ 按第 1 节约定显式处理，`np.where` 避免 `inf-inf` 的 NaN。

### 6.2 有效样本量（Kish ESS）

\[
\mathrm{ESS}=\frac{(\sum_i w_i)^2}{\sum_i w_i^2},\qquad
w_i=\rho_{0:H-1}^i,
\]

只在**非零权重** episode 上统计，并报告 `frac_zero_weight`（权重精确为 0 的 episode
比例）与 `frac_weight_top10_nz`（前 10% 非零 episode 占总权重比例）。MDP B 上
ESS$/n\approx 10^{-3}$、前 10% 占 99% 以上权重——这是“估计实际上只由极少数 episode
决定”的明确诊断，而不是用“稳健估计”搪塞。

### 6.3 Episode 层级 bootstrap

- 重采样单位是**整条 episode**（`rng.integers(0,n,n)` 在 episode 维有放回抽样），
  从不在时间步维重采样。
- PDIS 对每集贡献重采样取均值；SNIS 对每个重抽样本**重算比值**；
  **DR 在每个 bootstrap 样本内部重新做 K 折交叉拟合**（`bootstrap_dr` /
  `_bootstrap_dr`），所以区间同时反映采样噪声与 nuisance 估计噪声。
- 用百分位置信区间（默认 90%）。
- **这是经验（frequentist plug-in）不确定性区间，不是无条件有限样本覆盖保证。**
  IS 权重倾斜、$\widehat Q$ 有偏、SNIS 非线性、bootstrap 用估计分布近似——都会使
  真实覆盖率偏离名义水平。覆盖率是**通过独立重采数据实测的**（第 8 节），不是假设。
  泄漏设置下区间会明显偏窄、覆盖率跌破名义值（第 7 节）。

---

## 7. 三个可解析真值的小 MDP

真值全部由**已知核上的向后动态规划**独立给出（`dp_policy_value`），并用大样本目标
策略 MC 交叉验证（`test_dp_matches_monte_carlo_target`）。

### 7.1 MDP A：`good_coverage`

$S=3,A=2,H=4,\gamma=0.9$；随机但固定的随机转移核；行为策略与目标策略（偏向动作 1）
都在所有状态-动作上有正概率，比值温和。用于验证三个估计器在理想条件下的无偏/一致、
区间覆盖接近名义值。

### 7.2 MDP B：`tail_risk`（权重退化）

两状态开关链，$H=5,\gamma=0.95$。行为几乎确定（稀有动作概率 $\varepsilon=0.05$），
目标以 0.8 取稀有动作。逐步稀有动作比值 $0.8/0.05=16$，沿目标倾向轨迹累积到 $16^5$。
所有轨迹权重严格为正（不是精确 0），但 ESS 接近 1、权重高度集中。用于展示 PDIS 的
灾难方差、SNIS/截断的偏差–方差权衡，以及 DR 在 Q 较准时如何用近零残差抵消大权重。

### 7.3 MDP C：`no_coverage`（目标可达、行为零覆盖；明确标记不可识别）

状态 $\{0:start,1:branch,2:absorbing\}$，$H=3,\gamma=0.9$。两策略在阶段 0 都取 $a_0$
到达状态 1；在状态 1 目标取 $a_1$，而**行为对 $a_1$ 的概率严格为 0**。该单元的奖励
$r^\star$（构造参数 `secret_reward`）在日志中**从未被观测**。两个变体

- `secret_reward=0.0`：$J(\pi)=0.3$；
- `secret_reward=1.0`：$J(\pi)=0.3+\gamma\cdot1.0=1.2$；

产生**完全相同**的行为策略数据（测试断言两数据集奖励逐元素相同），但目标真值不同。

`identifiability_report` 用已知核计算：
`population_uncovered` 标记 $\mu(t,s,a)=0\ \&\ \pi(t,s,a)>0$ 且 $(t,s)$ 在目标占据度
下可达的单元，并给出落在这些单元上的目标占据质量；`empirical_target_gaps` 标记数据中
零访问的目标正质量单元。实验对 MDP C **不输出点估计或窄区间**，只报告：不可识别、
缺口单元位置、两个观测等价但不同的真值，以及“任何纯数据方法最多只能恢复可观测部分
0.3”这一事实。**绝不**用 epsilon 把零概率填成小正数（那会凭空捏造从未观测的
$r^\star$），也不输出貌似可信的区间。

---

## 8. Monte Carlo 评估协议与指标

`run_experiments.py` 对每个环境独立重采 $R$ 次训练数据（不同种子），每次重新拟合并
做 episode bootstrap。对估计序列 $\hat J_r$ 报告

\[
\text{bias}=\tfrac1R\sum_r(\hat J_r-J),\quad
\text{RMSE}=\sqrt{\tfrac1R\sum_r(\hat J_r-J)^2},\quad
\text{coverage}=\tfrac1R\sum_r\mathbf1[lo_r\le J\le hi_r],
\]

以及蒙特卡洛标准误

\[
\mathrm{MCSE(bias)}=\frac{\mathrm{sd}_r(\hat J_r)}{\sqrt R},\quad
\mathrm{MCSE(RMSE)}=\frac{\mathrm{sd}_r((\hat J_r-J)^2)}{2\,\mathrm{RMSE}\sqrt R},\quad
\mathrm{MCSE(coverage)}=\sqrt{\tfrac{c(1-c)}R}.
\]

对照：(i) `DR_crossfit_actionBlind`——Q 故意错误（动作池化）；
(ii) `DR_insample_LEAKY`——同一批数据既训练又评估（训练/评估泄漏），其 bootstrap
区间偏窄、覆盖率应低于交叉拟合版本，作为“过度乐观”的反例。所有估计器核心代码
（权重、PDIS、SNIS、DR、交叉拟合、bootstrap）均为本项目自行实现。

---

## 9. 实测发现：稀有单元缺失时“静默置零”如何放大 DR 偏差

在 MDP B（tail_risk，ESS$/n\approx0.02$）上观察到一个关键的**有限样本**现象，
本项目显式报告而非掩盖：

- Oracle Q（已知核 DP）下 DR 无偏：100 次重采均值约 3.12 vs 真值 2.99，sd≈0.53。
- 用表格模型在其他折拟合时，目标偏好的稀有单元访问极少（例如训练折对
  $(t{=}0,s{=}0,a{=}1)$ 仅约十几次、对 $(t{=}1,s{=}0,a{=}1)$ 可能 0 次），对应
  $\widehat Q$ 为 NaN。代码对缺失单元**拒绝凭空填奖励**，Bellman 修正项为 NaN 并在
  加权求和时置零。
- 问题在于：这些缺失单元**恰好是被最大累积权重 $\rho_{0:t}$ 放大的单元**。把它们的
  修正项置零，等于在权重最大处系统性丢掉“向上”的修正，造成明显下偏——实测中把
  DR 偏差从约 −0.18（若用任意常数填满单元）放大到约 −1.36，覆盖率崩到 4%。

这与“DR 无偏”的理论并不矛盾：无偏性要求 $\widehat Q$ 是**固定（数据独立）**的，
NaN→0 仍是固定函数，所以**期望无偏性在无限样本下成立**；但在极端权重 + 近零有效
样本量下，有限样本偏差与方差足以主导一切。报告列 `wNaNfrac`
（`weighted_nan_fraction`，落在 NaN 修正上的折扣权重占总权重比例）与
`max_weight_on_nan` 把这一风险量化出来：当该比例不可忽略时，点估计与区间都不可信，
应当采集更多目标倾向数据或改用有结构的外推假设——而不是用 epsilon/默认值粉饰。
