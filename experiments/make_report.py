"""Render results/*.json into a Markdown results table with MC errors."""

from __future__ import annotations

import json
import os

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")

ORDER = [
    ("good_coverage.json", "场景1 良好覆盖"),
    ("good_coverage_myopicQ.json", "场景1 + 故意错误Q(myopic)"),
    ("good_coverage_zeroQ.json", "场景1 + Q=0(DR退化为PDIS)"),
    ("tiny_behavior.json", "场景2 极小行为概率"),
    ("tiny_behavior_clip20.json", "场景2 + 权重截断20"),
    ("leakage.json", "泄漏反例: 交叉拟合 vs 同批训练评估"),
    ("no_coverage.json", "场景3 无覆盖(不可识别)"),
]

NAME = {"pdis": "PDIS", "snis": "SNIS", "dr": "序列DR", "plugin_v0": "模型插件 V0"}


def fnum(x, nd=4):
    if x is None:
        return "—"
    if isinstance(x, float) and not (x == x):
        return "NaN"
    return f"{x:.{nd}f}"


def main():
    lines = ["# 实验结果（由 results/*.json 生成）", ""]
    for fname, title in ORDER:
        path = os.path.join(RES, fname)
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        lines.append(f"## {title}")
        lines.append("")
        cfg = d.get("config", {})
        lines.append(
            f"n={cfg.get('n_episodes')} episodes, "
            f"重复={cfg.get('n_replicates')}, bootstrap={cfg.get('n_boot', '—')}, "
            f"K={cfg.get('k_folds', '—')}, 截断clip={cfg.get('clip')}, "
            f"Q模型={cfg.get('q_mode', '—')}"
        )
        lines.append("")
        if "methods" in d:
            lines.append(f"真值 J(pi) = **{fnum(d['truth'])}**; "
                         f"中位最小ESS/n = {fnum(d.get('min_ess_frac_median'), 4)}")
            if d.get("ess_frac_per_t_median"):
                curve = ", ".join(f"{x:.4f}" for x in d["ess_frac_per_t_median"])
                lines.append(f"每步 ESS/n 中位数（t=0..H-1）: [{curve}]")
            lines.append("")
            lines.append("| 估计器 | 均值 | 偏差 (MC SE) | RMSE (MC SE) | 95%覆盖 (SE) | 区间中位宽 | 有限值比例 |")
            lines.append("|---|---|---|---|---|---|---|")
            for key, st in d["methods"].items():
                lines.append(
                    f"| {NAME.get(key,key)} | {fnum(st.get('mean'))} "
                    f"| {fnum(st.get('bias'))} ({fnum(st.get('bias_mc_se'))}) "
                    f"| {fnum(st.get('rmse'))} ({fnum(st.get('rmse_mc_se'))}) "
                    f"| {fnum(st.get('coverage'))} ({fnum(st.get('coverage_mc_se'))}) "
                    f"| {fnum(st.get('median_width'))} | {fnum(st.get('finite_frac'),3)} |"
                )
            lines.append("")
            if d.get("scenario") == "tiny_behavior":
                lines.append(
                    "> 注：SNIS 仅当某重复至少采到一条全程动作1的轨迹时才有限（有限比例 0.075）；"
                    "这些重复里全合规轨迹回报几乎相同，故条件区间宽≈0；总体覆盖仍为 0，"
                    "因为 92.5% 的重复无法归一化。宽度0不是精确性，而是退化。")
                lines.append("")
        elif "pipelines" in d:
            lines.append(f"真值 J(pi) = **{fnum(d['truth'])}**")
            lines.append("")
            lines.append("| 流水线 | 偏差 | RMSE | 估计SD | 95%覆盖 (SE) | 区间中位宽 |")
            lines.append("|---|---|---|---|---|---|")
            for key, st in d["pipelines"].items():
                label = {"honest_crossfit": "诚实: episode交叉拟合",
                         "leaky_insample": "泄漏: 全量训练+同批评估"}[key]
                lines.append(
                    f"| {label} | {fnum(st['bias'])} | {fnum(st['rmse'])} "
                    f"| {fnum(st['sd_estimates'])} | {fnum(st['coverage'])} "
                    f"({fnum(st['coverage_mc_se'])}) | {fnum(st['median_ci_width'])} |"
                )
            lines.append("")
        else:
            lines.append(f"真值(仅模型可知, 数据不可识别) J(pi) = **{fnum(d['truth'])}**")
            lines.append("")
            lines.append(f"- 缺失支持的 (t,s,a): {d.get('missing_target_support_pairs')}")
            lines.append(f"- {d.get('config',{}).get('n_replicates', d.get('n_replicates'))} 次重新采集中，识别失败被检出比例 = "
                         f"**{fnum(d.get('identification_fail_detected_rate'), 3)}**")
            lines.append("- 所有 PDIS/SNIS/DR 返回 NaN，不构造区间，不做 epsilon 修补。")
            lines.append("")
    out = os.path.join(RES, "RESULTS.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
