import polars as pl
from dataclasses import dataclass, asdict
from typing import Any
from pathlib import Path
from utils.func import (
    independent_ttest,
    mannwhitney_u,
    holm_bonferroni,
    benjamini_hochberg
)
from utils.func import insert_blank_rows_by_block
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TestResult:
    """两组比较检验结果数据类"""
    block: str  # 所属数据模块（例如 IEG_Total/EEG/Knowledge）
    variable: str  # 被检验的变量名称（如 cognitive_load）
    test_name: str  # 使用的检验方法名称（如 'mannwhitney' 或 'ttest'）
    group_a: str  # 第一组名称（如 '触觉组'）
    group_b: str  # 第二组名称（如 '手势组'）
    n_a: int  # 组 A 的样本量
    n_b: int  # 组 B 的样本量

    statistic: float  # 检验统计量：t 检验为 t 值，Mann-Whitney U 检验为 U 值
    z_value: float | None  # 标准化统计量 Z（仅 Mann-Whitney U 检验计算）
    p_value: float  # 原始（未校正）p 值
    effect_size: float | None  # 效应量：t 检验为 Cohen's d，Mann-Whitney U 检验为 Z-based r

    p_holm: float | None  # Holm-Bonferroni 校正后的 p 值（控制 FWER）
    rejects_holm: bool | None  # 在 Holm-Bonferroni 下是否拒绝原假设

    p_bh: float | None  # Benjamini-Hochberg FDR 校正后的 p 值（控制 FDR）
    rejects_bh: bool | None  # 在 BH-FDR 下是否拒绝原假设


def run_two_group_test(
        df: pl.DataFrame,
        group_col: str,
        variable: str,
        block_name: str,
        group_a: str,
        group_b: str,
        test_func_name: str,
) -> TestResult | None:
    """
    对单个变量执行两组比较检验。
    
    参数:
        df: Polars DataFrame，包含所有变量
        group_col: 分组变量的列名
        variable: 需要检验的变量名
        block_name: 所属数据模块名称
        group_a: 第一组的标签
        group_b: 第二组的标签
        test_func_name: 检验方法名称（'ttest' 或 'mannwhitney'）
    
    返回:
        TestResult 对象，如果数据不足则返回 None
    """

    # 提取两组数据
    a = df.filter(pl.col(group_col) == group_a).select(pl.col(variable).drop_nulls())[variable].to_numpy()
    b = df.filter(pl.col(group_col) == group_b).select(pl.col(variable).drop_nulls())[variable].to_numpy()

    if len(a) == 0 or len(b) == 0:
        # 有一组没有数据，跳过
        return None

    # 选择检验函数
    if test_func_name == "ttest":
        test_func = independent_ttest
    elif test_func_name == "mannwhitney":
        test_func = mannwhitney_u
    else:
        raise ValueError(f"未知的检验方法：{test_func_name}")

    # 执行检验
    stat, p_val, eff, z_val = test_func(a, b)

    return TestResult(
        block=block_name,
        variable=variable,
        test_name=test_func_name,
        group_a=str(group_a),
        group_b=str(group_b),
        n_a=len(a),
        n_b=len(b),
        statistic=stat,
        p_value=p_val,
        z_value=z_val,
        effect_size=eff,
        p_holm=None,
        rejects_holm=None,
        p_bh=None,
        rejects_bh=None,
    )


def _significance_marker(p: float) -> str:
    """根据 p 值返回显著性标记符号。"""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


def visualize_test_results(
        test_results: list[TestResult],
        block_name: str,
        save_path: Path,
) -> None:
    """
    为两组检验结果生成森林图（效应量 + 95% CI）。
    
    参数:
        test_results: TestResult 对象列表
        block_name: 所属数据模块名称
        save_path: 保存图片的路径
    """
    if not test_results:
        return
    
    # 提取数据
    variables = [t.variable for t in test_results]
    effect_sizes = [t.effect_size if t.effect_size is not None else 0 for t in test_results]
    p_values = [t.p_bh if t.p_bh is not None else t.p_value for t in test_results]
    
    # 计算 95% 置信区间（简化估计）
    # 对于 Cohen's d: CI ≈ d ± 1.96 * SE(d)
    # SE(d) ≈ sqrt((n1+n2)/(n1*n2) + d^2/(2*(n1+n2)))
    ci_lower = []
    ci_upper = []
    for t in test_results:
        if t.effect_size is None:
            ci_lower.append(0)
            ci_upper.append(0)
        else:
            d = t.effect_size
            n1, n2 = t.n_a, t.n_b
            se_d = np.sqrt((n1 + n2) / (n1 * n2) + d**2 / (2 * (n1 + n2)))
            ci_lower.append(d - 1.96 * se_d)
            ci_upper.append(d + 1.96 * se_d)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(3.5, 2.8 + 0.3 * len(variables)))
    
    # 绘制森林图
    y_pos = np.arange(len(variables))
    
    # 根据显著性设置颜色
    colors = ['#d62728' if p < 0.05 else '#1f77b4' for p in p_values]
    
    # 绘制效应量点
    ax.scatter(effect_sizes, y_pos, s=60, c=colors, edgecolors='black', linewidth=0.8, zorder=3)
    
    # 绘制置信区间
    for i in range(len(variables)):
        ax.plot([ci_lower[i], ci_upper[i]], [y_pos[i], y_pos[i]], 
               color=colors[i], linewidth=1.5, zorder=2)
    
    # 添加零线
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # 设置标题和标签
    ax.set_yticks(y_pos)
    ax.set_yticklabels(variables)
    ax.set_xlabel('效应量 (Effect Size)', fontweight='normal')
    ax.set_title(f'{block_name} - 效应量森林图', fontweight='bold', pad=10)
    
    # 优化网格线
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 调整spine样式
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('black')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', edgecolor='black', label='p < 0.05'),
        Patch(facecolor='#1f77b4', edgecolor='black', label='p ≥ 0.05')
    ]
    ax.legend(handles=legend_elements, loc='best', frameon=True, edgecolor='black')
    
    # 保存图片
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
    print(f"[Info] 两组检验可视化已保存: {save_path}")
    
    plt.close(fig)


def visualize_boxplots(
        df: pl.DataFrame,
        group_col: str,
        test_results: list[TestResult],
        block_name: str,
        save_dir: Path,
) -> None:
    """
    为显著性检验的每个变量生成分组箱线图（小提琴图 + 箱线图 + 数据点 + 显著性标注）。

    图表采用学术论文标准风格，包含：
    - 半透明小提琴图显示数据分布
    - 箱线图显示四分位数
    - 散点图显示原始数据点
    - 括号式显著性标注（*、**、***、n.s.）

    参数:
        df: 包含原始数据的 Polars DataFrame
        group_col: 分组变量的列名
        test_results: 该模块的 TestResult 对象列表
        block_name: 所属数据模块名称
        save_dir: 保存图片的目录
    """
    if not test_results:
        return

    # 转为 pandas 便于绘图
    plot_source = df.to_pandas()

    # 学术配色方案
    palette = ['#4C72B0', '#DD8452']  # 蓝色/橙色

    for tr in test_results:
        var = tr.variable
        if var not in plot_source.columns:
            continue

        # 使用 BH 校正后的 p 值（如果有），否则使用原始 p 值
        p_val = tr.p_bh if tr.p_bh is not None else tr.p_value

        groups = [tr.group_a, tr.group_b]
        group_data = []
        for g in groups:
            vals = plot_source.loc[plot_source[group_col] == g, var].dropna().values
            group_data.append(vals)

        if any(len(d) == 0 for d in group_data):
            continue

        # ---- 创建图形 ----
        fig, ax = plt.subplots(figsize=(3.5, 3.2))

        positions = np.array([0, 1])

        # 1) 小提琴图
        vp = ax.violinplot(
            group_data,
            positions=positions,
            widths=0.65,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for i, body in enumerate(vp['bodies']):
            body.set_facecolor(palette[i])
            body.set_alpha(0.35)
            body.set_edgecolor('black')
            body.set_linewidth(0.8)

        # 2) 箱线图
        for i, data in enumerate(group_data):
            ax.boxplot(
                [data],
                positions=[positions[i]],
                widths=0.18,
                patch_artist=True,
                showfliers=False,
                boxprops=dict(facecolor='white', edgecolor='black', linewidth=1.0),
                whiskerprops=dict(color='black', linewidth=1.0),
                capprops=dict(color='black', linewidth=1.0),
                medianprops=dict(color='#e74c3c', linewidth=1.5),
            )

        # 3) 数据散点（抖动）
        for i, data in enumerate(group_data):
            jitter = np.random.default_rng(42).normal(0, 0.04, size=len(data))
            ax.scatter(
                positions[i] + jitter,
                data,
                s=18,
                alpha=0.45,
                color=palette[i],
                edgecolors='black',
                linewidths=0.3,
                zorder=3,
            )

        # 4) 显著性标注横线 + 标记
        sig_text = _significance_marker(p_val)
        y_max = max(d.max() for d in group_data)
        y_range = y_max - min(d.min() for d in group_data)
        bar_y = y_max + y_range * 0.08
        bar_tip = y_range * 0.02

        ax.plot(
            [positions[0], positions[0], positions[1], positions[1]],
            [bar_y - bar_tip, bar_y, bar_y, bar_y - bar_tip],
            color='black',
            linewidth=1.0,
        )
        ax.text(
            (positions[0] + positions[1]) / 2,
            bar_y + y_range * 0.01,
            sig_text,
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
        )

        # ---- 轴、标题 ----
        ax.set_xticks(positions)
        ax.set_xticklabels(groups)
        ax.set_xlabel(group_col, fontweight='normal')
        ax.set_ylabel(var, fontweight='normal')
        ax.set_title(f'{block_name} / {var}', fontweight='bold', pad=10)

        # 留出显著性标注空间
        ax.set_ylim(top=bar_y + y_range * 0.18)

        # 网格与边框
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color('black')

        # 保存
        safe_name = var.replace('/', '_').replace('\\', '_')
        save_path = save_dir / f"{safe_name}_boxplot.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.05)
        print(f"[Info] 箱线图已保存: {save_path}")
        plt.close(fig)


def process(
        input_excel_path: str | Path,
        input_sheet_name: Any,
        group_col: str,
        group_label_a: str,
        group_label_b: str,
        test_func_name: str,
        variable_blocks: dict[str, list[str]],
        add_blank_rows: bool = True,
        visualization_dir: Path | None = None,
) -> tuple[pl.DataFrame, str]:
    """
    两组比较检验的主处理流程。
    
    从 Excel 读取数据，对每个数据模块执行两组比较检验，
    并对每个模块分别进行 Holm-Bonferroni 和 Benjamini-Hochberg 校正。
    
    参数:
        input_excel_path: 输入的 Excel 文件路径
        input_sheet_name: 工作表名称或索引
        group_col: 分组变量的列名
        group_label_a: 第一组的标签
        group_label_b: 第二组的标签
        test_func_name: 检验方法名称（'ttest' 或 'mannwhitney'）
        variable_blocks: 数据模块字典，键为模块名，值为变量列表
        add_blank_rows: 是否在不同模块间添加空行
        visualization_dir: 可视化图表保存目录（可选）
    
    返回:
        tuple: (结果 DataFrame, sheet名称)
    """
    # 1. 读取数据
    df = pl.read_excel(input_excel_path, sheet_name=input_sheet_name)

    all_tests: list[TestResult] = []
    
    # 创建可视化目录（如果指定）- 放在 tests 子目录下
    tests_vis_dir = None
    if visualization_dir is not None:
        tests_vis_dir = visualization_dir / "tests"
        tests_vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Info] 两组检验可视化图表将保存到: {tests_vis_dir}")

    # 2. 对每个模块执行两组比较检验
    for block_name, vars_in_block in variable_blocks.items():
        # 对此模块的每个变量执行检验
        tests_block: list[TestResult] = []
        for var in vars_in_block:
            tr = run_two_group_test(
                df,
                group_col=group_col,
                variable=var,
                group_a=group_label_a,
                group_b=group_label_b,
                block_name=block_name,
                test_func_name=test_func_name,
            )
            if tr is not None:
                tests_block.append(tr)

        # 对此模块执行 Holm-Bonferroni 校正
        if tests_block:
            p_vals = [t.p_value for t in tests_block]
            rejects_holm, pvals_holm_adj = holm_bonferroni(p_vals)
            for t, rejects_holm, p_h in zip(tests_block, rejects_holm, pvals_holm_adj):
                t.p_holm = p_h
                t.rejects_holm = bool(rejects_holm)
        else:
            # 可以在所有 block 结束后一次性校正
            pass

        # 对此模块执行 Benjamini-Hochberg FDR 校正
        p_vals = [t.p_value for t in tests_block]
        rejects_bh, pvals_bh_adj = benjamini_hochberg(p_vals)

        for t, rejects_bh, pvals_bh_adj in zip(tests_block, rejects_bh, pvals_bh_adj):
            t.p_bh = pvals_bh_adj  # BH 校正的 p
            t.rejects_bh = bool(rejects_bh)  # BH 是否拒绝原假设

        all_tests.extend(tests_block)
        
        # 生成可视化（如果指定目录）
        if tests_vis_dir is not None and tests_block:
            try:
                # 森林图
                save_path = tests_vis_dir / f"{block_name}_forest.png"
                visualize_test_results(tests_block, block_name, save_path)
                # 分组箱线图
                block_box_dir = tests_vis_dir / block_name
                block_box_dir.mkdir(parents=True, exist_ok=True)
                visualize_boxplots(df, group_col, tests_block, block_name, block_box_dir)
            except Exception as e:
                print(f"[Warning] 生成 {block_name} 可视化时出错: {e}")

    # 3. 转换为 DataFrame
    test_df = pl.DataFrame([asdict(t) for t in all_tests])

    # 4. 添加空行分隔（如果需要）
    if add_blank_rows:
        test_df = insert_blank_rows_by_block(test_df, block_col="block")

    # 5. 打印结果
    print("\n===== 两组比较检验（Tests） =====")
    print(test_df.sort(["block", "variable"]))

    return test_df, "test"


def process_with_args(args: Any) -> tuple[pl.DataFrame, str]:
    """
    从命令行参数对象调用 process 函数。
    
    参数:
        args: 包含所有必要参数的对象
    
    返回:
        tuple: (结果 DataFrame, sheet名称)
    """
    return process(
        input_excel_path=args.input_excel_path,
        input_sheet_name=args.input_sheet_name,
        group_col=args.group_col,
        group_label_a=args.group_label_a,
        group_label_b=args.group_label_b,
        test_func_name=args.test_func_name,
        variable_blocks=args.variable_blocks,
        add_blank_rows=args.add_blank_rows,
        visualization_dir=args.visualization_dir,
    )
