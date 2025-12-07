import polars as pl
from dataclasses import dataclass, asdict
from typing import Any
from pathlib import Path
from ..utils.func import (
    independent_ttest,
    mannwhitney_u,
    holm_bonferroni,
    benjamini_hochberg
)
from ..utils.func import insert_blank_rows_by_block


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


def process(
        input_excel_path: str | Path,
        input_sheet_name: Any,
        group_col: str,
        group_label_a: str,
        group_label_b: str,
        test_func_name: str,
        variable_blocks: dict[str, list[str]],
        add_blank_rows: bool = True,
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
    
    返回:
        tuple: (结果 DataFrame, sheet名称)
    """
    # 1. 读取数据
    df = pl.read_excel(input_excel_path, sheet_name=input_sheet_name)

    all_tests: list[TestResult] = []

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
    )
