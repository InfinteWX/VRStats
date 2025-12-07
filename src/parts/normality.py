from pathlib import Path
from scipy import stats

import numpy as np
import polars as pl
from dataclasses import dataclass, asdict
from typing import Any
from ..utils.func import insert_blank_rows_by_block


@dataclass
class NormalityResult:
    """正态性检验结果数据类"""
    block: str  # 所属数据模块（如 IEG_Total / EEG / Knowledge）
    variable: str  # 变量名称（如 'immersion'）
    group: str  # 分组名称（如 '触觉组' 或 '手势组'）

    stat: float  # Shapiro-Wilk 检验统计量 W（越接近 1 表示越接近正态分布）
    p_value: float  # 正态性检验的 p 值（p < 0.05 表示拒绝正态性假设）
    is_normal: bool  # 是否判定为正态（True: p >= 0.05，False: p < 0.05）


def check_normality(
        df: pl.DataFrame,
        group_col: str,
        variables: list[str],
        block_name: str,
        alpha: float = 0.05,
) -> list[NormalityResult]:
    """
    对每个变量的每个组执行 Shapiro-Wilk 正态性检验。
    
    参数:
        df: Polars DataFrame，包含所有变量
        group_col: 分组变量的列名
        variables: 需要检验的变量列表
        block_name: 所属数据模块名称
        alpha: 显著性水平，默认 0.05
    
    返回:
        NormalityResult 对象列表
    """
    results: list[NormalityResult] = []

    for var in variables:
        for g_df in df.partition_by(group_col):
            g = g_df[group_col][0]
            x = g_df.select(pl.col(var).drop_nulls())[var].to_numpy()
            if len(x) < 3:
                # 样本量太少，无法可靠检验，默认视为正态
                stat, p_val = np.nan, np.nan
                is_norm = True
            else:
                stat, p_val = stats.shapiro(x)
                is_norm = bool(p_val > alpha)

            results.append(
                NormalityResult(
                    block=block_name,
                    variable=var,
                    group=str(g),
                    stat=float(stat) if stat is not np.nan else np.nan,
                    p_value=float(p_val) if p_val is not np.nan else np.nan,
                    is_normal=is_norm,
                )
            )

    return results


def process(
        input_excel_path: str | Path,
        input_sheet_name: Any,
        normality_alpha: float,
        group_col: str,
        variable_blocks: dict[str, list[str]],
        add_blank_rows: bool = True,
) -> tuple[pl.DataFrame, str]:
    """
    正态性检验的主处理流程。
    
    从 Excel 读取数据，对每个数据模块执行正态性检验，并返回结果 DataFrame。
    
    参数:
        input_excel_path: 输入的 Excel 文件路径
        input_sheet_name: 工作表名称或索引
        normality_alpha: 正态性检验的显著性水平
        group_col: 分组变量的列名
        variable_blocks: 数据模块字典，键为模块名，值为变量列表
        add_blank_rows: 是否在不同模块间添加空行
    
    返回:
        tuple: (结果 DataFrame, sheet名称)
    """
    # 1. 读取数据
    df = pl.read_excel(input_excel_path, sheet_name=input_sheet_name)

    all_norm: list[NormalityResult] = []

    # 2. 对每个模块执行正态性检验
    for block_name, vars_in_block in variable_blocks.items():
        # 计算此模块的正态性检验
        norm_block = check_normality(df, group_col, vars_in_block, block_name, alpha=normality_alpha)
        all_norm.extend(norm_block)

    # 3. 转换为 DataFrame
    norm_df = pl.DataFrame([asdict(d) for d in all_norm])

    # 4. 添加空行分隔（如果需要）
    if add_blank_rows:
        norm_df = insert_blank_rows_by_block(norm_df, block_col="block")

    # 5. 打印结果
    print("\n===== 正态性检验（Shapiro-Wilk） =====")
    print(norm_df.sort(["block", "variable", "group"]))

    return norm_df, "normality"


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
        normality_alpha=args.normality_alpha,
        group_col=args.group_col,
        variable_blocks=args.variable_blocks,
        add_blank_rows=args.add_blank_rows,
    )
