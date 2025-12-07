from pathlib import Path
import polars as pl
from dataclasses import dataclass, asdict
from typing import Any
from ..utils.func import insert_blank_rows_by_block


@dataclass
class DescriptiveStats:
    """描述性统计结果数据类"""
    block: str  # 所属数据模块（例如 IEG_Total / Knowledge / EEG）
    variable: str  # 变量名称（如 'cognitive_load'）
    group: str  # 分组名称（如 '触觉组' 或 '手势组'）

    n: int  # 样本量
    mean: float  # 均值
    std: float  # 样本标准差

    median: float  # 中位数（对偏态分布更稳健）
    q1: float  # 第 1 四分位数（25% 分位）
    q3: float  # 第 3 四分位数（75% 分位）

    minimum: float  # 最小值
    maximum: float  # 最大值


def compute_descriptives(
        df: pl.DataFrame,
        group_col: str,
        variables: list[str],
        block_name: str
) -> list[DescriptiveStats]:
    """
    按组计算每个变量的描述性统计。
    
    参数:
        df: Polars DataFrame，包含所有变量
        group_col: 分组变量的列名
        variables: 需要计算描述性统计的变量列表
        block_name: 所属数据模块名称
    
    返回:
        DescriptiveStats 对象列表
    """
    results: list[DescriptiveStats] = []

    # 只保留相关列
    sub = df.select([group_col] + variables)

    for var in variables:
        for g_df in sub.partition_by(group_col):
            g = g_df[group_col][0]
            x = g_df.select(pl.col(var).drop_nulls())[var]
            if x.is_empty():
                continue
            desc = DescriptiveStats(
                block=block_name,
                variable=var,
                group=str(g),
                n=int(x.len()),
                mean=float(x.mean()),
                std=float(x.std()),
                median=float(x.median()),
                q1=float(x.quantile(0.25)),
                q3=float(x.quantile(0.75)),
                minimum=float(x.min()),
                maximum=float(x.max()),
            )
            results.append(desc)

    return results


def process(
        input_excel_path: str | Path,
        input_sheet_name: Any,
        group_col: str,
        variable_blocks: dict[str, list[str]],
        add_blank_rows: bool = True,
) -> tuple[pl.DataFrame, str]:
    """
    描述性统计的主处理流程。
    
    从 Excel 读取数据，对每个数据模块计算描述性统计，并返回结果 DataFrame。
    
    参数:
        input_excel_path: 输入的 Excel 文件路径
        input_sheet_name: 工作表名称或索引
        group_col: 分组变量的列名
        variable_blocks: 数据模块字典，键为模块名，值为变量列表
        add_blank_rows: 是否在不同模块间添加空行
    
    返回:
        tuple: (结果 DataFrame, sheet名称)
    """
    # 1. 读取数据
    df = pl.read_excel(input_excel_path, sheet_name=input_sheet_name)

    all_desc: list[DescriptiveStats] = []

    # 2. 对每个模块计算描述性统计
    for block_name, vars_in_block in variable_blocks.items():
        # 计算此模块的描述性统计
        desc_block = compute_descriptives(df, group_col, vars_in_block, block_name)
        all_desc.extend(desc_block)

    # 3. 转换为 DataFrame
    desc_df = pl.DataFrame([asdict(d) for d in all_desc])

    # 4. 添加空行分隔（如果需要）
    if add_blank_rows:
        desc_df = insert_blank_rows_by_block(desc_df, block_col="block")

    # 5. 打印结果
    print("\n===== 描述性统计（Descriptives） =====")
    print(desc_df.sort(["block", "variable", "group"]))

    return desc_df, "descriptives"


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
        variable_blocks=args.variable_blocks,
        add_blank_rows=args.add_blank_rows,
    )

