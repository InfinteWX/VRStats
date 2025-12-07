import numpy as np
import polars as pl
from scipy.stats import chi2_contingency, ttest_ind
from typing import Any
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class BaselineTestResult:
    """基线特征检验结果数据类"""
    variable: str  # 变量名（如：年龄、性别、VR使用经验、前测成绩）
    test: str  # 检验方法名称："Chi-square" 或 "Welch t-test"
    stat_name: str  # 统计量名称："chi2" 或 "t"
    stat: float  # 统计量值
    df: float  # 自由度（卡方检验为整数，Welch t 检验为浮点数）
    p_value: float  # 原始 p 值
    effect_size: Optional[float]  # 效应量：卡方检验为 Cramér's V，t 检验为 None
    n_total: Optional[int] = None  # 卡方检验的总样本量
    n_group_a: Optional[int] = None  # t 检验中组 A 的样本量
    n_group_b: Optional[int] = None  # t 检验中组 B 的样本量


def run_baseline_tests(
        df: pl.DataFrame,
        group_col: str,
        group_a: str,
        group_b: str,
        categorical_vars: list[str],
        continuous_vars: list[str],
) -> list[BaselineTestResult]:
    """
    对分类变量和连续变量执行基线特征检验。
    
    对分类变量使用卡方检验，对连续变量使用 Welch t 检验。
    
    参数:
        df: Polars DataFrame，包含所有变量
        group_col: 分组变量的列名
        group_a: 第一组的标签
        group_b: 第二组的标签
        categorical_vars: 分类变量列表
        continuous_vars: 连续变量列表
    
    返回:
        BaselineTestResult 对象列表
    """
    results: list[BaselineTestResult] = []

    # ----- 1) 分类变量：卡方检验 -----
    for var in categorical_vars:
        if var not in df.columns:
            print(f"[Warning] 分类变量 {var} 不存在，跳过")
            continue

        # 使用 Polars 计算交叉表
        ct_df = df.group_by([group_col, var]).agg(pl.len().alias("count"))
        ct_pivot = ct_df.pivot(values="count", index=group_col, columns=var).fill_null(0)
        ct_values = ct_pivot.select(pl.all().exclude(group_col)).to_numpy()

        chi2, p, dof, expected = chi2_contingency(ct_values)

        # Cramér's V
        n = ct_values.sum()
        r, c = ct_values.shape
        if n > 0 and min(r - 1, c - 1) > 0:
            cramers_v = np.sqrt(chi2 / (n * min(r - 1, c - 1)))
        else:
            cramers_v = None

        results.append(
            BaselineTestResult(
                variable=var,
                test="Chi-square",
                stat_name="chi2",
                stat=float(chi2),
                df=float(dof),
                p_value=float(p),
                effect_size=cramers_v,
                n_total=int(n),
            )
        )

    # ----- 2) 连续变量：Welch t 检验 -----
    for var in continuous_vars:
        if var not in df.columns:
            print(f"[Warning] 连续变量 {var} 不存在，跳过")
            continue

        a = df.filter(pl.col(group_col) == group_a).select(
            pl.col(var).drop_nulls().cast(pl.Float64)).to_numpy().flatten()
        b = df.filter(pl.col(group_col) == group_b).select(
            pl.col(var).drop_nulls().cast(pl.Float64)).to_numpy().flatten()

        n1, n2 = len(a), len(b)
        if n1 < 2 or n2 < 2:
            print(f"[Warning] 连续变量 {var} 样本量不足，跳过 Welch t-test")
            continue

        t_res = ttest_ind(a, b, equal_var=False)
        t_stat = float(t_res.statistic)
        p_val = float(t_res.pvalue)

        s1_sq, s2_sq = np.var(a, ddof=1), np.var(b, ddof=1)
        v1, v2 = s1_sq / n1, s2_sq / n2
        df_welch = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))

        results.append(
            BaselineTestResult(
                variable=var,
                test="Welch t-test",
                stat_name="t",
                stat=t_stat,
                df=df_welch,
                p_value=p_val,
                effect_size=None,
                n_group_a=n1,
                n_group_b=n2,
            )
        )

    return results


def process(
        input_excel_path: str | Path,
        input_sheet_name: Any,
        group_col: str,
        group_a: str,
        group_b: str,
        categorical_vars: list[str],
        continuous_vars: list[str],
) -> tuple[pl.DataFrame, str]:
    """
    基线特征检验的主处理流程。
    
    从 Excel 读取数据，执行基线特征检验，并返回结果 DataFrame。
    
    参数:
        input_excel_path: 输入的 Excel 文件路径
        input_sheet_name: 工作表名称或索引
        group_col: 分组变量的列名
        group_a: 第一组的标签
        group_b: 第二组的标签
        categorical_vars: 分类变量列表
        continuous_vars: 连续变量列表
    
    返回:
        tuple: (结果 DataFrame, sheet名称)
    """
    # 1. 读取数据
    df = pl.read_excel(input_excel_path, sheet_name=input_sheet_name)

    # 2. 运行基线特征检验
    baseline_results = run_baseline_tests(
        df=df,
        group_col=group_col,
        group_a=group_a,
        group_b=group_b,
        categorical_vars=categorical_vars,
        continuous_vars=continuous_vars,
    )

    # 3. 转换为 DataFrame并输出
    baseline_df = pl.DataFrame([asdict(r) for r in baseline_results])
    print("\n===== 基线特征检验（年龄 / 性别 / 前测 Welch t） =====")
    print(baseline_df)

    return baseline_df, "baseline"


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
        group_a=args.group_label_a,
        group_b=args.group_label_b,
        categorical_vars=args.baseline_categorical_vars,
        continuous_vars=args.baseline_continuous_vars,
    )
