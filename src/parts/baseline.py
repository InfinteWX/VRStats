import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind
from typing import Any
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class BaselineTestResult:
    variable: str  # 变量名：年龄 / 性别 / VR使用情况 / Pre_test
    test: str  # "Chi-square" 或 "Welch t-test"
    stat_name: str  # "chi2" 或 "t"
    stat: float  # 统计量
    df: float  # 自由度（卡方是 int，Welch t 是 float）
    p_value: float  # p 值
    effect_size: Optional[float]  # Cramér’s V 或 None
    n_total: Optional[int] = None  # 卡方的总样本量
    n_group_a: Optional[int] = None  # 连续变量组 A 样本量
    n_group_b: Optional[int] = None  # 连续变量组 B 样本量


def run_baseline_tests(
        df: pd.DataFrame,
        group_col: str,
        group_a: str,
        group_b: str,
        categorical_vars: list[str],
        continuous_vars: list[str],
) -> list[BaselineTestResult]:
    results: list[BaselineTestResult] = []

    # ----- 1) 多个分类变量：卡方 -----
    for var in categorical_vars:
        if var not in df.columns:
            print(f"[Warning] 分类变量 {var} 不存在，跳过")
            continue

        ct = pd.crosstab(df[group_col], df[var])
        chi2, p, dof, expected = chi2_contingency(ct)

        # Cramér's V
        n = ct.values.sum()
        r, c = ct.shape
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

    # ----- 2) 多个连续变量：Welch t-test -----
    for var in continuous_vars:
        if var not in df.columns:
            print(f"[Warning] 连续变量 {var} 不存在，跳过")
            continue

        a = df.loc[df[group_col] == group_a, var].dropna().astype(float)
        b = df.loc[df[group_col] == group_b, var].dropna().astype(float)

        n1, n2 = a.size, b.size
        if n1 < 2 or n2 < 2:
            print(f"[Warning] 连续变量 {var} 样本量不足，跳过 Welch t-test")
            continue

        t_res = ttest_ind(a, b, equal_var=False)
        t_stat = float(t_res.statistic)
        p_val = float(t_res.pvalue)

        s1_sq, s2_sq = a.var(ddof=1), b.var(ddof=1)
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
        output_excel_path: str | Path,
):
    # 1. 读取数据
    df = pd.read_excel(input_excel_path, sheet_name=input_sheet_name)

    # 2. 运行基线特征检验
    # ====== 新增：基线特征（年龄 / 性别 / 前测 Welch t） ======
    # ===== 新增：基线特征检验 =====
    baseline_results = run_baseline_tests(
        df=df,
        group_col=group_col,
        group_a=group_a,
        group_b=group_b,
        categorical_vars=categorical_vars,
        continuous_vars=continuous_vars,
    )

    baseline_df = pd.DataFrame([asdict(r) for r in baseline_results])

    # 7. 打印汇总结果（你也可以保存为 Excel）
    pd.set_option("display.width", 120)
    pd.set_option("display.max_columns", None)

    print("\n===== 基线特征检验（年龄 / 性别 / 前测 Welch t） =====")
    print(baseline_df)

    # 如需保存结果：
    if output_excel_path.exists():
        # 已存在文件 → “覆盖 sheet 或添加 sheet”
        with pd.ExcelWriter(
                output_excel_path,
                mode="a",
                if_sheet_exists="replace"  # 存在表就覆盖，不存在就自动新增
        ) as writer:
            baseline_df.to_excel(writer, sheet_name="baseline", index=False)
    else:
        # 文件不存在 → 创建新文件
        with pd.ExcelWriter(
                output_excel_path,
                mode="w"
        ) as writer:
            baseline_df.to_excel(writer, sheet_name="baseline", index=False)


def main(args: Any):
    process(
        input_excel_path=args.input_excel_path,
        input_sheet_name=args.input_sheet_name,
        group_col=args.group_col,
        group_a=args.group_label_a,
        group_b=args.group_label_b,
        categorical_vars=args.baseline_categorical_vars,
        continuous_vars=args.baseline_continuous_vars,
        output_excel_path=args.output_excel_path,
    )