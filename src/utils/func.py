import numpy as np
import polars as pl
from scipy import stats
from statsmodels.stats.multitest import multipletests


def insert_blank_rows_by_block(df: pl.DataFrame, block_col: str = "block") -> pl.DataFrame:
    """
    在每个 block 之间插入空行，用于在 Excel 输出中分隔不同的数据块。
    
    参数:
        df: Polars DataFrame，包含需要分隔的数据
        block_col: 用于分组的列名，默认为 "block"
    
    返回:
        添加了空行分隔的 Polars DataFrame
    """

    # 1. 根据 DataFrame 的列类型构建空行数据
    row_data = {}
    for col in df.columns:
        dtype = df[col].dtype
        if dtype == pl.String or dtype == pl.Utf8:
            row_data[col] = ""
        elif dtype in [pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
            row_data[col] = None
        elif dtype in [pl.Float64, pl.Float32]:
            row_data[col] = None
        elif dtype == pl.Boolean:
            row_data[col] = None
        else:
            row_data[col] = None

    blank_row = pl.DataFrame([row_data])

    # 2. 为每个 block 后面插入空行
    new_dfs = []
    for sub in df.partition_by(block_col, maintain_order=True):
        new_dfs.append(sub)
        new_dfs.append(blank_row)

    # 3. 合并所有 DataFrame，使用 diagonal 模式处理不同类型的列
    return pl.concat(new_dfs, how="diagonal")


# ---------- 具体检验方法封装（方便扩展/替换） ----------

def independent_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    """
    执行 Welch t 检验（不假定方差齐性的独立样本 t 检验）。
    
    参数:
        a: 第一组数据的 numpy 数组
        b: 第二组数据的 numpy 数组
    
    返回:
        tuple: (t统计量, p值, Cohen's d效应量, Z值(对于t检验为NaN))
    """
    t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)

    # 计算 Cohen's d 效应量（使用合并标准差）
    na, nb = len(a), len(b)
    sa, sb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_sd = np.sqrt(((na - 1) * sa + (nb - 1) * sb) / (na + nb - 2)) if na + nb - 2 > 0 else np.nan
    d = (np.mean(a) - np.mean(b)) / pooled_sd if pooled_sd > 0 else np.nan

    z_val = np.nan  # t 检验不计算 Z 值

    return float(t_stat), float(p_val), float(d), z_val


def mannwhitney_u(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    """
    执行 Mann-Whitney U 检验（非参数检验，用于两独立样本）。
    使用连续性校正和 ties 修正计算 Z 值。
    
    参数:
        a: 第一组数据的 numpy 数组
        b: 第二组数据的 numpy 数组
    
    返回:
        tuple: (U统计量, p值, 基于Z的秩双列相关系数r, Z值)
    """
    # 执行 Mann-Whitney U 检验
    res = stats.mannwhitneyu(a, b, alternative="two-sided")
    U = float(res.statistic)
    p_val = float(res.pvalue)

    n1, n2 = len(a), len(b)

    # ========== Z 值计算（包含连续性校正和 ties 修正）==========

    # 计算秩次（用于 ties 修正）
    combined = np.concatenate([a, b])
    N = n1 + n2
    ranks = stats.rankdata(combined)

    # 计算 ties 修正项（处理并列秩次）
    _, counts = np.unique(combined, return_counts=True)
    tie_term = np.sum(counts ** 3 - counts)

    # 计算 U 统计量的均值
    mu_U = n1 * n2 / 2

    # 计算 U 统计量的标准差（包含 ties 修正）
    sigma_U_sq = (n1 * n2 / 12) * ((N + 1) - tie_term / (N * (N - 1)))
    sigma_U = np.sqrt(sigma_U_sq)

    # 连续性校正
    if U > mu_U:
        cc = -0.5
    elif U < mu_U:
        cc = +0.5
    else:
        cc = 0.0

    # Z 值
    Z = (U - mu_U + cc) / sigma_U

    # 计算基于 Z 值的效应量（秩双列相关系数 r）
    r_rb = abs(Z) / np.sqrt(N)

    # print("U:", U, "mu_U:", mu_U, "cc:", cc, "sigma_U:", sigma_U, "Z:", Z, "n1:", n1, "n2:", n2, "r_rb:", r_rb)

    # 返回 4 个值
    return U, p_val, r_rb, float(Z)


# ---------- Holm-Bonferroni 校正 ----------

def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    执行 Holm-Bonferroni 多重比较校正，控制家族错误率 (FWER)。
    
    参数:
        p_values: 原始 p 值列表
        alpha: 显著性水平，默认 0.05
    
    返回:
        tuple: (校正后的p值数组, 是否拒绝原假设的布尔数组)
    """
    rejects_holm, pvals_holm_adj, _, _ = multipletests(p_values, alpha=alpha, method="holm")
    return rejects_holm, pvals_holm_adj


# ---------- Benjamini-Hochberg FDR 校正 ----------

def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    执行 Benjamini-Hochberg FDR 校正，控制错误发现率 (FDR)。
    
    参数:
        p_values: 原始 p 值列表
        alpha: 显著性水平，默认 0.05
    
    返回:
        tuple: (校正后的p值数组, 是否拒绝原假设的布尔数组)
    """
    rejects_bh, pvals_bh_adj, _, _ = multipletests(p_values, alpha=alpha, method="fdr_bh")
    return rejects_bh, pvals_bh_adj
