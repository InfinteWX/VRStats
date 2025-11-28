from typing import Dict, List
import argparse
from pathlib import Path

INPUT_EXCEL_PATH = Path(__file__).parent.parent / "data/all.xlsx"  # TODO: 改成你的 Excel 文件路径
OUTPUT_EXCEL_PATH = Path(__file__).parent.parent / "output/report.xlsx"  # TODO: 输出报告文件名
SHEET_NAME = 0  # 可以是 sheet 名，也可以是索引 0,1,...

# 分组信息
GROUP_COL = "组别"  # TODO: 分组变量列名，例如 'group'
GROUP_LABEL_A = "触觉组"  # TODO: 触觉组在 Excel 里的取值
GROUP_LABEL_B = "手势组"  # TODO: 手势组在 Excel 里的取值

# ========= 基线特征检验（多个卡方变量） =========
BASELINE_CATEGORICAL_VARS = [
    "年龄",
    "性别",
    "VR使用情况",  # 新增
]

# 前测成绩（连续变量 → Welch t）
BASELINE_CONTINUOUS_VARS = [
    "Pre_test",
    "Post_test",
    "Gain",
]

# 正态性检验的 alpha
NORMALITY_ALPHA = 0.05

TEST_FUNC_NAME = "mannwhitney"

# 变量模块（方便分别做校正与汇报）
VARIABLE_BLOCKS: Dict[str, List[str]] = {
    # 问卷量表
    "IEG_Total": [
        "Intrinsic",  # 认知负荷
        "Extraneous",  # 学习动机
        "Germane",  # 沉浸感
        "IEG_Total"
    ],
    "ARCS_Total": [
        "Attention",  # 注意
        "Relevance",  # 相关性
        "Confidence",  # 信心
        "Satisfaction",  # 满意度
        "ARCS_Total"
    ],
    "PIR_Total": [
        "Presence",  # 存在感
        "Involvement",  # 参与度
        "Realism",  # 真实感
        "PIR_Total"
    ],
    # 知识测试
    "Knowledge": [
        "Pre_test",  # 知识测试前测分数
        "Post_test",  # 知识测试后测分数
        "Gain",  # 知识测试提升
    ],
    # EEG 指标（示例）
    "EEG": [
        "Alpha_Fz",
        "Alpha_Cz",
        "Alpha_Pz",
        "Alpha_Oz",
        "Beta_Fz",
        "Beta_Cz",
        "Beta_Pz",
        "Beta_Oz",
        "Theta_Fz",
        "Theta_Cz",
        "Theta_Pz",
        "Theta_Oz",
        # ... 自行补充
    ],
}

parser = argparse.ArgumentParser()  # 不要叫 parse_args
parser.add_argument(
    "--input_excel_path",
    type=Path,
    default=INPUT_EXCEL_PATH,
    help="输入的 Excel 数据文件路径"
)
parser.add_argument(
    "--input_sheet_name",
    type=str,
    default=SHEET_NAME,
    help="输入的 Excel 工作表名称或索引，默认为第一个工作表"
)
parser.add_argument(
    "--group_col",
    type=str,
    default=GROUP_COL,
    help="分组变量列名，例如 '组别'"
)
parser.add_argument(
    "--variable_blocks",
    type=lambda s: eval(s),
    default=VARIABLE_BLOCKS,
    help="变量块字典"
)
parser.add_argument(
    "--output_excel_path",
    type=Path,
    default=OUTPUT_EXCEL_PATH,
    help="输出的 Excel 文件路径"
)
parser.add_argument(
    "--normality_alpha",
    type=float,
    default=NORMALITY_ALPHA,
    help="正态性检验的显著性水平 alpha"
)
parser.add_argument(
    "--group_label_a",
    type=str,
    default=GROUP_LABEL_A,
    help="触觉组在 Excel 里的取值"
)
parser.add_argument(
    "--group_label_b",
    type=str,
    default=GROUP_LABEL_B,
    help="手势组在 Excel 里的取值"
)
parser.add_argument(
    "--test_func_name",
    type=str,
    default="mannwhitney",
    help="两组比较的检验方法，支持 'ttest' 或 'mannwhitney'"
)
parser.add_argument(
    "--baseline_categorical_vars",
    type=eval,
    default=BASELINE_CATEGORICAL_VARS,
    help="基线特征检验的分类变量列表"
)
parser.add_argument(
    "--baseline_continuous_vars",
    type=eval,
    default=BASELINE_CONTINUOUS_VARS,
    help="基线特征检验的连续变量"
)
parser.add_argument(
    "--add_blank_rows",
    type=eval,
    default=True,
    help="是否在输出的描述性统计表中添加空行以区分不同模块"
)

args = parser.parse_args()  # ★★ 关键
