from typing import Dict, List
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scienceplots


def setup_publication_style():
    """
    设置符合顶级会议论文标准的绘图风格（CHI、IEEE VR等）。
    
    特点：
    - 使用 scienceplots 的学术论文风格
    - 支持中文显示（SimHei 黑体）
    - 高分辨率、清晰的线条和边框
    - 适合双栏排版的字号设置
    
    建议在程序入口处调用一次即可。
    """
    # 设置中文字体支持（必须在 style.use 之前设置）
    rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']  # 中文字体
    rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 使用 scienceplots 的学术论文风格
    plt.style.use(['science', 'ieee', 'no-latex'])
    
    # 设置学术论文标准字体
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.size'] = 10
    rcParams['axes.labelsize'] = 10
    rcParams['axes.titlesize'] = 11
    rcParams['xtick.labelsize'] = 9
    rcParams['ytick.labelsize'] = 9
    rcParams['legend.fontsize'] = 9
    
    # 设置线条和边框样式（清晰、专业）
    rcParams['axes.linewidth'] = 1.0
    rcParams['grid.linewidth'] = 0.5
    rcParams['lines.linewidth'] = 1.5
    rcParams['patch.linewidth'] = 1.0
    
    # 设置图形边距和布局
    rcParams['figure.autolayout'] = False
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.pad_inches'] = 0.05
    
    # 设置高质量输出
    rcParams['savefig.dpi'] = 300
    rcParams['figure.dpi'] = 100


INPUT_EXCEL_PATH = Path(__file__).parent.parent / "data/Preprocessed_data.xlsx"  # TODO: 改成你的 Excel 文件路径
OUTPUT_EXCEL_PATH = Path(__file__).parent.parent / "output/Analysis.xlsx"  # TODO: 输出报告文件名
VISUALIZATION_DIR = Path(__file__).parent.parent / "output/visualization"
SHEET_NAME = "sheet"  # 可以是 sheet 名，也可以是索引 0,1,...

# 分组信息
GROUP_COL = "组别"  # TODO: 分组变量列名，例如 'group'
GROUP_LABEL_A = "实验组"  # TODO: 触觉组在 Excel 里的取值
GROUP_LABEL_B = "对照组"  # TODO: 手势组在 Excel 里的取值

# ========= 基线特征检验（多个卡方变量） =========
BASELINE_CATEGORICAL_VARS = [
    "年龄",
    "性别",
    "惯用手",
    "触觉经验",
    "VR经验",
    "双经验",
]

# 前测成绩（连续变量 → Welch t）
BASELINE_CONTINUOUS_VARS = [
    "前测总分",
    "后测总分",
    "提升",
]

# 正态性检验的 alpha
NORMALITY_ALPHA = 0.05

TEST_FUNC_NAME = "mannwhitney"

# 变量模块（方便分别做校正与汇报）
# 注意：变量名必须与 Excel 文件中的列名完全一致
VARIABLE_BLOCKS: Dict[str, List[str]] = {
    # 认知负荷量表
    "认知负荷": [
        "ICL", "GCL", "ECL",
        "认知负荷总分"
    ],
    # 沉浸感量表
    "沉浸感": [
        "G1", "SP", "INV", "REAL",
        "沉浸感总分",
    ],
    # 具身感量表
    "具身感": [
        "AC", "CO",
        "具身感总分",
    ],
    # 易用性量表
    "易用性": [
        "AIH", "CPH", 
        "易用性总分",
    ],
    # 知识测试
    "知识测试": [
        "前测总分",
        "后测总分",
        "提升",
    ],
    # 交互数据
    "交互数据": [
        "time", 
        "empty",
        "empty_rate",
        "high_rate",       
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
parser.add_argument(
    "--apply_timestamp",
    type=eval,
    default=True,
    help="是否在输出的描述性统计表文件名中添加时间戳"
)
parser.add_argument(
    "--visualization_dir",
    type=Path,
    default=VISUALIZATION_DIR,
    help="可视化结果保存目录"
)


args = parser.parse_args()  # ★★ 关键
