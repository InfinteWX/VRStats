[toc]

# VRStats：两组比较自动统计分析工具

---

## 1. 项目简介

VRStats 是一个基于 **Polars** 开发的高性能两组比较自动统计分析工具，专为实验研究设计：

- 实验组 vs 控制组
- 触觉反馈 vs 手势交互
- 条件 A vs 条件 B

### 核心特性

✨ **高性能数据处理**：使用 Polars 库，处理速度比传统方案快 5-10 倍

📊 **全自动分析流程**：

- 基线特征检验（卡方检验 + Welch t 检验）
- 描述性统计（均值、标准差、中位数、四分位数等）
- 正态性检验（Shapiro-Wilk）
- 自动选择检验方法：
  - 正态分布：Welch t 检验（不假定方差齐性）
  - 非正态分布：Mann-Whitney U 检验
- 多重比较校正：
  - Holm-Bonferroni（控制家族错误率 FWER）
  - Benjamini-Hochberg（控制错误发现率 FDR）

📈 **专业输出**：

- Excel 多工作表报告（baseline、descriptives、normality、tests）
- 高质量学术论文级可视化图表（使用 scienceplots，符合 Nature/Science/IEEE 标准）
- 自动模块分隔，便于阅读
- 包含完整的统计量、p 值和效应量

### 适用场景

非常适合心理学实验、HCI/VR 用户研究、教育实验等需要两组比较的研究情境。

---

## 2. 技术栈与环境要求

### 2.1 核心技术

- **数据处理**：Polars（高性能 DataFrame 库）
- **统计计算**：SciPy、NumPy、Statsmodels
- **可视化**：scienceplots + Matplotlib（符合 Nature/Science/IEEE 等顶级期刊标准）
- **Excel 处理**：Polars + fastexcel + xlsxwriter
- **依赖管理**：uv（快速 Python 包管理器）

### 2.2 运行环境

- **操作系统**：Windows 10/11、macOS、Linux
- **Python 版本**：3.11 或更高
- **依赖管理工具**：[uv](https://github.com/astral-sh/uv)

### 2.3 项目结构

```text
VRStats/
├── pyproject.toml      # 项目配置和依赖
├── uv.lock            # 依赖版本锁定
├── README.md          # 项目文档
├── data/
│   └── Origin.xlsx    # 原始数据文件（需自行准备）
├── output/
│   ├── Analysis.xlsx  # 自动生成的分析报告
│   └── visualization/ # 可视化图表输出目录
│       ├── baseline/      # 基线特征可视化
│       ├── descriptives/  # 描述性统计可视化
│       ├── normality/     # 正态性检验Q-Q图
│       └── tests/         # 两组检验森林图
└── src/
    ├── main.py        # 主入口程序
    ├── setting.py     # 统一配置文件（包含绘图风格设置）
    ├── parts/
    │   ├── baseline.py          # 基线特征检验 + 可视化
    │   ├── descriptives.py      # 描述性统计 + 可视化
    │   ├── normality.py         # 正态性检验 + 可视化
    │   └── two_group_tests.py   # 两组比较检验 + 可视化
    └── utils/
        └── func.py              # 工具函数
```

### 2.4 安装步骤

1. **克隆或下载项目**

   ```bash
   git clone <repository-url> VRStats
   cd VRStats
   ```
2. **安装 uv**（如果尚未安装）

   Windows (PowerShell):

   ```powershell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   macOS/Linux:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
3. **安装项目依赖**

   ```bash
   uv sync
   ```
4. **验证安装**

   ```bash
   uv run python -c "import polars; print('VRStats 环境配置成功！')"
   ```

---

## 3. 数据准备

### 3.1 Excel 文件要求

**文件位置**：`data/Origin.xlsx`（默认）

**基本要求**：

- 第一行必须是列名（表头）
- 每行代表一个被试/样本
- 必须包含分组变量列
- 必须包含要分析的指标变量列

### 3.2 必需的列

1. **分组变量列**（例如 `Group`）

   - 标识每个样本属于哪一组
   - 示例值：`Haptic`、`Gesture`
2. **基线特征变量**（用于组间基线检验）

   - 分类变量（如年龄、性别、VR经验）→ 卡方检验
   - 连续变量（如前测成绩）→ Welch t 检验
3. **分析指标变量**

   - 需要进行两组比较的指标
   - 可按模块分组（如问卷量表、知识测试、生理指标等）

### 3.3 数据示例

| Group   | Age | Gender | VR_Experience | Pre_test | Intrinsic | Extraneous | Post_test |
| ------- | --- | ------ | ------------- | -------- | --------- | ---------- | --------- |
| Haptic  | 22  | Male   | Yes           | 75       | 5.2       | 3.1        | 82        |
| Gesture | 24  | Female | No            | 73       | 4.8       | 3.5        | 78        |
| Haptic  | 21  | Male   | Yes           | 78       | 5.5       | 2.9        | 85        |

### 3.4 注意事项

⚠️ **列名规范**：

- 建议使用英文字母、数字和下划线
- 避免使用空格和特殊字符
- 列名必须与 `setting.py` 中的配置完全一致

⚠️ **数据质量**：

- 数值列应为纯数字，避免包含文本
- 避免空行和空列
- 缺失值用空白单元格表示

---

## 4. 配置说明（`src/setting.py`）

配置文件是整个分析的控制中心，所有参数都在此设置。

### 4.1 文件路径配置

```python
# 输入数据文件
INPUT_EXCEL_PATH = Path(__file__).parent.parent / "data/Origin.xlsx"

# 输出报告文件
OUTPUT_EXCEL_PATH = Path(__file__).parent.parent / "output/Analysis.xlsx"

# 工作表名称或索引（0 表示第一个工作表）
SHEET_NAME = "Sheet1"
```

### 4.2 分组信息配置

```python
# 分组变量列名
GROUP_COL = "Group"

# 第一组标签（如触觉组）
GROUP_LABEL_A = "Haptic"

# 第二组标签（如手势组）
GROUP_LABEL_B = "Gesture"
```

### 4.3 基线特征变量配置

```python
# 分类变量（使用卡方检验）
BASELINE_CATEGORICAL_VARS = [
    "Age",           # 年龄
    "Gender",        # 性别
    "VR_Experience", # VR使用经验
]

# 连续变量（使用 Welch t 检验）
BASELINE_CONTINUOUS_VARS = [
    "Pre_test",  # 前测成绩
    "Post_test", # 后测成绩
    "Gain",      # 提升分数
]
```

### 4.4 检验方法配置

```python
# 正态性检验的显著性水平
NORMALITY_ALPHA = 0.05

# 两组比较的检验方法
# "ttest" - Welch t 检验
# "mannwhitney" - Mann-Whitney U 检验
TEST_FUNC_NAME = "mannwhitney"
```

### 4.5 变量模块配置

```python
VARIABLE_BLOCKS: Dict[str, List[str]] = {
    # 问卷量表 - 认知负荷
    "IEG_Total": [
        "Intrinsic",   # 内在认知负荷
        "Extraneous",  # 外在认知负荷
        "Germane",     # 生成认知负荷
        "IEG_Total"    # 总分
    ],
  
    # 问卷量表 - 学习动机
    "ARCS_Total": [
        "Attention",    # 注意
        "Relevance",    # 相关性
        "Confidence",   # 信心
        "Satisfaction", # 满意度
        "ARCS_Total"    # 总分
    ],
  
    # 知识测试
    "Knowledge": [
        "Pre_test",  # 前测
        "Post_test", # 后测
        "Gain",      # 提升
    ],
  
    # 其他模块...
}
```

---

## 5. 使用指南

### 5.1 快速开始

1. **准备数据文件**

   ```bash
   # 将数据文件放置到 data 目录
   cp your_data.xlsx data/Origin.xlsx
   ```
2. **配置参数**

   ```bash
   # 编辑配置文件
   # 根据你的数据调整分组变量和指标变量
   nano src/setting.py  # 或使用其他编辑器
   ```
3. **运行分析**

   ```bash
   uv run -m src.main
   ```
4. **查看结果**

   ```bash
   # 输出文件位于
   output/Analysis.xlsx
   ```

### 5.2 命令行参数

可以通过命令行参数临时覆盖配置文件中的设置：

```bash
# 使用不同的输入文件
uv run -m src.main --input_excel_path data/experiment2.xlsx

# 更改分组列名
uv run -m src.main --group_col "实验组别"

# 更改检验方法
uv run -m src.main --test_func_name "ttest"

# 禁用空行分隔
uv run -m src.main --add_blank_rows False
```

---

## 6. 输出结果

### 6.1 输出文件

分析完成后，会在 `output/` 目录生成以下文件：

#### 📊 Excel 分析报告 (`Analysis.xlsx`)

包含以下工作表：

##### 📋 **baseline** 工作表

基线特征检验结果

- 分类变量：卡方检验结果 + Cramér's V 效应量
- 连续变量：Welch t 检验结果

#### 📊 **descriptives** 工作表

描述性统计结果（按组、按模块）

- 样本量（n）
- 均值（mean）
- 标准差（std）
- 中位数（median）
- 四分位数（Q1、Q3）
- 最小值、最大值

#### 📈 **normality** 工作表

正态性检验结果

- Shapiro-Wilk 统计量
- p 值
- 是否符合正态分布（True/False）

#### 🔬 **tests** 工作表

两组比较检验结果

- 检验方法（ttest 或 mannwhitney）
- 统计量（t 值或 U 值）
- 原始 p 值
- 效应量（Cohen's d 或 rank-biserial r）
- Holm-Bonferroni 校正后的 p 值
- Benjamini-Hochberg FDR 校正后的 p 值
- 显著性判断（True/False）

#### 🎨 可视化图表

所有可视化图表保存在 `output/visualization/` 目录下，采用学术论文标准：

##### **baseline/** - 基线特征可视化

- **分类变量**：分组条形图（含数值标签）
- **连续变量**：小提琴图 + 箱线图 + 数据点
- 文件命名：`categorical_{变量名}.png`、`continuous_{变量名}.png`

##### **descriptives/** - 描述性统计可视化

- **组间对比柱状图**：均值 ± 标准误（Mean±SEM）
- 按模块分类，每个变量一张图
- 文件命名：`{模块名}/{变量名}_comparison.png`

##### **normality/** - 正态性检验可视化

- **Q-Q 图**：理论分位数 vs 样本分位数
- 每个变量的所有组并排显示
- 文件命名：`{模块名}/{变量名}_qqplot.png`

##### **tests/** - 两组检验可视化

- **森林图**：效应量 + 95% 置信区间
- 显著性颜色标记（红色 p<0.05，蓝色 p≥0.05）
- 文件命名：`{模块名}_forest.png`

**图表特点**：

- ✅ 300 DPI 高分辨率，适合论文发表
- ✅ scienceplots 学术风格（Nature/Science/IEEE 标准）
- ✅ 色盲友好配色方案
- ✅ 中文字体支持（SimHei 黑体）
- ✅ 双栏排版优化尺寸（3.5 英寸宽）
- ✅ 专业排版，无需后期调整

### 6.2 结果解读

#### 基线特征检验

- **目的**：验证两组在基线特征上是否均衡
- **p > 0.05**：两组无显著差异，基线均衡✅
- **p < 0.05**：两组存在显著差异，需要注意⚠️

#### 正态性检验

- **p > 0.05**：数据符合正态分布，可使用 t 检验
- **p < 0.05**：数据不符合正态分布，应使用非参数检验

#### 两组比较检验

- **原始 p 值**：未校正的显著性
- **p_holm**：Holm-Bonferroni 校正（更保守）
- **p_bh**：Benjamini-Hochberg FDR 校正（较宽松）
- **效应量**：
  - Cohen's d: 0.2(小)、0.5(中)、0.8(大)
  - rank-biserial r: 解释类似相关系数，0.1(小)、0.3(中)、0.5(大)

### 6.3 后续分析

生成的 Excel 文件可以：

- 直接用于论文报告
- 导入 SPSS/R/Python 进行进一步分析
- 使用 Excel 制作图表
- 作为研究数据的存档
