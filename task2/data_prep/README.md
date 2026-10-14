# Task 2 — Rainfall Prediction Dataset Preparation

截至 2026-10-14，已完成 2017–2024 年全量预测数据集准备、检查和远程导出；GBT 训练及模型评估尚未实现。本文依据本地代码、Notebook 保存的执行输出和项目所有者提供的说明。本次没有重新运行全量作业或直接读取远程数据。

## 已完成

1. 合并八年 cleaned Parquet，检查站点、时间戳、五分钟网格、无效降雨值与重复记录；Notebook 输出显示通过。
2. 使用按站点划分的真实时间窗口，构建历史降雨特征及未来标签。
3. 只保留历史十二个、未来六个观测完整的时间点；缺失不填零。
4. 为 0、1、5、10 mm 四个阈值生成样本，按年份划分数据。
5. 汇总年度完整性、各阈值类别分布，导出 Parquet、CSV 和 manifest。

## 特征与标签

预测目标：在 t 的观测已可用时，预测未来 30 分钟累计降雨严格超过阈值的概率。

`label = float(future_30m_mm > threshold_mm)`

| 输入特征 | 定义 |
|---|---|
| station_id | 站点标识，需类别编码 |
| hour | 新加坡时间的小时 |
| month | 月份 |
| past_5m_mm | t 时刻的五分钟降雨 |
| past_30m_mm | t−25 至 t，共六个观测之和 |
| past_60m_mm | t−55 至 t，共十二个观测之和 |
| threshold_mm | 0、1、5、10 mm |

未来窗口为 t+5 至 t+30，共六个观测。每个有效站点时间点产生四条阈值样本。

`future_30m_mm` 仅用于审计，严禁放入模型输入；`label`、`split` 同样不能作为特征。编码器只在训练集拟合，并明确处理验证、测试中新出现的站点。

**边界规则：**未来标签窗口不能跨分割边界；历史窗口允许使用前一分割的观测，因为预测时这些观测已经发生。原说明中的“跨边界窗口全部排除”应改为这一精确表述。

## 全量结果

以下数值与本地 Notebook 保存输出一致。

| 数据集 | 时期 | 有效站点时间点 | 四阈值样本数 |
|---|---|---:|---:|
| Train | 2017–2022 | 34,023,023 | 136,092,092 |
| Validation | 2023 | 6,686,625 | 26,746,500 |
| Test | 2024 | 6,338,426 | 25,353,704 |
| 合计 | 2017–2024 | 47,048,074 | 188,192,296 |

输入为 48,538,627 条 cleaned observations，保留率约 96.93%，排除 1,490,553 个时间点。历史不完整和未来不完整的计数存在重叠，不能直接相加作为总排除量。2017、2018 年完整性相对较低。

| 阈值 | Train 正例率 | Validation 正例率 | Test 正例率 |
|---|---:|---:|---:|
| 0 mm | 6.01% | 6.67% | 6.36% |
| 1 mm | 2.33% | 2.63% | 2.57% |
| 5 mm | 0.81% | 0.87% | 0.84% |
| 10 mm | 0.39% | 0.41% | 0.38% |

类别高度不平衡。后续逐阈值报告 PR-AUC、ROC-AUC、Brier Score 和概率校准情况，不应只使用 accuracy。

## 未同步至本地的 data

远程 cleaned 输入目录：

```text
/scratch/e0969984/singapore-rainfall-analysis-and-prediction/data/processed/
  rainfall_2017/ ... rainfall_2024/
```

Notebook 保存输出记录的实际导出目录：

```text
/nfs/home/svu/le.pan/singapore-rainfall-analysis-and-prediction/task2/data/task2/notebook_20261014T073533Z/
├── examples/
│   ├── split=train/       # year=2017 至 year=2022 分区
│   ├── split=validation/  # year=2023 分区
│   └── split=test/        # year=2024 分区
├── class_balance.csv
├── quality_by_year.csv
└── manifest.json
```

Parquet 输出是多个文件组成的分区目录，不是单个 data 文件。

| 输出 | 用途 |
|---|---|
| examples/ | 所有模型共用的训练、验证及测试数据 |
| class_balance.csv | 各 split/threshold 的样本数、正例数、正例率 |
| quality_by_year.csv | 年度输入量、窗口不完整数、边界排除数、有效时间点数 |
| manifest.json | 特征、阈值、时区、分割规则、Spark 版本、完成状态 |

Notebook 已保存导出完成输出，代码在写入成功后将 manifest 改为 `complete`。本次未直接核验远程 manifest 或全部 Parquet 文件，训练前应确认状态与分区计数。

本地根目录 data/ 目前只有 geography/。原始 CSV、cleaned Parquet 和 Task 2 导出未在本地发现。task2/.gitignore 排除了 data/，因此远程数据不会随代码提交自动同步。

建议先同步三个小型元数据文件留档；大规模训练继续在 Vanda 进行。以下在本地 PowerShell 执行，将 YOUR_ACCOUNT 替换为自己的登录名：

```powershell
Set-Location 'D:\python\Projects\singapore-rainfall-analysis-and-prediction\task2'
New-Item -ItemType Directory -Force 'data/task2/notebook_20261014T073533Z'
scp YOUR_ACCOUNT@vanda.nus.edu.sg:/nfs/home/svu/le.pan/singapore-rainfall-analysis-and-prediction/task2/data/task2/notebook_20261014T073533Z/manifest.json data/task2/notebook_20261014T073533Z/
scp YOUR_ACCOUNT@vanda.nus.edu.sg:/nfs/home/svu/le.pan/singapore-rainfall-analysis-and-prediction/task2/data/task2/notebook_20261014T073533Z/class_balance.csv data/task2/notebook_20261014T073533Z/
scp YOUR_ACCOUNT@vanda.nus.edu.sg:/nfs/home/svu/le.pan/singapore-rainfall-analysis-and-prediction/task2/data/task2/notebook_20261014T073533Z/quality_by_year.csv data/task2/notebook_20261014T073533Z/
```

## 实现与运行环境

- notebooks/task2_data_preparation.ipynb：已保存全量执行输出。
- src/build_task2_dataset.py：批处理与 Notebook 共用的窗口构建函数。
- 当前使用已有 Jupyter 会话；本地已无独立启动辅助脚本。
- 本地已无 jobs/；使用 Notebook 或直接运行数据准备脚本。
- tests/check_task2_dataset.py：小型 Spark 语义检查。
- docs/：数据契约与远程操作说明。

本次 Notebook 输出记录的环境：CN-149、Python 3.11.17、Spark 3.5.6、local[4]、128 shuffle partitions、Asia/Singapore 时区。原 PBS 指南使用 Spark 3.3.1 模块，是不同配置。

在 Vanda 的 task2/data_prep/ 目录中可运行：

```bash
spark-submit tests/check_task2_dataset.py
spark-submit src/build_task2_dataset.py --input-root /path/to/processed --output-root /path/to/new/output
```

检查脚本覆盖精确窗口、站点隔离、严格阈值、缺失窗口、标签边界及历史跨界。脚本存在，但本次未执行，也未发现保存的 PASS 记录。批处理重跑必须使用新输出目录。

Notebook 的质量 CSV 列名为 boundary_excluded，批处理为 split_boundary_excluded；两者表达同一统计。批处理 manifest 额外包含未来降雨分位数，Notebook manifest 没有该字段。

详见 [数据契约](docs/DATA.md) 和 [Notebook 操作指南](docs/VANDA_NOTEBOOK.md)。

## 下一步

1. 在远程确认 manifest 和分区计数，固定数据版本，下载小型元数据；各模型作者共用同一输出。
2. 运行小型 Spark 语义检查并保存通过记录。
3. 使用训练集各阈值正例率建立常数概率基线。
4. 实现 PySpark GBT：训练集拟合站点编码和特征组装，先用小规模训练样本验证流程，再扩大数据量。
5. 仅用验证集调参、选择类别权重及概率校准。采样或加权后仍保持验证/测试自然分布，并检查概率校准。
6. 配置固定后评估 2024 测试集，逐阈值报告指标，检查概率随阈值升高是否非增。
7. 保存模型、处理器、配置、数据版本、指标和日志，与其他模型比较。

当前完成的是共享建模数据集，尚无模型预测性能或模型优劣结论。

## 目录调整（2026-10-14）

现有代码、Notebook、测试、文档已整体移至 task2/data_prep/。task2/.gitignore 仍在父目录并覆盖本模块。Notebook 新输出仍写入 task2/data/task2/；上面的远程路径和 Notebook 保存输出是历史运行记录，未随本地代码移动。
