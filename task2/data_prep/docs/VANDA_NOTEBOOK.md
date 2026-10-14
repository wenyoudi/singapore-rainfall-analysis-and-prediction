# 在已有 Jupyter 中运行数据准备

当前使用已有 Jupyter 会话。本地 jobs/ 和 start_task2_jupyter.py 已不存在；本指南只描述当前文件布局。

1. 将 task2/data_prep/ 整体同步到远程项目的 task2/data_prep/，保留 src、notebooks、tests 的相对位置。
2. 打开 data_prep/notebooks/task2_data_preparation.ipynb。
3. 使用能够导入 PySpark 的 Python 内核，并确认 INPUT 指向可读的 cleaned Parquet。
4. Notebook 会从当前目录及上级目录查找 data_prep/src/build_task2_dataset.py；可从项目根目录、task2、data_prep 或 notebooks 工作目录启动。
5. 数据输出仍在 task2/data/task2/notebook_<timestamp>/，每次使用新目录；保存 Notebook 以保留运行输出。

已生成全量数据时，后续建模应直接读取已有 examples/，无需重新运行数据准备。

已保存的全量运行记录来自 CN-149、Spark 3.5.6、Python 3.11.17。迁移代码不会移动远程数据或改变历史输出。Jupyter 会话如何申请计算资源，继续使用你当前已可用的启动方式。

[已完成工作与远程数据](../README.md)
