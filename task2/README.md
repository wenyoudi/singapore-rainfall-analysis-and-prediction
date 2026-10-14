# Task 2 — Rainfall Prediction

现有数据准备模块已整理到 [data_prep/](data_prep/README.md)，后续模型训练可以在 task2 下独立组织。

```text
task2/
├── .gitignore
├── README.md
└── data_prep/
    ├── README.md
    ├── notebooks/task2_data_preparation.ipynb
    ├── src/build_task2_dataset.py
    ├── tests/check_task2_dataset.py
    └── docs/
```

在 Jupyter 打开 `data_prep/notebooks/task2_data_preparation.ipynb`。Notebook 会查找新的代码目录，新的数据输出仍位于 `task2/data/task2/`，保持原数据布局。Notebook 保存的旧运行路径属于历史记录。

完整结果、远程数据路径及下一步见 [数据准备 README](data_prep/README.md)。本地整理不会自动移动 Vanda 上的代码或数据；需同步 data_prep 文件夹后再使用新路径。
