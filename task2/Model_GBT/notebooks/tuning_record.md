# GBT tuning experiment record

Six configurations were evaluated on the same deterministic 2% training and 10% validation samples. Each station-time retains all four rainfall thresholds. GBT-05 was selected using the recorded mean Brier-skill ranking.

This is an experiment record. The original remote `tune_gbt.py` is unavailable in this checkout. Use `task2_gbt_training.ipynb` for new training runs and `test_gbt_final.ipynb` for frozen-model evaluation. The commands below describe the recorded environment and are not a runnable workflow.

```python
import os
print(os.getcwd())
```

```text
/nfs/home/svu/le.pan/singapore-rainfall-analysis-and-prediction/task2/Model_GBT/notebooks
```

```python
%cd /nfs/home/svu/le.pan/singapore-rainfall-analysis-and-prediction/task2/Model_GBT/src
```

```text
/nfs/home/svu/le.pan/singapore-rainfall-analysis-and-prediction/task2/Model_GBT/src
```

```python
import sys
import pyspark

print("Python:", sys.executable)
print("PySpark:", pyspark.__version__)
```

```text
Python: /home/svu/le.pan/rainfall-env/bin/python
PySpark: 3.5.6
```

```python
!python -u tune_gbt.py \
  --dataset-root /scratch/e1496655/task2/final-1458499.stdct-mgmt-02/prepared/ \
  --output-root ../outputs/tuning_stage1 \
  --train-fraction 0.02 \
  --validation-fraction 0.10 \
  --cores 4 \
  --driver-memory 16g
```

```text
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
26/10/15 18:07:29 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
START GBT-01: iter=20 depth=4 step=0.1
Training counts: [{'threshold_mm': 1.0, 'samples': 680053, 'positives': 15931.0}, {'threshold_mm': 10.0, 'samples': 680053, 'positives': 2655.0}, {'threshold_mm': 5.0, 'samples': 680053, 'positives': 5509.0}, {'threshold_mm': 0.0, 'samples': 680053, 'positives': 40950.0}]
26/10/15 18:08:45 WARN InstanceBuilder: Failed to load implementation from:dev.ludovic.netlib.blas.JNIBLAS
{'samples': 668341, 'positives': 44713.0, 'brier': 0.03739437731760529, 'split': 'validation', 'threshold_mm': 0.0, 'positive_rate': 0.06690147694066352, 'areaUnderROC': 0.8777281910406882, 'areaUnderPR': 0.6379972728205175}
{'samples': 668341, 'positives': 17662.0, 'brier': 0.019082346130878294, 'split': 'validation', 'threshold_mm': 1.0, 'positive_rate': 0.02642662951996062, 'areaUnderROC': 0.8570015253017623, 'areaUnderPR': 0.49466147056813925}
{'samples': 668341, 'positives': 5946.0, 'brier': 0.008876017723848982, 'split': 'validation', 'threshold_mm': 5.0, 'positive_rate': 0.008896656048334608, 'areaUnderROC': 0.8432509997262555, 'areaUnderPR': 0.2965785110563906}
{'samples': 668341, 'positives': 2828.0, 'brier': 0.005550870221287012, 'split': 'validation', 'threshold_mm': 10.0, 'positive_rate': 0.00423137290694421, 'areaUnderROC': 0.849717928831288, 'areaUnderPR': 0.19112021426597262}
COMPLETE GBT-01
START GBT-02: iter=40 depth=4 step=0.1
Training counts: [{'threshold_mm': 1.0, 'samples': 680053, 'positives': 15931.0}, {'threshold_mm': 10.0, 'samples': 680053, 'positives': 2655.0}, {'threshold_mm': 5.0, 'samples': 680053, 'positives': 5509.0}, {'threshold_mm': 0.0, 'samples': 680053, 'positives': 40950.0}]
{'samples': 668341, 'positives': 44713.0, 'brier': 0.03655904217309721, 'split': 'validation', 'threshold_mm': 0.0, 'positive_rate': 0.06690147694066352, 'areaUnderROC': 0.8740028951713144, 'areaUnderPR': 0.6417382367670511}
{'samples': 668341, 'positives': 17662.0, 'brier': 0.018052533884969853, 'split': 'validation', 'threshold_mm': 1.0, 'positive_rate': 0.02642662951996062, 'areaUnderROC': 0.8336436104169195, 'areaUnderPR': 0.49670595846283583}
{'samples': 668341, 'positives': 5946.0, 'brier': 0.007739185565420598, 'split': 'validation', 'threshold_mm': 5.0, 'positive_rate': 0.008896656048334608, 'areaUnderROC': 0.8774375928291304, 'areaUnderPR': 0.2986475886628962}
{'samples': 668341, 'positives': 2828.0, 'brier': 0.004344576775136742, 'split': 'validation', 'threshold_mm': 10.0, 'positive_rate': 0.00423137290694421, 'areaUnderROC': 0.8858739524525125, 'areaUnderPR': 0.19316515317562896}
COMPLETE GBT-02
START GBT-03: iter=60 depth=4 step=0.05
Training counts: [{'threshold_mm': 1.0, 'samples': 680053, 'positives': 15931.0}, {'threshold_mm': 10.0, 'samples': 680053, 'positives': 2655.0}, {'threshold_mm': 5.0, 'samples': 680053, 'positives': 5509.0}, {'threshold_mm': 0.0, 'samples': 680053, 'positives': 40950.0}]
{'samples': 668341, 'positives': 44713.0, 'brier': 0.03680671046552346, 'split': 'validation', 'threshold_mm': 0.0, 'positive_rate': 0.06690147694066352, 'areaUnderROC': 0.8732075879414702, 'areaUnderPR': 0.6427412132035456}
{'samples': 668341, 'positives': 17662.0, 'brier': 0.018363167097424426, 'split': 'validation', 'threshold_mm': 1.0, 'positive_rate': 0.02642662951996062, 'areaUnderROC': 0.8335903950553974, 'areaUnderPR': 0.4969047772800085}
{'samples': 668341, 'positives': 5946.0, 'brier': 0.008080759176988923, 'split': 'validation', 'threshold_mm': 5.0, 'positive_rate': 0.008896656048334608, 'areaUnderROC': 0.8786208694267043, 'areaUnderPR': 0.2996420286020301}
{'samples': 668341, 'positives': 2828.0, 'brier': 0.004712466917328817, 'split': 'validation', 'threshold_mm': 10.0, 'positive_rate': 0.00423137290694421, 'areaUnderROC': 0.8860121760012633, 'areaUnderPR': 0.19327756056191125}
COMPLETE GBT-03
START GBT-04: iter=40 depth=3 step=0.1
Training counts: [{'threshold_mm': 1.0, 'samples': 680053, 'positives': 15931.0}, {'threshold_mm': 10.0, 'samples': 680053, 'positives': 2655.0}, {'threshold_mm': 5.0, 'samples': 680053, 'positives': 5509.0}, {'threshold_mm': 0.0, 'samples': 680053, 'positives': 40950.0}]
{'samples': 668341, 'positives': 44713.0, 'brier': 0.03677186004474567, 'split': 'validation', 'threshold_mm': 0.0, 'positive_rate': 0.06690147694066352, 'areaUnderROC': 0.873473892982488, 'areaUnderPR': 0.644703613530029}
{'samples': 668341, 'positives': 17662.0, 'brier': 0.018036959090798502, 'split': 'validation', 'threshold_mm': 1.0, 'positive_rate': 0.02642662951996062, 'areaUnderROC': 0.8571088941318034, 'areaUnderPR': 0.49577465937935167}
{'samples': 668341, 'positives': 5946.0, 'brier': 0.007731538487112824, 'split': 'validation', 'threshold_mm': 5.0, 'positive_rate': 0.008896656048334608, 'areaUnderROC': 0.8433087209879544, 'areaUnderPR': 0.3024231006378101}
{'samples': 668341, 'positives': 2828.0, 'brier': 0.004366034774028599, 'split': 'validation', 'threshold_mm': 10.0, 'positive_rate': 0.00423137290694421, 'areaUnderROC': 0.8496453693916474, 'areaUnderPR': 0.19798344912577007}
COMPLETE GBT-04
START GBT-05: iter=40 depth=5 step=0.1
Training counts: [{'threshold_mm': 1.0, 'samples': 680053, 'positives': 15931.0}, {'threshold_mm': 10.0, 'samples': 680053, 'positives': 2655.0}, {'threshold_mm': 5.0, 'samples': 680053, 'positives': 5509.0}, {'threshold_mm': 0.0, 'samples': 680053, 'positives': 40950.0}]
{'samples': 668341, 'positives': 44713.0, 'brier': 0.03649413287162647, 'split': 'validation', 'threshold_mm': 0.0, 'positive_rate': 0.06690147694066352, 'areaUnderROC': 0.8770345776908974, 'areaUnderPR': 0.6452556732672143}
{'samples': 668341, 'positives': 17662.0, 'brier': 0.017941201269324414, 'split': 'validation', 'threshold_mm': 1.0, 'positive_rate': 0.02642662951996062, 'areaUnderROC': 0.8815875778712712, 'areaUnderPR': 0.4897402288499753}
{'samples': 668341, 'positives': 5946.0, 'brier': 0.007754533084659855, 'split': 'validation', 'threshold_mm': 5.0, 'positive_rate': 0.008896656048334608, 'areaUnderROC': 0.8889701245087127, 'areaUnderPR': 0.29274647935155357}
{'samples': 668341, 'positives': 2828.0, 'brier': 0.004340757799098651, 'split': 'validation', 'threshold_mm': 10.0, 'positive_rate': 0.00423137290694421, 'areaUnderROC': 0.8990918268682058, 'areaUnderPR': 0.1883591548304611}
COMPLETE GBT-05
START GBT-06: iter=60 depth=5 step=0.05
Training counts: [{'threshold_mm': 1.0, 'samples': 680053, 'positives': 15931.0}, {'threshold_mm': 10.0, 'samples': 680053, 'positives': 2655.0}, {'threshold_mm': 5.0, 'samples': 680053, 'positives': 5509.0}, {'threshold_mm': 0.0, 'samples': 680053, 'positives': 40950.0}]
{'samples': 668341, 'positives': 44713.0, 'brier': 0.03672521841211432, 'split': 'validation', 'threshold_mm': 0.0, 'positive_rate': 0.06690147694066352, 'areaUnderROC': 0.8774807793698185, 'areaUnderPR': 0.6369741681707952}
{'samples': 668341, 'positives': 17662.0, 'brier': 0.01825397929973507, 'split': 'validation', 'threshold_mm': 1.0, 'positive_rate': 0.02642662951996062, 'areaUnderROC': 0.88581491950119, 'areaUnderPR': 0.49421286802710707}
{'samples': 668341, 'positives': 5946.0, 'brier': 0.008083943695214707, 'split': 'validation', 'threshold_mm': 5.0, 'positive_rate': 0.008896656048334608, 'areaUnderROC': 0.889538747907642, 'areaUnderPR': 0.29581338996399864}
{'samples': 668341, 'positives': 2828.0, 'brier': 0.004697318323591977, 'split': 'validation', 'threshold_mm': 10.0, 'positive_rate': 0.00423137290694421, 'areaUnderROC': 0.8994395034872343, 'areaUnderPR': 0.1911484513188853}
COMPLETE GBT-06
Validation ranking (Brier skill first, PR lift second):
{'run': 'GBT-05', 'mean_pr_auc_lift': 26.399266673039747, 'mean_brier_skill': 0.20210166446622435, 'mean_roc_auc': 0.8866710267347718, 'mean_pr_auc': 0.40402538407480104}
{'run': 'GBT-02', 'mean_pr_auc_lift': 26.901793557408286, 'mean_brier_skill': 0.20096845578070657, 'mean_roc_auc': 0.8677395127174692, 'mean_pr_auc': 0.40756423426710303}
{'run': 'GBT-04', 'mean_pr_auc_lift': 27.294834894934155, 'mean_brier_skill': 0.19921114620796712, 'mean_roc_auc': 0.8558842193734733, 'mean_pr_auc': 0.4102212056682403}
{'run': 'GBT-06', 'mean_pr_auc_lift': 26.66161199139163, 'mean_brier_skill': 0.16764130075183636, 'mean_roc_auc': 0.8880684875664712, 'mean_pr_auc': 0.40453721937019654}
{'run': 'GBT-03', 'mean_pr_auc_lift': 26.94200788739609, 'mean_brier_skill': 0.16544544301387842, 'mean_roc_auc': 0.8678577571062088, 'mean_pr_auc': 0.40814139491187384}
{'run': 'GBT-01', 'mean_pr_auc_lift': 26.689510946003164, 'mean_brier_skill': 0.08381060092377202, 'mean_roc_auc': 0.8569246612249984, 'mean_pr_auc': 0.405089367177755}
Review all four thresholds before choosing a final model.
```

```python
import pandas as pd

results = pd.read_csv("../outputs/tuning_stage1/tuning_summary.csv")

display(results)
```
