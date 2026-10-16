"""Create report figures from the recorded, completed GBT-05 test evaluation."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
# Resolve paths relative to this script so the working directory is unrestricted.
base = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=base / 'outputs/tuning_stage1/GBT-05/test_evaluation/20261016T103557030872Z', help='Completed GBT-05 test evaluation directory')
src = parser.parse_args().source.resolve()
out=base/'results/figures';out.mkdir(parents=True,exist_ok=True)
m=pd.read_csv(src/'test_metrics.csv').sort_values('threshold_mm')
b=pd.read_csv(src/'test_reliability_bins.csv')
f=pd.read_csv(src/'feature_importance.csv')
c=pd.read_csv(src/'validation_test_comparison.csv').sort_values('threshold_mm')
assert json.loads((src/'test_manifest.json').read_text())['status']=='complete'
# Bin aggregates must reconcile with the metrics before they are plotted.
for t,g in b.groupby('threshold_mm'):
 r=m[m.threshold_mm==t].iloc[0]
 assert g.samples.sum()==r.samples
 assert abs(np.average(g.observed_rate,weights=g.samples)-r.positive_rate)<1e-10
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titlesize':12,'savefig.facecolor':'white'})
blue='#25678D';orange='#C77730'
def save(fig,name):
 """Export matching raster and vector figures, then release the canvas."""
 fig.savefig(out/(name+'.png'),dpi=300,bbox_inches='tight')
 fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
 plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(10,8))

# Preserve sparse bins while marking their uncertainty in the diagrams.
cal=[]
for ax,(t,g) in zip(axs.flat,b.groupby('threshold_mm',sort=True)):
 g=g.sort_values('mean_probability');ax.plot([0,1],[0,1],'--',color='0.4',lw=1,label='Perfect calibration')
 ax.plot(g.mean_probability,g.observed_rate,color=blue,lw=1.4)
 for _,r in g.iterrows():ax.scatter(r.mean_probability,r.observed_rate,s=35,facecolor=blue if r.samples>=100 else 'white',edgecolor=blue,zorder=3)
 ax.set(xlim=(0,1),ylim=(0,1),title=f'Rainfall > {t:g} mm',xlabel='Mean predicted probability',ylabel='Observed event frequency')
 ax.grid(alpha=.15)
 inset=ax.inset_axes([.55,.12,.4,.36]);inset.plot([0,.1],[0,.1],'--',color='0.5',lw=.8);inset.plot(g.mean_probability,g.observed_rate,'o-',color=blue,ms=3,lw=1);inset.set(xlim=(0,.1),ylim=(0,.1));inset.tick_params(labelsize=7)
 cal.append(f'> {t:g} mm: overall predicted {np.average(g.mean_probability,weights=g.samples):.3%}, observed {np.average(g.observed_rate,weights=g.samples):.3%}; lowest bin predicted {g.iloc[0].mean_probability:.3%}, observed {g.iloc[0].observed_rate:.3%}, share {g.iloc[0].samples/g.samples.sum():.2%}.')
fig.suptitle('GBT-05 probability calibration on the complete test partition',fontsize=14)
fig.tight_layout(rect=[0,.03,1,.95]);fig.text(.5,.005,'6,335,645 samples per threshold | Insets: 0–0.1 probability | Hollow markers: fewer than 100 samples',ha='center',fontsize=9)
save(fig,'gbt_test_calibration')
labels={'threshold_mm':'Rainfall threshold','rainfall_previous_5m':'Previous 5-minute rainfall','rainfall_previous_30m':'Previous 30-minute rainfall','rainfall_previous_60m':'Previous 60-minute rainfall','hour':'Hour','station_index':'Station identifier','month':'Month'}
f=f.sort_values('importance');fig,ax=plt.subplots(figsize=(9,4.8));bars=ax.barh(f.feature.map(labels),f.importance,color=blue);ax.bar_label(bars,labels=[f'{v:.2%}' for v in f.importance],padding=5);ax.set_xlim(0,.4);ax.xaxis.set_major_formatter(PercentFormatter(1));ax.set(xlabel='Share of recorded tree feature importance',title='GBT-05 feature importance');ax.grid(axis='x',alpha=.15);fig.tight_layout();save(fig,'gbt_feature_importance')
x=np.arange(4);fig,axs=plt.subplots(1,3,figsize=(12,3.9))
for ax,key,title in zip(axs,['areaUnderROC','areaUnderPR','brier'],['ROC-AUC','PR-AUC','Brier score']):
 ax.bar(x-.18,c['validation_'+key],.36,label='Validation (10% sample)',color=orange)
 ax.bar(x+.18,c['test_'+key],.36,label='Test (complete partition)',color=blue)
 ax.set_xticks(x,['>0','>1','>5','>10']);ax.set(xlabel='Rainfall threshold (mm)',title=title);ax.grid(axis='y',alpha=.15)
 if key!='brier':ax.set_ylim(0,1)
axs[0].legend(fontsize=8);fig.suptitle('GBT-05 validation and test performance',fontsize=14);fig.tight_layout();save(fig,'gbt_validation_test_comparison')
fig,ax=plt.subplots(figsize=(9,4.8))
for delta,key,color in [(-.18,'precision',blue),(.18,'recall',orange)]:
 bars=ax.bar(x+delta,m[key],.36,label=key.title(),color=color);ax.bar_label(bars,labels=[f'{v:.1%}' for v in m[key]],padding=3,fontsize=9)
ax.set_xticks(x,['>0','>1','>5','>10']);ax.set(ylim=(0,1),xlabel='Rainfall threshold (mm)',ylabel='Proportion',title='GBT-05 test precision and recall at probability cutoff 0.5');ax.yaxis.set_major_formatter(PercentFormatter(1));ax.legend();ax.grid(axis='y',alpha=.15);fig.tight_layout();save(fig,'gbt_precision_recall')
text='''# GBT report figures and captions

Source: completed GBT-05 test run `20261016T103557030872Z`. Each threshold uses 6,335,645 test samples. The frozen model used a 2% training sample. Figures are exported as 300 dpi PNG and vector PDF. CSV metrics and reliability-bin totals were reconciled before plotting. No model was retrained.

## Section 6.3 — gbt_test_calibration.png

Figure X. GBT-05 reliability diagrams for future 30-minute rainfall exceedance on the complete test partition. Points compare mean predicted probabilities with observed event frequencies within occupied equal-width probability bins. The dashed diagonal indicates perfect calibration; points below it indicate overprediction and points above it indicate underprediction. Insets magnify the 0–0.1 probability region without changing bins. Hollow markers identify bins with fewer than 100 samples. No post-training calibration was applied.

Calibration checks:

'''+ '\n'.join(cal)+'''

These overall means do not describe all within-bin errors. The lowest probability bin overpredicts event frequency at every threshold; the pattern in higher bins must be read from the individual diagrams. Event prevalence and bin population should accompany any calibration claim.

## Section 6.5 — gbt_feature_importance.png

Figure X. Recorded tree feature importance for the selected GBT-05 model. Recent rainfall over the previous 5, 30 and 60 minutes accounts for 53.63% of total importance. Importance indicates model usage rather than causal influence; overlapping historical windows are correlated.

## Section 6.1 or 6.4 — gbt_validation_test_comparison.png

Figure X. GBT-05 discrimination and probability error on the sampled validation set and complete test partition. Each threshold has 668,341 validation samples and 6,335,645 test samples. ROC-AUC and PR-AUC are lower on test data at every threshold. Lower test Brier scores do not independently establish better calibration because event prevalence also differs. Higher AUC and lower Brier scores indicate better performance.

## Section 6.2 — gbt_precision_recall.png

Figure X. Test precision and recall for GBT-05 at a fixed classification probability cutoff of 0.5. Recall falls to 4.02% for rainfall exceeding 10 mm, indicating that most heavy-rainfall exceedances are missed at this cutoff. The cutoff was not optimized using test outcomes.
'''
(out/'captions_and_notes.md').write_text(text,encoding='utf-8')
print('\n'.join(cal));print('Created four PNG/PDF figure pairs:',out.resolve())
