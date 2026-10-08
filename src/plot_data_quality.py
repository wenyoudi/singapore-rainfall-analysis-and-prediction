"""Plot Section 3.1 data-quality figures from the Spark gap-analysis CSVs.
Run from anywhere: python src/plot_data_quality.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / 'outputs' / 'gap_analysis'
OUTPUT = ROOT / 'outputs' / 'figures' / 'data_quality'
# Also supports running the script directly from a directory containing the CSV files.
if not INPUT.exists() and (Path(__file__).resolve().parent / 'station_coverage.csv').exists():
    INPUT = Path(__file__).resolve().parent
    OUTPUT = Path(__file__).resolve().parent / 'data_quality_figures'
OUTPUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'font.size':10, 'axes.titlesize':12, 'axes.labelsize':10,
                     'savefig.dpi':300, 'figure.dpi':140, 'axes.spines.top':False,
                     'axes.spines.right':False})

def save(fig, stem):
    fig.savefig(OUTPUT / (stem + '.png'), dpi=300, bbox_inches='tight')
    fig.savefig(OUTPUT / (stem + '.pdf'), bbox_inches='tight')
    plt.close(fig)

station = pd.read_csv(INPUT / 'station_coverage.csv')
gaps = pd.read_csv(INPUT / 'annual_gap_distribution.csv')
monthly = pd.read_csv(INPUT / 'monthly_gap_counts.csv')
years = sorted(station['year'].unique())

# Figure 1: station coverage by year (within observed station span)
fig, ax = plt.subplots(figsize=(9,4.8))
values = [100 * station.loc[station.year.eq(y), 'coverage_span_approx'].dropna().to_numpy() for y in years]
ax.boxplot(values, tick_labels=[str(y) for y in years], patch_artist=True,
           showfliers=True, widths=.58,
           boxprops={'facecolor':'#bcd4e6','edgecolor':'#304d66'},
           medianprops={'color':'#9c3b30','linewidth':1.8},
           whiskerprops={'color':'#304d66'}, capprops={'color':'#304d66'},
           flierprops={'marker':'.','markersize':4,'alpha':.45,'markerfacecolor':'#304d66','markeredgecolor':'#304d66'})
ax.set(ylabel='Station coverage within observed span (%)', xlabel='Year',
       title='Station-level temporal coverage, 2017–2024')
ax.set_ylim(0,104)
ax.grid(axis='y', alpha=.22)
fig.tight_layout(); save(fig,'figure_1_station_coverage')

# Figure 2: conditional distribution of gaps >5min
classes=['5_to_10min','10_to_30min','30_to_60min','1_to_24h','1_to_7d','over_7d']
labels=['5–10 min','10–30 min','30–60 min','1–24 h','1–7 days','>7 days']
pivot = gaps.pivot_table(index='year',columns='gap_class',values='intervals',aggfunc='sum',fill_value=0).reindex(index=years,columns=classes,fill_value=0)
pct=pivot.div(pivot.sum(axis=1),axis=0)*100
fig,ax=plt.subplots(figsize=(10,5.0))
colors=plt.get_cmap('Blues')(np.linspace(.30,.90,len(classes)))
bottom=np.zeros(len(years))
for i,(col,label) in enumerate(zip(classes,labels)):
    ax.bar([str(y) for y in years],pct[col].values,bottom=bottom,label=label,color=colors[i],width=.72)
    bottom+=pct[col].values
ax.set(ylabel='Share of intervals exceeding 5 minutes (%)',xlabel='Year',
       title='Duration composition of irregular observation intervals')
ax.set_ylim(0,100); ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.16),frameon=False)
fig.tight_layout(); save(fig,'figure_2_gap_duration_distribution')

# Figure 3: monthly heatmap (log transform to make smaller years visible)
mat=monthly.pivot_table(index='year',columns='month',values='gaps_over_5min',aggfunc='sum',fill_value=0).reindex(index=years,columns=range(1,13),fill_value=0)
fig,ax=plt.subplots(figsize=(10,4.9))
from matplotlib.colors import LogNorm
im=ax.imshow(mat.to_numpy(),aspect='auto',cmap='YlOrRd',norm=LogNorm(vmin=1,vmax=max(1,mat.to_numpy().max())))
ax.set_xticks(range(12),['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
ax.set_yticks(range(len(years)),[str(y) for y in years]); ax.set(xlabel='Month',ylabel='Year',title='Monthly frequency of observation gaps exceeding 5 minutes')
cb=fig.colorbar(im,ax=ax,pad=.02);cb.set_label('Number of irregular intervals (log scale)')
fig.tight_layout();save(fig,'figure_3_monthly_gap_heatmap')

# Figure 4: two panels comparing counts vs implied missing readings
agg=gaps.groupby('year',as_index=True).agg({'estimated_missing_slots':'sum'})
counts=pivot.sum(axis=1)
fig,axes=plt.subplots(1,2,figsize=(11,4.4))
axes[0].bar([str(y) for y in years],counts.reindex(years).values,color='#477a9c')
axes[0].set(title='A. Intervals exceeding 5 minutes',ylabel='Number of intervals',xlabel='Year')
axes[1].bar([str(y) for y in years],agg.reindex(years)['estimated_missing_slots'].values,color='#c17c50')
axes[1].set(title='B. Estimated missing 5-minute readings',ylabel='Estimated readings',xlabel='Year')
for ax in axes:
    ax.tick_params(axis='x',rotation=45)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x/1000:,.0f}k' if x>=1000 else f'{x:,.0f}'))
    ax.grid(axis='y',alpha=.18)
fig.suptitle('Gap frequency and estimated missing observations are not equivalent',y=1.02)
fig.tight_layout();save(fig,'figure_4_gap_counts_vs_missing')
print('Saved four figures (PNG and PDF) to:',OUTPUT)
