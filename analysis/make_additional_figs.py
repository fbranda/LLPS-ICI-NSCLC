from pathlib import Path
import pandas as pd,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).parent;R=P/'results';plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
a=pd.read_csv(R/'sensitivity_HOT_comparison.csv')
fig,ax=plt.subplots(figsize=(7.8,4.0))
labels=['Published HOT score','LLPS (22 genes)','HOT + LLPS'];yy=[2,1,0]
for i,(_,row) in enumerate(a.iterrows()):
 y=2-i;ax.scatter(row.mean_nested_AUC,y,s=80,color=['#27728b','#b46740','#6c7581'][i],zorder=3)
 ax.plot([row.min_nested_AUC,row.max_nested_AUC],[y,y],linewidth=4,color=['#27728b','#b46740','#6c7581'][i],alpha=.65)
 ax.text(row.max_nested_AUC+.009,y,f'{row.mean_nested_AUC:.3f}',va='center',fontsize=10)
ax.axvline(.5,ls='--',lw=1,color='gray');ax.set_xlim(.22,.77);ax.set_yticks(yy);ax.set_yticklabels(labels);ax.set_xlabel('Nested CV AUROC, mean and range across 3 splits');ax.set_title('Incremental value against the published HOT score (n=76)');fig.tight_layout();fig.savefig(R/'Figure4_HOT_vs_LLPS.png',dpi=240);plt.close(fig)

# LAT cohort-wise rank distributions plotted separately to avoid mixing expression scales
b=pd.read_csv(P/'data/clinical_extraction/GSE161537_metadata_all.csv',dtype={'patient_id':str}).set_index('patient_id')
e=pd.read_csv(P/'data/raw/GSE161537_nivobio_log2cpm.csv.gz',sep=';',decimal=',',index_col=0)
b=b.loc[e.columns.astype(str)];y=b.best_response_on_immunotherapy_recist.map({'CR':1,'PR':1,'SD':0,'PD':0});ok=y.notna()
xx=e.loc['LAT',b.index[ok]].astype(float)
meta=pd.read_csv(P/'data/clinical_extraction/GSE126044_metadata_all.csv');meta['sid']=meta.title.str.extract(r'(Dis_\d+)')[0];meta=meta.set_index('sid')
count=pd.read_csv(P/'data/raw/GSE126044_counts.txt.gz',sep='\t',index_col=0)
vals=np.log2((count.loc['LAT']/count.sum(axis=0)*1e6)+1)
y2=meta.loc[vals.index].patient_response.map({'responder':1,'non-responder':0})
fig,axs=plt.subplots(1,2,figsize=(8.5,4.2))
for ax,data,labels,ttl in [(axs[0],xx,y[ok], 'GSE161537 (targeted log2CPM)'),(axs[1],vals,y2,'GSE126044 (RNA-seq log2CPM)')]:
 rng=np.random.default_rng(42)
 for k in [0,1]:
  v=data[labels==k].values
  ax.scatter(np.full(len(v),k)+rng.uniform(-.17,.17,len(v)),v,s=23,alpha=.8,color=['#627e9c','#c0825f'][k]);ax.hlines(np.median(v),k-.18,k+.18,color='#2d3136',linewidth=2.1)
 ax.set_xticks([0,1]);ax.set_xticklabels(['Non-response','Response']);ax.set_ylabel('LAT expression, log2 CPM');ax.set_title(ttl,fontsize=10)
fig.suptitle('LAT: nominal associations are exploratory (no FDR significance)',y=1.01,fontsize=11)
fig.tight_layout();fig.savefig(R/'Supplementary_Figure2_LAT_by_cohort.png',dpi=240,bbox_inches='tight');plt.close(fig)
