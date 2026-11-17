#!/usr/bin/env python3
"""Post-review analyses; original public GEO matrices, no fabricated values.
Sensitivity and comparator calculations are exploratory and reported as such.
"""
from pathlib import Path
from collections import Counter
from itertools import combinations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, norm, mannwhitneyu
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

R=Path(__file__).resolve().parent; ROOT=R.parent; OUT=R/'corrected_results'; OUT.mkdir(exist_ok=True)
O=R/'results'
seed=111326
nivo=pd.read_csv(ROOT/'data/raw/GSE161537_nivobio_log2cpm.csv.gz',sep=';',decimal=',',index_col=0)
meta=pd.read_csv(ROOT/'data/clinical_extraction/GSE161537_metadata_all.csv',dtype={'patient_id':str}).set_index('patient_id')
meta=meta.loc[nivo.columns.astype(str)]
yall=meta.best_response_on_immunotherapy_recist.map({'CR':1,'PR':1,'SD':0,'PD':0})
ok=yall.notna(); samples=yall.index[ok];y=yall[ok].astype(int).values
hot=meta.loc[samples,'hot_score'].to_numpy(float)
G=set((ROOT/'gene_sets/PhaSePro_human.txt').read_text().splitlines())
respgenes=json.load(open(O/'response_machine_learning.json'))['shared_genes']
assert len(respgenes)==22
Xll=(nivo.loc[respgenes,samples].T.rank(axis=1,pct=True,method='average')-.5).values

# A. External exact conditional label-permutation test (conditional on fixed locked model, all choose(16,5) allocations).
ep=pd.read_csv(O/'response_external_predictions.csv')
ey=ep.response.to_numpy(int);es=ep.model_probability.to_numpy(float)
obs=roc_auc_score(ey,es)
aucs=[]
for c in combinations(range(len(ey)),int(ey.sum())):
 yy=np.zeros(len(ey),dtype=int);yy[list(c)]=1
 aucs.append(roc_auc_score(yy,es))
aucs=np.array(aucs)
p_upper=float(np.mean(aucs>=obs-1e-12)); p_two=float(np.mean(abs(aucs-.5)>=abs(obs-.5)-1e-12))
ext_result={'n':len(ey),'responders':int(ey.sum()),'AUROC':float(obs),'n_exact_label_assignments':len(aucs),'one_sided_label_permutation_p':p_upper,'two_sided_label_permutation_p':p_two,'test_conditions':'Fixed locked scores; response-label exchangeability within independent external cohort; does not correct model selection across analyses.'}

# B. LAT correlation with a T-cell cytotoxic signature, and associations before and after adjustment.
markers=[g for g in ['CD3D','CD8A','PTPRC','GZMB','IFNG','CXCL9','CXCL10'] if g in nivo.index]
Z=nivo.loc[markers,samples].T
z=(Z-Z.mean())/Z.std(ddof=0).replace(0,1)
tscore=z.mean(axis=1).to_numpy(float)
lat=nivo.loc['LAT',samples].values.astype(float)
latz=(lat-lat.mean())/lat.std(ddof=0)
hotz=(hot-hot.mean())/hot.std(ddof=0)
latcor={'Tcell_gene_panel':markers,'LAT_Tcell_spearman_rho':float(spearmanr(lat,tscore).statistic),'LAT_Tcell_spearman_p':float(spearmanr(lat,tscore).pvalue),'LAT_HOT_spearman_rho':float(spearmanr(lat,hot).statistic),'LAT_HOT_spearman_p':float(spearmanr(lat,hot).pvalue)}
latmodels=[]
for name,df in [('LAT_only',pd.DataFrame({'LAT':latz})),('LAT_plus_Tcell',pd.DataFrame({'LAT':latz,'Tcell':tscore})),('LAT_plus_HOT',pd.DataFrame({'LAT':latz,'HOT':hotz})),('LAT_plus_Tcell_and_HOT',pd.DataFrame({'LAT':latz,'Tcell':tscore,'HOT':hotz}))]:
 try:
  fit=sm.GLM(y,sm.add_constant(df,has_constant='add'),family=sm.families.Binomial()).fit()
  latmodels.append({'model':name,'LAT_logOR':float(fit.params['LAT']),'LAT_OR':float(np.exp(fit.params['LAT'])),'LAT_p':float(fit.pvalues['LAT']),'LAT_OR_95CI_lower':float(np.exp(fit.conf_int().loc['LAT',0])),'LAT_OR_95CI_upper':float(np.exp(fit.conf_int().loc['LAT',1])),'n':len(y),'exploratory_posthoc':True})
 except Exception as e:latmodels.append({'model':name,'error':str(e)})
pd.DataFrame(latmodels).to_csv(OUT/'LAT_immune_adjustment_exploratory.csv',index=False)

# C. Matched design for HOT only and HOT + LLPS, with HOT forced in the joint model.
def auc_cv(mtype, X, repeat):
 outer=StratifiedKFold(n_splits=5,shuffle=True,random_state=260112+repeat)
 ps=np.zeros(len(y));bf=[]
 for fold,(tr,te) in enumerate(outer.split(X,y)):
  if mtype=='HOT_only':
   estimator=Pipeline([('scale',StandardScaler()),('lr',LogisticRegression(solver='liblinear',max_iter=2000,random_state=seed))])
   grid={'lr__C':[.03,.3,3]}
  elif mtype=='HOT_plus_LLPS':
   pre=ColumnTransformer([('hot',StandardScaler(),[0]),('llps',Pipeline([('z',StandardScaler()),('sel',SelectKBest(f_classif))]),list(range(1,23)))])
   estimator=Pipeline([('features',pre),('lr',LogisticRegression(solver='liblinear',max_iter=2000,random_state=seed))])
   grid={'features__llps__sel__k':[3,5,10,22],'lr__C':[.03,.3,3]}
  else: raise ValueError(mtype)
  search=GridSearchCV(estimator,grid,scoring='roc_auc',cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=260112+repeat*100+fold),n_jobs=1)
  search.fit(X[tr],y[tr]);ps[te]=search.predict_proba(X[te])[:,1]
  bf.append({'repeat':repeat,'fold':fold,'model':mtype,'best_params':search.best_params_})
 return float(roc_auc_score(y,ps)), float(brier_score_loss(y,ps)),ps,bf
hotx=hot[:,None];joint=np.column_stack([hot,Xll])
cvrows=[];pdraw=[];grids=[]
for rep in range(3):
 for label,mat in [('HOT_only',hotx),('HOT_plus_LLPS',joint)]:
  auc,brier,pred,gs=auc_cv(label,mat,rep);cvrows.append({'model':label,'repeat':rep,'AUROC':auc,'Brier':brier});grids.extend(gs)
  for i,v in enumerate(pred):pdraw.append({'sample':samples[i],'y':int(y[i]),'model':label,'repeat':rep,'p_oof':float(v)})
pd.DataFrame(cvrows).to_csv(OUT/'HOT_forced_nested_cv.csv',index=False)
pd.DataFrame(pdraw).to_csv(OUT/'HOT_forced_nested_predictions.csv',index=False)
base_brier=float(y.mean()*(1-y.mean()))

# D. LLPS-versus-random gene panel reference under fixed, identically applied 5-fold model.
# Permutes panel membership without outcome knowledge. Exploratory, not an exact null hypothesis test.
avail=[g for g in nivo.index.astype(str) if g not in G and nivo.loc[g,samples].std()>1e-7 and not pd.isna(nivo.loc[g,samples]).any()]
meanstd=np.array([(nivo.loc[g,samples].mean(),nivo.loc[g,samples].std()) for g in avail])
mu=np.array([(nivo.loc[g,samples].mean(),nivo.loc[g,samples].std()) for g in respgenes])
scale=np.std(meanstd,axis=0);scale[scale==0]=1
rng=np.random.default_rng(seed)
knn=[]
for v in mu:
  d=np.sum(((meanstd-v)/scale)**2,axis=1)
  nearest=np.argsort(d)[:min(70,len(d))]
  knn.append(nearest)

def fixed_cv(mat):
 outer=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)
 p=np.zeros(len(y))
 for tr,te in outer.split(mat,y):
  clf=Pipeline([('z',StandardScaler()),('lr',LogisticRegression(C=.3,solver='liblinear',max_iter=2000))])
  clf.fit(mat[tr],y[tr]);p[te]=clf.predict_proba(mat[te])[:,1]
 return roc_auc_score(y,p)
fixed_llps=float(fixed_cv(Xll));refvals=[]
for i in range(200):
 chosen=set(); names=[]
 for positions in knn:
  candidates=[j for j in positions if j not in chosen]
  if not candidates: candidates=[j for j in range(len(avail)) if j not in chosen]
  idx=int(rng.choice(candidates)); chosen.add(idx);names.append(avail[idx])
 Xg=(nivo.loc[names,samples].T.rank(axis=1,pct=True,method='average')-.5).values
 refvals.append({'iteration':i,'AUROC':float(fixed_cv(Xg))})
pd.DataFrame(refvals).to_csv(OUT/'matched_random_panels_200.csv',index=False)
randaucs=np.array([v['AUROC'] for v in refvals]);rand_summary={'LLPS_fixed_C_point3_5fold_AUROC':fixed_llps,'random_matched_200_median_AUROC':float(np.median(randaucs)),'random_matched_200_q05_q95':list(map(float,np.quantile(randaucs,[.05,.95]))),'fraction_random_panels_AUROC_at_least_LLPS':float(np.mean(randaucs>=fixed_llps)), 'purpose':'Exploratory assay-specific specificity benchmark, not formal evidence of no biological association.'}

# E. Fixed six-month PFS cohort analysis restricted to exact GEO-linked France3 sample titles.
f3=pd.read_csv(ROOT/'data/raw/GSE190265_TPM_France3.csv.gz',sep=';',index_col=0)
f4=pd.read_csv(ROOT/'data/raw/GSE190266_TPM_France4.csv.gz',sep=';',decimal=',',index_col=0)
info3=pd.read_csv(ROOT/'data/raw/GSE190265_samples_info_France3.csv.gz',sep=';',dtype={'sample':str}).set_index('sample')
geo3=pd.read_csv(ROOT/'data/clinical_extraction/GSE190265_metadata_all.csv')
meta4=pd.read_csv(ROOT/'data/clinical_extraction/GSE190266_metadata_all.csv').set_index('title')
linked3=sorted(set(geo3.title.astype(str)) & set(info3.index) & set(f3.index.astype(str)))
assert len(linked3)==26
assert set(f4.index.astype(str))==set(meta4.index.astype(str))
common=sorted(G & set(f3.columns) & set(f4.columns))
ref=['MGAT5B','E2F7','IGF2BP1','F2','POPDC3','GRIA1'];mlors_common=sorted(set(ref)&set(f3.columns)&set(f4.columns))

def fit_one(data, info, gen, cohort):
 x=np.log2(data[gen].astype(float)+1)
 x=(x-x.mean())/x.std(ddof=0)
 if cohort=='France3':
  t=info['time_PFS'].astype(float)
  e=info['evtPFS'].astype(int)
 else:
  t=info['pfs_time_6_months'].astype(float)
  e=info['pfs_evt_6_months'].astype(int)
 time=np.minimum(t.to_numpy(float),6)
 ev=((e.values==1)&(t.values<=6)).astype(int)
 try:
  m=PHReg(time,x.to_numpy().reshape(-1,1),status=ev,ties='efron').fit(disp=0)
  beta=float(m.params[0]);se=float(m.bse[0]);return {'gene':gen,'cohort':cohort,'N':len(x),'events':int(ev.sum()),'beta':beta,'SE':se,'HR':float(np.exp(beta)),'p':float(m.pvalues[0])}
 except: return None

def meta_table(genes):
 all_rows=[];pooled=[]
 for g in genes:
  a=fit_one(f3.loc[linked3],info3.loc[linked3],g,'France3')
  b=fit_one(f4.loc[meta4.index],meta4,g,'France4')
  if a is None or b is None:continue
  all_rows.extend([a,b]);weights=1/np.square([a['SE'],b['SE']]);effects=np.array([a['beta'],b['beta']]);mu=float(np.sum(weights*effects)/weights.sum());se=float(np.sqrt(1/weights.sum()))
  pv=2*norm.sf(abs(mu/se));q=weights.sum()*(np.var(effects) if False else 0)
  Q=float(np.sum(weights*(effects-mu)**2));I2=max(0.,(Q-1)/Q) if Q>0 else 0
  pooled.append({'gene':g,'N_France3_GEO_linked':26,'N_France4':len(f4),'PFS_6m_events_total':sum(x['events'] for x in [a,b]),'HR_pooled':float(np.exp(mu)),'HR95_low':float(np.exp(mu-1.96*se)),'HR95_high':float(np.exp(mu+1.96*se)),'p':float(pv),'Q':Q,'I2':I2,'HR_France3':a['HR'],'HR_France4':b['HR']})
 out=pd.DataFrame(pooled)
 if len(out):out['BH_FDR_within_family']=multipletests(out.p,method='fdr_bh')[1]
 return out,pd.DataFrame(all_rows)
pf,rows=meta_table(common);pf.sort_values('p').to_csv(OUT/'PFS_GEO_linked_26_plus_70_genes.csv',index=False);rows.to_csv(OUT/'PFS_GEO_linked_26_plus_70_per_cohort.csv',index=False)
pl,lr=meta_table(mlors_common);pl.sort_values('p').to_csv(OUT/'PFS_MLORS_reference_exploratory.csv',index=False)

# F. independent external response reference genes (not construction of Yan risk score).
count=pd.read_csv(ROOT/'data/raw/GSE126044_counts.txt.gz',sep='\t',index_col=0)
count=count.groupby(level=0).sum();meta126=pd.read_csv(ROOT/'data/clinical_extraction/GSE126044_metadata_all.csv');meta126['sid']=meta126.title.str.extract(r'(Dis_\d+)')[0];meta126=meta126.set_index('sid').loc[count.columns]
response=meta126.patient_response.str.lower().map({'responder':1,'non-responder':0}).values
ll=np.log2(count.div(count.sum(axis=0),axis=1)*1e6+1)
rr=[]
for gene in ref:
 if gene not in ll.index:continue
 vals=ll.loc[gene].to_numpy(float)
 p=float(mannwhitneyu(vals[response==1],vals[response==0],alternative='two-sided').pvalue)
 rr.append({'gene':gene,'AUROC':float(roc_auc_score(response,vals)),'MW_p':p,'N':len(response),'responders':int(response.sum())})
rdf=pd.DataFrame(rr);rdf['FDR_six_gene_family']=multipletests(rdf.MW_p,method='fdr_bh')[1];rdf.to_csv(OUT/'MLORS_genes_GSE126044_exploratory.csv',index=False)

# G. scope and multiplicity (post hoc global BH for screens only, same underlying observations not independent).
orig=pd.read_csv(O/'response_univariate_bh.csv')
main=orig[orig.cohort.isin(['GSE161537_all_NSCLC','GSE126044_NSCLC'])].copy();main['FDR_across_two_response_screens']=multipletests(main.MW_p,method='fdr_bh')[1];main.to_csv(OUT/'response_main_global_BH.csv',index=False)

# H. diagnostic plots
plt.rcParams.update({'font.size':9,'axes.spines.right':False,'axes.spines.top':False})
fig,ax=plt.subplots(figsize=(7.3,4.1))
for name,col in [('HOT_only','#087f8c'),('HOT_plus_LLPS','#b45636')]:
 sub=pd.DataFrame(cvrows);sub=sub[sub.model==name];ax.scatter([0 if name=='HOT_only' else 1]*len(sub),sub.AUROC,c=col,s=45,label=name,zorder=4)
 ax.plot([0 if name=='HOT_only' else 1]*len(sub),sub.AUROC,c=col,alpha=.3)
ax.axhline(.5,color='grey',ls='--',lw=1);ax.set_xticks([0,1],['HOT only','HOT + LLPS (HOT forced)']);ax.set_ylim(0,1);ax.set_ylabel('Out-of-fold AUROC');ax.set_title('Matched nested-CV: HOT included in joint model');ax.grid(axis='y',alpha=.18)
fig.tight_layout();fig.savefig(OUT/'FigureA_HOT_forced_comparison.png',dpi=240);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(9,3.5))
axes[0].scatter(tscore,latz,c=y,cmap='coolwarm',s=25,alpha=.8);axes[0].set_xlabel('T-cell/cytotoxic transcript score');axes[0].set_ylabel('LAT expression (standardized)');axes[0].set_title(f"LAT versus immune signal (rho={latcor['LAT_Tcell_spearman_rho']:.2f})")
axes[1].hist(randaucs,bins=20,color='#7b98a5',edgecolor='white');axes[1].axvline(fixed_llps,color='#b45636',lw=2,label=f'LLPS fixed-model {fixed_llps:.2f}');axes[1].axvline(.5,color='grey',ls='--');axes[1].legend(frameon=False,fontsize=8);axes[1].set_xlabel('Fivefold CV AUROC');axes[1].set_title('200 expression-matched random gene panels');fig.tight_layout();fig.savefig(OUT/'FigureB_immune_confounds_and_null_panels.png',dpi=240);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(9,3.7))
fpr,tpr,_=roc_curve(ey,es);axes[0].plot(fpr,tpr,color='#087f8c',lw=2.2,label=f'External AUROC {obs:.3f}');axes[0].plot([0,1],[0,1],color='grey',ls='--');axes[0].set(xlabel='1 - specificity',ylabel='Sensitivity',title='Fixed external scores (n=16)');axes[0].legend(frameon=False,loc='lower right')
axes[1].hist(aucs,bins=np.arange(0,1.02,.05),color='#b2c5ca',edgecolor='white');axes[1].axvline(obs,color='#b45636',lw=2,label=f'observed {obs:.3f}');axes[1].legend(frameon=False);axes[1].set(xlabel='AUROC with permuted labels',ylabel='Number of assignments',title=f'Exact permutations ({len(aucs):,} allocations)');fig.tight_layout();fig.savefig(OUT/'FigureC_external_exact_permutation.png',dpi=240);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4.6))
top=pf.sort_values('p').head(9).sort_values('HR_pooled');pos=np.arange(len(top));hr=top.HR_pooled.to_numpy();lo=top.HR95_low.to_numpy();hi=top.HR95_high.to_numpy();ax.errorbar(hr,pos,xerr=[hr-lo,hi-hr],fmt='o',capsize=2,color='#087f8c');ax.axvline(1,color='grey',ls='--');ax.set_yticks(pos,top.gene);ax.set_xlabel('Fixed-effect HR per within-cohort SD (95% CI)');ax.set_title('Six-month PFS: GEO-linked France3 (n=26) + France4 (n=70)');ax.text(.01,-.16,'Exploratory: incomplete France4 gene coverage; no FDR-confirmed associations.',transform=ax.transAxes,fontsize=8);fig.tight_layout();fig.savefig(OUT/'FigureD_GEO_linked_PFS_forest.png',dpi=240);plt.close(fig)
summary={'external_exact_permutation':ext_result,'LAT_immune_proxy':latcor,'HOT_forced_comparison':cvrows,'prevalence_only_Brier_development':base_brier,'random_panel_assay_benchmark':rand_summary,'GEO_linked_PFS':{'N_France3':len(linked3),'N_France4':len(f4),'tested_genes':len(pf),'lowest_p':float(pf.p.min()),'lowest_FDR':float(pf.BH_FDR_within_family.min()),'MED1_row':pf[pf.gene=='MED1'].to_dict('records')},'reference_MLORS_PFS_genes_tested':mlors_common,'GSE126044_MLORS_six_reference':rr,'response_global_screens':{'number_tests':len(main),'min_global_FDR':float(main.FDR_across_two_response_screens.min())}}
(OUT/'CORRECTIVE_SUMMARY.json').write_text(json.dumps(summary,indent=2,default=float))
print(json.dumps(summary,indent=2)[:9000]);print('Corrective analysis done at',OUT)
