#!/usr/bin/env python3
"""Auditable, outcome-blinded cross-cohort LLPS hypothesis tests on five GEO matrices.
No simulated patient values. For source identifiers and definitions see README.
"""
import json, warnings, re
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, norm, spearmanr
from statsmodels.stats.multitest import multipletests
from statsmodels.duration.hazard_regression import PHReg
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, roc_curve
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

R=Path(__file__).resolve().parent
OUT=R/'results'; OUT.mkdir(exist_ok=True)
G=set((R/'gene_sets/PhaSePro_human.txt').read_text().splitlines())
REF=['MGAT5B','E2F7','IGF2BP1','F2','POPDC3','GRIA1']
SEED=260112

def read_exp(name, sep, decimal='.', transposed=False):
    d=pd.read_csv(R/'data/raw'/name,sep=sep,decimal=decimal,index_col=0)
    if transposed:d=d.T
    d.index=d.index.astype(str);d.columns=d.columns.astype(str)
    if d.index.duplicated().any():
        d=d.groupby(level=0).median()
    return d.astype(float)

# === EXPRESSION, labels, source audits ===
nivo=read_exp('GSE161537_nivobio_log2cpm.csv.gz',';',',')
nivo.index=nivo.index.str.upper()
meta161=pd.read_csv(R/'data/clinical_extraction/GSE161537_metadata_all.csv',dtype={'patient_id':str})
assert meta161.patient_id.notna().all() and meta161.patient_id.is_unique
assert set(meta161.patient_id)==set(nivo.columns)
clin161=meta161.set_index('patient_id').loc[nivo.columns]
assert set(clin161.best_response_on_immunotherapy_recist.dropna())<=set(['CR','PR','SD','PD'])
resp161=clin161.best_response_on_immunotherapy_recist.map({'CR':1,'PR':1,'SD':0,'PD':0})
assert resp161.notna().sum()==76

counts=read_exp('GSE126044_counts.txt.gz','\t')
counts=counts.groupby(level=0).sum()
counts_cpm=counts.div(counts.sum(axis=0),axis=1)*1e6
counts_log=np.log2(counts_cpm+1)
meta126=pd.read_csv(R/'data/clinical_extraction/GSE126044_metadata_all.csv')
meta126['matrix_id']=meta126.title.str.extract(r'(Dis_\d+)')[0]
assert meta126.matrix_id.is_unique and set(meta126.matrix_id)==set(counts.columns)
meta126=meta126.set_index('matrix_id').loc[counts.columns]
resp126=meta126.patient_response.str.lower().map({'responder':1,'non-responder':0})
assert resp126.notna().sum()==16

f3=read_exp('GSE190265_TPM_France3.csv.gz',';',transposed=True)
f4=read_exp('GSE190266_TPM_France4.csv.gz',';',',',transposed=True)
f3.index=f3.index.str.upper();f4.index=f4.index.str.upper()
f3_log=np.log2(f3+1);f4_log=np.log2(f4+1)
info3=pd.read_csv(R/'data/raw/GSE190265_samples_info_France3.csv.gz',sep=';',dtype={'sample':str})
assert info3['sample'].is_unique and set(info3['sample'])==set(f3.columns)
meta4=pd.read_csv(R/'data/clinical_extraction/GSE190266_metadata_all.csv')
assert meta4.title.is_unique and set(meta4.title)==set(f4.columns)
meta3=pd.read_csv(R/'data/clinical_extraction/GSE190265_metadata_all.csv')
n_f3_linked_to_geo=len(set(meta3.title)&set(info3['sample']))
assert n_f3_linked_to_geo >= 20, 'GSE190265 provenance linkage insufficient'

outcome3=info3.set_index('sample').loc[f3.columns]
outcome4=meta4.set_index('title').loc[f4.columns]
t3=outcome3.time_PFS.astype(float).clip(upper=6.0)
e3=((outcome3.evtPFS==1)&(outcome3.time_PFS<=6.0)).astype(int)
t4=outcome4.pfs_time_6_months.astype(float)
e4=outcome4.pfs_evt_6_months.astype(int)
assert t4.between(0.001,6.01).all() and (t3>0).all()
assert set(e4.dropna().unique())<=set([0,1])

meta135=pd.read_csv(R/'data/clinical_extraction/GSE135222_metadata_all.csv')
ensg=read_exp('GSE135222_GEO_RNA-seq_omicslab_exp.tsv.gz','\t')
assert set(meta135.title.str.replace(' ','',regex=False))==set(ensg.columns)
# Dataset GSE135222 is kept for PFS audit, not tested for gene effects: requires authoritative ENSG -> HGNC map.

shared_response=sorted(G.intersection(nivo.index,counts_log.index))
shared_pfs=sorted(G.intersection(f3_log.index,f4_log.index))

audit={
 'PhaSePro_source_records_human_symbols':len(G),
 'GSE161537':{'matrix_genes':len(nivo),'matrix_samples':len(nivo.columns),'meta_samples':len(meta161),'RECIST_labeled':int(resp161.notna().sum()),'responders':int(resp161.sum()),'histology_AD_samples':int((clin161.histology=='AD').sum()),'AD_response_labeled':int(((clin161.histology=='AD')&resp161.notna()).sum()),'PhaSePro_genes_measured':len(G.intersection(nivo.index)),'MLORS_genes_measured':sorted(set(REF).intersection(nivo.index))},
 'GSE126044':{'matrix_genes':len(counts_log),'matrix_samples':len(counts_log.columns),'meta_samples':len(meta126),'response_labeled':int(resp126.notna().sum()),'responders':int(resp126.sum()),'PhaSePro_genes_measured':len(G.intersection(counts_log.index)),'MLORS_genes_measured':sorted(set(REF).intersection(counts_log.index))},
 'GSE190265':{'expression_genes':len(f3_log),'expression_samples_with_supplemental_PFS':len(info3),'GEO_annotated_samples':len(meta3),'GEO_matched_with_expression':n_f3_linked_to_geo,'PFS_with_unlinked_GEO_accession':len(info3)-n_f3_linked_to_geo,'six_month_events':int(e3.sum()),'samples_censored_before_6months':int(((e3==0)&(t3<6)).sum()),'PhaSePro_genes_measured':len(G.intersection(f3_log.index))},
 'GSE190266':{'expression_genes':len(f4_log),'matrix_samples':len(f4_log.columns),'GEO_annotated_samples':len(meta4),'six_month_events':int(e4.sum()),'samples_censored_before_6months':int(((e4==0)&(t4<6)).sum()),'PhaSePro_genes_measured':len(G.intersection(f4_log.index)),'last_gene':f4_log.index[-1],'max_feature_count_matches_Excel_columns':len(f4_log)==16383},
 'GSE135222':{'matrix_ensembl_genes':len(ensg),'samples':len(ensg.columns),'PFS_events':int(meta135.progression_free_survival_pfs.sum()),'status':'PFS available; gene mapping missing, no candidate testing'},
 'common_PhaSePro_response_genes':shared_response,'common_PhaSePro_PFS_genes':shared_pfs,
 'source_expression_overlap_with_MLORS':{k: sorted(set(REF)&set(v)) for k,v in [('GSE190265',f3.index),('GSE190266',f4.index)]}
}
(OUT/'cohort_and_gene_audit.json').write_text(json.dumps(audit,indent=2))

# === RESPONSE UNIVARIATE ===
rs=[]
for cohort,expr,clinical,mask in [
 ('GSE161537_all_NSCLC',nivo,resp161,resp161.notna()),
 ('GSE161537_AD_only',nivo,resp161,(resp161.notna())&(clin161.histology=='AD')),
 ('GSE126044_NSCLC',counts_log,resp126,resp126.notna())]:
    for gene in sorted(G & set(expr.index)):
        x=expr.loc[gene,mask.index[mask]].astype(float).values
        y=clinical.loc[mask.index[mask]].astype(int).values
        if len(set(x))<3:continue
        r=x[y==1];n=x[y==0]
        p=mannwhitneyu(r,n,alternative='two-sided').pvalue
        au=roc_auc_score(y,x)
        rs.append(dict(cohort=cohort,gene=gene,n=len(x),n_responder=len(r),n_nonresponder=len(n),median_responder=float(np.median(r)),median_nonresponder=float(np.median(n)),median_difference=float(np.median(r)-np.median(n)),AUC_single_gene=float(au),MW_p=float(p)))
resp_uni=pd.DataFrame(rs)
resp_uni['BH_FDR']=np.nan
for name,idx in resp_uni.groupby('cohort').groups.items():
    resp_uni.loc[idx,'BH_FDR']=multipletests(resp_uni.loc[idx,'MW_p'],method='fdr_bh')[1]
resp_uni.sort_values(['cohort','MW_p']).to_csv(OUT/'response_univariate_bh.csv',index=False)

# === RESPONSE SUPERVISED: sample-wise relative ranks for cross-platform feature compatibility ===
# Individual patient rank among 21 shared LLPS genes (NOT standardising on validation labels).
# Inference is rank-based and does not require a cohort-wise reference distribution.
shared_response=[g for g in shared_response if nivo.loc[g].std()>0 and counts_log.loc[g].std()>0]
def patient_rank_features(mat, samples, genes):
    sub=mat.loc[genes,samples].T
    return (sub.rank(axis=1,method='average',pct=True)-0.5).to_numpy(dtype=float)
X=patient_rank_features(nivo,resp161.index[resp161.notna()],shared_response)
y=resp161.dropna().to_numpy(dtype=int)
Xt=patient_rank_features(counts_log,resp126.index,shared_response)
yt=resp126.to_numpy(dtype=int)

# 2-stage nested cross-validation; all supervised steps are refit within each training fold.
pipe=Pipeline([('scale',StandardScaler()),('select',SelectKBest(f_classif)),('model',LogisticRegression(penalty='l2',solver='liblinear',max_iter=5000,random_state=SEED))])
grid={'select__k':sorted(set([3,5,10,len(shared_response)])), 'model__C':[0.03,0.3,3]}

# Full nested outer 5 folds, repeat 3 times for assessment of split instability.
preds=[];bp=[];coefs=[]
for repeat in range(3):
    outer=StratifiedKFold(n_splits=5,shuffle=True,random_state=SEED+repeat)
    pred=np.full(len(y),np.nan)
    for fold,(train,test) in enumerate(outer.split(X,y)):
        inner=StratifiedKFold(n_splits=3,shuffle=True,random_state=SEED+repeat*100+fold)
        gs=GridSearchCV(clone(pipe),grid,scoring='roc_auc',cv=inner,refit=True,n_jobs=1)
        gs.fit(X[train],y[train]);pred[test]=gs.predict_proba(X[test])[:,1]
        bp.append({'repeat':repeat,'fold':fold,'test_n':len(test),'params':str(gs.best_params_)})
        selected=gs.best_estimator_.named_steps['select'].get_support()
        for g in np.array(shared_response)[selected]:coefs.append((repeat,fold,g))
    assert np.isfinite(pred).all()
    for i in range(len(y)):
        preds.append({'repeat':repeat,'sample':str(resp161.index[resp161.notna()][i]),'y':int(y[i]),'p_oof':float(pred[i])})

pd.DataFrame(preds).to_csv(OUT/'response_nested_oof_predictions.csv',index=False)
pred_df=pd.DataFrame(preds)
cv_auc=[roc_auc_score(sub.y,sub.p_oof) for _,sub in pred_df.groupby('repeat')]
cv_ap=[average_precision_score(sub.y,sub.p_oof) for _,sub in pred_df.groupby('repeat')]
cv_brier=[brier_score_loss(sub.y,sub.p_oof) for _,sub in pred_df.groupby('repeat')]

# Final model locked on complete dev using internal tuning, tested on independent study.
full=GridSearchCV(clone(pipe),grid,scoring='roc_auc',cv=StratifiedKFold(n_splits=4,shuffle=True,random_state=SEED+1000),refit=True)
full.fit(X,y)
p_external=full.predict_proba(Xt)[:,1]
external_auc=roc_auc_score(yt,p_external)
external_ap=average_precision_score(yt,p_external)
external_brier=brier_score_loss(yt,p_external)
from collections import Counter
sel=Counter(g for _,_,g in coefs)
pd.DataFrame({'gene':shared_response,'selected_outer_folds':[sel[g] for g in shared_response]}).sort_values('selected_outer_folds',ascending=False).to_csv(OUT/'response_feature_selection.csv',index=False)
pd.DataFrame({'matrix_id':Xt.shape[0] and resp126.index,'response':yt,'model_probability':p_external}).to_csv(OUT/'response_external_predictions.csv',index=False)

# Single gene direction: median responder - non-responder in training. report external independently.
gene_cons=[]
for gene in shared_response:
    a=resp_uni[(resp_uni.cohort=='GSE161537_all_NSCLC')&(resp_uni.gene==gene)]
    b=resp_uni[(resp_uni.cohort=='GSE126044_NSCLC')&(resp_uni.gene==gene)]
    if len(a)==len(b)==1:
        ra=a.iloc[0];rb=b.iloc[0]
        gene_cons.append({'gene':gene,'AUC_development':ra.AUC_single_gene,'AUC_external':rb.AUC_single_gene,'p_development':ra.MW_p,'FDR_development':ra.BH_FDR,'p_external':rb.MW_p,'FDR_external':rb.BH_FDR,'direction_concordant':bool(np.sign(ra.median_difference)==np.sign(rb.median_difference))})
pd.DataFrame(gene_cons).to_csv(OUT/'response_gene_replication.csv',index=False)

def bootstrap_auc(y,p,seed=SEED,n_boot=3000):
    rng=np.random.default_rng(seed);vals=[]
    for i in range(n_boot):
        ix=rng.choice(len(y),size=len(y),replace=True)
        if len(np.unique(np.array(y)[ix]))==2:vals.append(roc_auc_score(np.array(y)[ix],np.array(p)[ix]))
    return [float(z) for z in np.quantile(vals,[.025,.975])]
model_summary={'development_n':len(y),'development_response_n':int(y.sum()),'external_n':len(yt),'external_response_n':int(yt.sum()),'num_shared_LLPS_features':len(shared_response),'shared_genes':shared_response,'rank_features':'within-patient percentile ranks among shared genes','nested_cv_aucs_3_repeats':cv_auc,'nested_cv_average_precision_3_repeats':cv_ap,'nested_cv_brier_3_repeats':cv_brier,'mean_nested_AUC':float(np.mean(cv_auc)),'range_nested_AUC':[min(cv_auc),max(cv_auc)],'final_hyperparameters':full.best_params_,'external_AUC':float(external_auc),'external_auc_naive_patient_bootstrap_95CI':bootstrap_auc(yt,p_external),'external_average_precision':float(external_ap),'external_brier':float(external_brier),'selection_stability_folds':15,'MAJOR_LIMITATIONS':['all cohorts treated, no comparative arm','external n=16','different assay platforms','within-sample ranks contain platform biases','selected model on limited events']}
(OUT/'response_machine_learning.json').write_text(json.dumps(model_summary,indent=2))

# === PFS UNIVARIATE COX, 6-month administrative truncation in both cohorts ===
pfs_rows=[]
for cohort,mat,t,e in [('GSE190265',f3_log,t3,e3),('GSE190266',f4_log,t4,e4)]:
    t=t.loc[mat.columns];e=e.loc[mat.columns]
    for gene in shared_pfs:
        x=mat.loc[gene,mat.columns].to_numpy(dtype=float)
        xstd=x.std(ddof=0)
        if xstd<1e-9:continue
        z=(x-x.mean())/xstd
        try:
            result=PHReg(t.to_numpy(dtype=float),z.reshape(-1,1),status=e.to_numpy(dtype=int),ties='efron').fit(disp=0)
            coef=float(result.params[0]);se=float(result.bse[0]);p=float(result.pvalues[0])
            if not np.isfinite(se) or se<=0:continue
            pfs_rows.append(dict(cohort=cohort,gene=gene,n=len(z),events=int(e.sum()),logHR=coef,se=se,HR=np.exp(coef),CI_lo=np.exp(coef-1.96*se),CI_hi=np.exp(coef+1.96*se),p=p,mean_log2TPM=float(x.mean())))
        except Exception as err:
            warnings.warn(f'Cox failed {cohort}:{gene}: {err}')
pfs_uni=pd.DataFrame(pfs_rows);pfs_uni['BH_FDR']=np.nan
for cohort,ids in pfs_uni.groupby('cohort').groups.items():pfs_uni.loc[ids,'BH_FDR']=multipletests(pfs_uni.loc[ids,'p'],method='fdr_bh')[1]
pfs_uni.sort_values(['cohort','p']).to_csv(OUT/'PFS_6month_cox_univariate.csv',index=False)

meta_rows=[]
for gene in shared_pfs:
    z=pfs_uni[pfs_uni.gene==gene]
    if len(z)!=2:continue
    w=1/z.se.to_numpy()**2;means=z.logHR.to_numpy();beta=np.sum(w*means)/sum(w);se=np.sqrt(1/sum(w));pv=2*norm.sf(abs(beta/se))
    q=np.sum(w*(means-beta)**2)
    p_hetero=float(norm.sf(np.sqrt(q))*2) if len(z)==2 else None
    meta_rows.append({'gene':gene,'cohorts':'GSE190265;GSE190266','HR_fixed_effect':np.exp(beta),'CI_lo':np.exp(beta-1.96*se),'CI_hi':np.exp(beta+1.96*se),'p':pv,'direction_concordant':bool(np.sign(means[0])==np.sign(means[1])),'HR_cohort_190265':z.set_index('cohort').loc['GSE190265','HR'],'HR_cohort_190266':z.set_index('cohort').loc['GSE190266','HR'],'Q_heterogeneity':q,'p_heterogeneity_chi2_df1':float(__import__('scipy').stats.chi2.sf(q,1))})
meta=pd.DataFrame(meta_rows)
meta['BH_FDR']=multipletests(meta.p,method='fdr_bh')[1]
meta=meta.sort_values('p')
meta.to_csv(OUT/'PFS_6month_cox_fixed_effect_meta.csv',index=False)

# Average transcriptomic expression score, explicitly not physical LLPS activity.
composite=[]
for cohort,mat,t,e in [('GSE190265',f3_log,t3,e3),('GSE190266',f4_log,t4,e4)]:
    m=mat.loc[shared_pfs]
    z=(m.sub(m.mean(axis=1),axis=0)).div(m.std(axis=1,ddof=0).replace(0,np.nan),axis=0)
    score=z.mean(axis=0).fillna(0)
    score=(score-score.mean())/score.std(ddof=0)
    fit=PHReg(t.loc[score.index].values,score.values.reshape(-1,1),status=e.loc[score.index].values,ties='efron').fit(disp=0)
    b=float(fit.params[0]);s=float(fit.bse[0])
    composite.append({'cohort':cohort,'n':len(score),'events':int(e.sum()),'HR_per_1SD':float(np.exp(b)),'CI_low':float(np.exp(b-1.96*s)),'CI_high':float(np.exp(b+1.96*s)),'p':float(fit.pvalues[0])})
pd.DataFrame(composite).to_csv(OUT/'PFS_LLPS_mean_expression_score.csv',index=False)

# === publication-ready figures ===
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(7.2,5.2))
sub=pred_df[pred_df.repeat==0]
for label,Y,P in [('Development nested CV (repeat 1)',sub.y,sub.p_oof),('External GSE126044 (n=16)',yt,p_external)]:
    fpr,tpr,_=roc_curve(Y,P);a=roc_auc_score(Y,P)
    ax.plot(fpr,tpr,label=f'{label}, AUC={a:.2f}',lw=2)
ax.plot([0,1],[0,1],ls=':',color='gray');ax.set(xlabel='1 - specificity',ylabel='Sensitivity',title='LLPS-associated response model (exploratory)',xlim=(0,1),ylim=(0,1));ax.legend(frameon=False,fontsize=8,loc='lower right');fig.tight_layout();fig.savefig(OUT/'Figure1_response_ROC.png',dpi=240);plt.close(fig)

# FDR plot with points colored by replicated direction, without implying significance.
c=meta.head(25).iloc[::-1]
fig,ax=plt.subplots(figsize=(8,7.2))
ax.scatter(-np.log10(c.p),np.arange(len(c)),s=38,c=np.where(c.direction_concordant,'#2a6d79','#a57142'))
ax.set_yticks(np.arange(len(c)));ax.set_yticklabels(c.gene)
ax.axvline(-np.log10(.05),color='gray',ls='--',lw=.8)
ax.set(xlabel='-log10(fixed-effect meta-analysis p)',title='6-month PFS association across two cohorts')
fig.tight_layout();fig.savefig(OUT/'Figure2_PFS_meta_gene_screen.png',dpi=240);plt.close(fig)

# PFS forest for mean signature.
compo=pd.DataFrame(composite)
fig,ax=plt.subplots(figsize=(7.4,3.6))
for i,r in compo.iterrows():
    ax.errorbar(r.HR_per_1SD,i,xerr=[[r.HR_per_1SD-r.CI_low],[r.CI_high-r.HR_per_1SD]],fmt='o',capsize=4,color='#1b6279')
ax.axvline(1,ls=':',color='gray');ax.set_yticks([0,1]);ax.set_yticklabels(compo.cohort);ax.set_xlabel('Hazard ratio per 1-SD LLPS gene-expression score (95% CI)');ax.set_title('Pre-specified shared-gene mean-score sensitivity analysis');fig.tight_layout();fig.savefig(OUT/'Figure3_PFS_mean_score_forest.png',dpi=240);plt.close(fig)

# Cohort audit plot.
fig,ax=plt.subplots(figsize=(7.7,4.2))
cs=['GSE161537','GSE126044','GSE190265','GSE190266','GSE135222'];sizes=[82,16,43,70,27]
ax.barh(cs[::-1],sizes[::-1],color='#5f8395');ax.set_xlabel('Expression profiles available');ax.set_title('Five public GEO series: molecular data availability')
for i,val in enumerate(sizes[::-1]):ax.text(val+0.5,i,str(val),va='center',fontsize=9)
ax.set_xlim(0,100);fig.tight_layout();fig.savefig(OUT/'Supplementary_Figure1_cohort_inventory.png',dpi=220);plt.close(fig)

result_summary={'audit':audit,'response':model_summary,'response_screen':{'number_tests':{str(k):int(v) for k,v in resp_uni.groupby('cohort').size().items()},'FDR_lt_0_05':{str(k):int((v.BH_FDR<.05).sum()) for k,v in resp_uni.groupby('cohort')},'top_genes':resp_uni.sort_values('MW_p').groupby('cohort').head(5)[['cohort','gene','MW_p','BH_FDR','AUC_single_gene']].to_dict(orient='records'),'direction_concordant':int(pd.DataFrame(gene_cons).direction_concordant.sum()),'number_shared_gene_tests':len(gene_cons)},'PFS':{'cohort_n':[43,70],'events_six_month':[int(e3.sum()),int(e4.sum())],'genes_shared':len(shared_pfs),'FDR_lt_0_05':{str(k):int((v.BH_FDR<.05).sum()) for k,v in pfs_uni.groupby('cohort')},'meta_FDR_lt_0_05':int((meta.BH_FDR<.05).sum()),'top_meta':meta.head(10).to_dict('records'),'mean_score':composite}}
(OUT/'SUMMARY.json').write_text(json.dumps(result_summary,indent=2,default=str))
print(json.dumps({'response_development_N':len(y),'response_external_N':len(yt),'shared_response_genes':len(shared_response),'repeated_nested_CV_AUC':cv_auc,'external_AUC':external_auc,'external_AUC_CI':model_summary['external_auc_naive_patient_bootstrap_95CI'],'PFS_shared_genes':len(shared_pfs),'PFS_meta_FDR_lt_0_05':int((meta.BH_FDR<.05).sum()),'top_PFS_genes':meta.head(5).gene.tolist(),'output':str(OUT)},indent=2))
