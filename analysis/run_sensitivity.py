from pathlib import Path
import json, warnings
import numpy as np,pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.duration.hazard_regression import PHReg
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
warnings.filterwarnings('ignore',category=FutureWarning)
ROOT=Path(__file__).parent;OUT=ROOT/'results'
met=pd.read_csv(ROOT/'data/clinical_extraction/GSE161537_metadata_all.csv',dtype={'patient_id':str}).set_index('patient_id')
expr=pd.read_csv(ROOT/'data/raw/GSE161537_nivobio_log2cpm.csv.gz',sep=';',decimal=',',index_col=0)
met=met.loc[expr.columns.astype(str)];response=met.best_response_on_immunotherapy_recist.map({'CR':1,'PR':1,'SD':0,'PD':0});good=response.notna()
y=response.loc[good].to_numpy(dtype=int)
shared=json.load(open(OUT/'cohort_and_gene_audit.json'))['common_PhaSePro_response_genes']
ranking=(expr.loc[shared,met.index[good]].T.rank(axis=1,method='average',pct=True)-0.5).to_numpy()
hot=met.loc[good,'hot_score'].to_numpy(dtype=float).reshape(-1,1)
assert np.isfinite(hot).all()
models={'HOT_only':hot,'LLPS_22':ranking,'HOT_plus_LLPS':np.column_stack((hot,ranking))}
outs=[];details=[]
for name,X in models.items():
    aucs=[]
    for rep in range(3):
        outcv=StratifiedKFold(n_splits=5,shuffle=True,random_state=260112+rep)
        pred=np.full(len(y),np.nan)
        for fold,(train,test) in enumerate(outcv.split(X,y)):
            pipe=Pipeline([('scaler',StandardScaler()),('select',SelectKBest(f_classif)),('clf',LogisticRegression(solver='liblinear',max_iter=5000))])
            ks=[1] if X.shape[1]==1 else sorted(set([3,5,10,X.shape[1]]))
            grid={'select__k':ks,'clf__C':[0.03,0.3,3]}
            gs=GridSearchCV(pipe,grid,cv=StratifiedKFold(n_splits=3,shuffle=True,random_state=260112+rep*100+fold),scoring='roc_auc',n_jobs=1)
            gs.fit(X[train],y[train]);pred[test]=gs.predict_proba(X[test])[:,1]
        aucs.append(float(roc_auc_score(y,pred)))
    outs.append({'model':name,'N':len(y),'mean_nested_AUC':np.mean(aucs),'min_nested_AUC':min(aucs),'max_nested_AUC':max(aucs),'cv1':aucs[0],'cv2':aucs[1],'cv3':aucs[2]})
df=pd.DataFrame(outs);df.to_csv(OUT/'sensitivity_HOT_comparison.csv',index=False);print(df.to_string(index=False))
# Verify PFS associations using only the subset traceable to GSE190265 GEO sample titles
raw3=pd.read_csv(ROOT/'data/raw/GSE190265_TPM_France3.csv.gz',sep=';',index_col=0)
info=pd.read_csv(ROOT/'data/raw/GSE190265_samples_info_France3.csv.gz',sep=';',dtype={'sample':str}).set_index('sample')
geo=pd.read_csv(ROOT/'data/clinical_extraction/GSE190265_metadata_all.csv')
linked=sorted(set(geo.title)&set(raw3.index.astype(str))&set(info.index))
for gene in ['MED1','MORC3','ESR1','LAT']:
 if gene not in raw3.columns:continue
 for subset,samples in [('all_43',info.index),('GEO_traceable_subset',linked)]:
  x=np.log2(raw3.loc[samples,gene].astype(float)+1)
  z=(x-x.mean())/x.std(ddof=0)
  t=info.loc[samples,'time_PFS'].clip(upper=6).astype(float)
  e=((info.loc[samples,'evtPFS']==1)&(info.loc[samples,'time_PFS']<=6)).astype(int)
  fit=PHReg(t.values,z.values.reshape(-1,1),status=e.values,ties='efron').fit(disp=0)
  details.append({'gene':gene,'subset':subset,'N':len(samples),'events':int(e.sum()),'HR':float(np.exp(fit.params[0])),'p':float(fit.pvalues[0])})
pd.DataFrame(details).to_csv(OUT/'sensitivity_GSE190265_GEO_linked.csv',index=False)
print(pd.DataFrame(details).to_string(index=False))
