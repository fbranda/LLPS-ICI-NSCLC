#!/usr/bin/env python3
"""Secondary non-tuned forest/boosting robustness tests on three repeated fivefold splits."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from sklearn.base import clone
R=Path(__file__).resolve().parent;O=R/'results';j=json.load(open(O/'response_machine_learning.json'))
meta=pd.read_csv(R/'data/clinical_extraction/GSE161537_metadata_all.csv',dtype={'patient_id':str}).set_index('patient_id')
xmat=pd.read_csv(R/'data/raw/GSE161537_nivobio_log2cpm.csv.gz',sep=';',decimal=',',index_col=0)
meta=meta.loc[xmat.columns.astype(str)];resp=meta.best_response_on_immunotherapy_recist.map({'CR':1,'PR':1,'SD':0,'PD':0});ids=resp.index[resp.notna()];y=resp.loc[ids].to_numpy(dtype=int)
cnt=pd.read_csv(R/'data/raw/GSE126044_counts.txt.gz',sep='\t',index_col=0)
met2=pd.read_csv(R/'data/clinical_extraction/GSE126044_metadata_all.csv');met2['matrix_id']=met2.title.str.extract(r'(Dis_\d+)')[0];met2=met2.set_index('matrix_id').loc[cnt.columns];yt=met2.patient_response.map({'responder':1,'non-responder':0}).to_numpy(dtype=int)
mat2=np.log2(cnt.div(cnt.sum(axis=0),axis=1)*1e6+1);gs=j['shared_genes']
X=(xmat.loc[gs,ids].T.rank(axis=1,pct=True)-.5).to_numpy();Xt=(mat2.loc[gs].T.rank(axis=1,pct=True)-.5).to_numpy()
mods={
 'RandomForest_shallow':RandomForestClassifier(n_estimators=50,max_depth=3,min_samples_leaf=3,class_weight='balanced_subsample',random_state=260112,n_jobs=1),
 'GradientBoosting_shallow':GradientBoostingClassifier(n_estimators=40,max_depth=2,learning_rate=.05,random_state=260112)
}
rows=[];predrows=[]
for name,est in mods.items():
 pipe=Pipeline([('feature_selection',SelectKBest(score_func=f_classif,k=10)),('model',est)])
 au=[]
 for rep in range(3):
  folds=StratifiedKFold(n_splits=5,shuffle=True,random_state=260112+rep)
  pred=np.zeros(len(y))
  for train,test in folds.split(X,y):
   fitted=clone(pipe).fit(X[train],y[train]);pred[test]=fitted.predict_proba(X[test])[:,1]
  au.append(roc_auc_score(y,pred))
  for i,z in enumerate(pred):predrows.append({'model':name,'repeat':rep,'sample_id':str(ids[i]),'response':int(y[i]),'prediction':float(z)})
 fitted=clone(pipe).fit(X,y);ext=fitted.predict_proba(Xt)[:,1]
 rows.append({'model':name,'development_n':len(y),'AUROC_cv_1':au[0],'AUROC_cv_2':au[1],'AUROC_cv_3':au[2],'mean_CV_AUROC':float(np.mean(au)),'external_n':len(yt),'external_AUROC':float(roc_auc_score(yt,ext)),'external_Brier':float(brier_score_loss(yt,ext)),'scope':'post-hoc robustness check with fixed hyperparameters; not nested tuning'})
 for i,z in enumerate(ext):predrows.append({'model':name,'repeat':'external','sample_id':str(cnt.columns[i]),'response':int(yt[i]),'prediction':float(z)})
pd.DataFrame(rows).to_csv(O/'secondary_ML_algorithm_benchmark.csv',index=False);pd.DataFrame(predrows).to_csv(O/'secondary_ML_benchmark_predictions.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
