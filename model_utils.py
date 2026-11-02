import os, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix,classification_report

def load_demo_data(): return pd.read_csv(os.path.join(os.path.dirname(__file__),"data","demo_telemetry.csv"))
def prepare(df):
    if "fault_label" not in df: raise ValueError("Required target column: fault_label")
    X=df.drop(columns=["fault_label"]).copy()
    for c in X: X[c]=pd.to_numeric(X[c],errors="coerce")
    X=X.replace([np.inf,-np.inf],np.nan)
    X=SimpleImputer(strategy="median").fit_transform(X)
    y=LabelEncoder().fit_transform(df.fault_label.astype(str))
    return X,y,list(df.drop(columns=["fault_label"]).columns)
def metric(y,p,labels):
    a=accuracy_score(y,p); pr,re,f,_=precision_recall_fscore_support(y,p,average="weighted",zero_division=0)
    return {"accuracy":float(a),"precision":float(pr),"recall":float(re),"f1":float(f),"confusion":confusion_matrix(y,p,labels=labels).tolist()}
def run_experiment(df,label_ratio=.1,threshold=.8):
    X,y,features=prepare(df)
    if len(np.unique(y))<2: raise ValueError("Need at least two classes.")
    Xt,Xv,yt,yv=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
    rng=np.random.RandomState(42); ids=[]
    for cl in np.unique(yt):
        ix=np.where(yt==cl)[0]; n=max(1,round(len(ix)*label_ratio))
        ids.extend(rng.choice(ix,min(n,len(ix)),replace=False).tolist())
    ids=np.array(sorted(set(ids))); semi_y=np.full(len(yt),-1,dtype=int); semi_y[ids]=yt[ids]
    base=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight="balanced"))
    model=SelfTrainingClassifier(base,threshold=threshold,max_iter=10).fit(Xt,semi_y)
    semi=metric(yv,model.predict(Xv),list(np.unique(y)))
    baseline=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight="balanced")).fit(Xt[ids],yt[ids])
    sup=metric(yv,baseline.predict(Xv),list(np.unique(y)))
    return {"model":"Self-Training (Pseudo-Labeling) + Logistic Regression","label_ratio":label_ratio,"threshold":threshold,
    "n_train":len(Xt),"n_labeled":len(ids),"n_unlabeled":len(Xt)-len(ids),"n_test":len(Xv),"classes":list(np.unique(y)),
    "supervised":sup,"semi":semi,"pseudo_labeled":max(0,int(np.sum(model.transduction_!=-1)-len(ids))),
    "note":"This prototype implements pseudo-labeling/self-training. Ladder Network is NOT implemented in this version; treat it as planned extension unless you add and validate it."}
