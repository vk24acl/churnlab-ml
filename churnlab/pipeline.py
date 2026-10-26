"""Chronological training, disjoint calibration and validation, untouched test."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit, logit
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, precision_recall_curve, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ['tenure_months', 'weekly_sessions', 'support_tickets', 'monthly_fee', 'contract_monthly', 'usage_change']

def make_data(n=6000, seed=42):
    """One snapshot per fictional customer. Features precede simulated next-month label."""
    r = np.random.default_rng(seed)
    t = np.linspace(0, 1, n)
    tenure = r.integers(1, 61, n)
    sessions = r.gamma(3, 2, n)
    tickets = r.poisson(1.1, n)
    fee = r.uniform(15, 120, n)
    monthly = r.binomial(1, .65, n)
    change = np.clip(r.normal(-.1 - .2*t, .35, n), -1, 1)
    x = np.column_stack([tenure, sessions, tickets, fee, monthly, change])
    probability = expit(-2.4-.018*tenure-.18*sessions+.4*tickets+.008*fee+.8*monthly-1.7*change+.45*(tickets>2)*(sessions<4)+.25*t)
    return x, r.binomial(1, probability)

def split_indices(n):
    a,b,c = [int(n*f) for f in (.55,.70,.85)]
    return [np.arange(0,a),np.arange(a,b),np.arange(b,c),np.arange(c,n)]

def score_logit(model, x):
    return logit(np.clip(model.predict_proba(x)[:,1], 1e-6, 1-1e-6)).reshape(-1,1)

def choose_threshold(y, p):
    precision, recall, thresholds = precision_recall_curve(y,p)
    f = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    return float(thresholds[np.argmax(f)])

def targeting(y, p, fraction=.1):
    if not 0 < fraction <= 1: raise ValueError('fraction must be in (0,1]')
    k = max(1,int(np.ceil(len(y)*fraction)))
    selected = np.argsort(-p,kind='stable')[:k]
    precision = float(np.mean(y[selected]))
    return {'capacity':k,'precision':precision,'recall':float(np.sum(y[selected])/max(1,np.sum(y))), 'lift':precision/max(float(np.mean(y)),1e-12)}

def metrics(y,p,threshold):
    return {'average_precision':float(average_precision_score(y,p)), 'roc_auc':float(roc_auc_score(y,p)), 'brier':float(brier_score_loss(y,p)), 'f1':float(f1_score(y,p>=threshold)), 'prevalence':float(np.mean(y)), 'targeting_10pct':targeting(y,p)}

def run(output='reports'):
    x,y=make_data(); train,cal,val,test=split_indices(len(y))
    models={'prevalence_baseline':DummyClassifier(strategy='prior'), 'logistic_regression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000)), 'gradient_boosting':HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=10,l2_regularization=5,random_state=42)}
    candidates={}; fitted={}
    for name,model in models.items():
        model.fit(x[train],y[train])
        calibrator=LogisticRegression(C=1e3).fit(score_logit(model,x[cal]),y[cal])
        p=calibrator.predict_proba(score_logit(model,x[val]))[:,1]
        candidates[name]={'validation_ap':float(average_precision_score(y[val],p))}
        fitted[name]=(model,calibrator,p)
    winner=max(candidates,key=lambda name:candidates[name]['validation_ap'])
    model,calibrator,pval=fitted[winner]
    threshold=choose_threshold(y[val],pval)
    ptest=calibrator.predict_proba(score_logit(model,x[test]))[:,1]
    result={'seed':42,'data':'6000 synthetic customers, chronological cohort order; no real customer records', 'features':FEATURES,'split_sizes':dict(zip(['train','calibration','validation','test'],map(len,[train,cal,val,test]))),'candidates':candidates,'selected_model':winner,'validation_threshold':threshold,'test':metrics(y[test],ptest,threshold),'capacity_curve':[dict(fraction=f,**targeting(y[test],ptest,f)) for f in [.05,.1,.2,.3,.5,1]],'calibration':[],'permutation_importance':[]}
    # Fixed-model permutation diagnostic on test; never used for model/threshold selection.
    rng=np.random.default_rng(43)
    for j,name in enumerate(FEATURES):
        losses=[]
        for _ in range(5):
            perm=x[test].copy();perm[:,j]=rng.permutation(perm[:,j])
            prob=calibrator.predict_proba(score_logit(model,perm))[:,1]
            losses.append(result['test']['average_precision']-average_precision_score(y[test],prob))
        result['permutation_importance'].append({'feature':name,'ap_drop_mean':float(np.mean(losses)),'ap_drop_std':float(np.std(losses))})
    for lo in np.arange(0,1,.1):
        mask=(ptest>=lo)&(ptest<lo+.1)
        if mask.any():result['calibration'].append({'count':int(mask.sum()),'predicted':float(ptest[mask].mean()),'observed':float(y[test][mask].mean())})
    path=Path(output);path.mkdir(parents=True,exist_ok=True)
    (path/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    if winner=='logistic_regression':
        scaler,linear=model.steps[0][1],model.steps[1][1]
        export={'features':FEATURES,'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'coefficients':linear.coef_[0].tolist(),'intercept':float(linear.intercept_[0]),'calibration_coefficient':float(calibrator.coef_[0,0]),'calibration_intercept':float(calibrator.intercept_[0]),'threshold':threshold}
        (path/'model.json').write_text(json.dumps(export,indent=2)+'\n')
        fixture={'x':x[test[:10]].tolist(),'probabilities':ptest[:10].tolist()}
        (path/'parity.json').write_text(json.dumps(fixture,indent=2)+'\n')
    (path/'summary.md').write_text(f"# Measured results\n\nSelected on validation: **{winner}**.\n\nHeld-out synthetic test: AP {result['test']['average_precision']:.3f}; ROC AUC {result['test']['roc_auc']:.3f}; Brier {result['test']['brier']:.3f}; top-decile lift {result['test']['targeting_10pct']['lift']:.2f}x.\n\nThese results measure prediction on a simulated cohort, not retention uplift.\n")
    return result

if __name__=='__main__':
    print(json.dumps(run(),indent=2))
