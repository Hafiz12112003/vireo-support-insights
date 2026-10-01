"""Cheap local text model (no API calls). Learns what a ticket from a defective lot *sounds like*
from the customer's opening message alone, so tickets with no order number can still be counted."""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, precision_score, recall_score


def _clean(s):
    return s.str.lower().str.replace(r"\[ivr transcript\]", " ", regex=True).str.replace(r"\bvr\d+\b", " ", regex=True)


def fit_symptom_model(t, sku, flagged_lots, seed=0):
    """Label = ticket belongs to a flagged lot. Trained on tickets with a known lot only."""
    d = t[(t.product_sku == sku) & t.lot_base.notna() & ~t.junk_text].copy()
    d["y"] = d.lot_base.isin(flagged_lots).astype(int)
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=5, sublinear_tf=True),
                         LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced"))
    X, y = _clean(d.customer_message), d.y.values
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = (oof >= 0.5).astype(int)
    metrics = dict(n=len(d), flagged_share=float(y.mean()), auc=roc_auc_score(y, oof),
                   precision=precision_score(y, pred), recall=recall_score(y, pred))
    pipe.fit(X, y)
    return pipe, metrics, d.assign(oof=oof)


def score_all(pipe, t, sku):
    m = (t.product_sku == sku) & ~t.junk_text
    s = pd.Series(np.nan, index=t.index)
    s[m] = pipe.predict_proba(_clean(t.loc[m, "customer_message"]))[:, 1]
    return s
