"""Train the strategy-selection classifiers and evaluate them against the
rule-based baseline.

Two feature sets are compared: the eight features named in the specification,
and an extended fourteen.  The comparison is the point -- the required set
contains no feature that separates constant folding from constant propagation,
and the gap between the two columns measures exactly what that costs.
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optselect.features import EXTENDED_FEATURES, REQUIRED_FEATURES
from optselect.optimizations import SHORT, STRATEGIES
from optselect.selectors import RuleBasedSelector

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'out')
SEED = 42


def build_models():
    return {
        'DecisionTree': DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=3, random_state=SEED),
        'RandomForest': RandomForestClassifier(
            n_estimators=400, max_depth=12, min_samples_leaf=1,
            random_state=SEED, n_jobs=-1),
        'LogisticRegression': Pipeline([
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(max_iter=5000, C=2.0,
                                       random_state=SEED))]),
        'MLP': Pipeline([
            ('scale', StandardScaler()),
            ('clf', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=4000,
                                  alpha=1e-3, random_state=SEED))]),
    }


def per_class_table(y_true, y_pred):
    rep = classification_report(y_true, y_pred, labels=STRATEGIES,
                                target_names=STRATEGIES, output_dict=True,
                                zero_division=0)
    return {s: {'precision': rep[s]['precision'], 'recall': rep[s]['recall'],
                'f1': rep[s]['f1-score'], 'support': int(rep[s]['support'])}
            for s in STRATEGIES}


def evaluate_rule_based(df_test):
    sel = RuleBasedSelector()
    preds = []
    for _, row in df_test.iterrows():
        feats = {k: float(row[k]) for k in EXTENDED_FEATURES}
        best, _conf, _sc = sel.select(feats)
        preds.append(best)
    return np.array(preds)


def main():
    df = pd.read_csv(os.path.join(OUT, 'dataset.csv'))
    y = df['best_strategy'].values
    results = {'n_programs': int(len(df)), 'seed': SEED, 'feature_sets': {}}

    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.30, random_state=SEED, stratify=y)
    df_train, df_test = df.iloc[train_idx], df.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f'{len(df)} programs -> {len(df_train)} train / {len(df_test)} test '
          f'(stratified)\n')

    # ---------------------------------------------------- rule-based baseline
    rb_pred = evaluate_rule_based(df_test)
    rb_acc = accuracy_score(y_test, rb_pred)
    rb_full = evaluate_rule_based(df)
    results['rule_based'] = {
        'test_accuracy': float(rb_acc),
        'test_macro_f1': float(f1_score(y_test, rb_pred, average='macro',
                                        labels=STRATEGIES, zero_division=0)),
        'full_accuracy': float(accuracy_score(y, rb_full)),
        'per_class': per_class_table(y_test, rb_pred),
        'confusion': confusion_matrix(y_test, rb_pred, labels=STRATEGIES).tolist(),
    }
    print(f'rule-based baseline : test accuracy {rb_acc:.3f}, '
          f'macro-F1 {results["rule_based"]["test_macro_f1"]:.3f}')

    best_overall = (None, -1.0, None, None)

    for set_name, feature_names in (('required_8', REQUIRED_FEATURES),
                                    ('extended_14', EXTENDED_FEATURES)):
        X = df[feature_names].values.astype(float)
        X_train, X_test = X[train_idx], X[test_idx]
        entry = {'features': feature_names, 'models': {}}
        print(f'\n--- feature set: {set_name} ({len(feature_names)} features) ---')

        for name, model in build_models().items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)
            macro = f1_score(y_test, pred, average='macro', labels=STRATEGIES,
                             zero_division=0)
            cv = cross_val_score(
                model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                scoring='accuracy', n_jobs=-1)
            entry['models'][name] = {
                'test_accuracy': float(acc),
                'test_macro_f1': float(macro),
                'cv_accuracy_mean': float(cv.mean()),
                'cv_accuracy_std': float(cv.std()),
                'per_class': per_class_table(y_test, pred),
                'confusion': confusion_matrix(y_test, pred,
                                              labels=STRATEGIES).tolist(),
            }
            print(f'  {name:20} acc {acc:.3f}  macroF1 {macro:.3f}  '
                  f'CV {cv.mean():.3f} +/- {cv.std():.3f}')
            if acc > best_overall[1]:
                best_overall = (f'{name}[{set_name}]', acc, model, feature_names)

            if name == 'RandomForest':
                rf = model
                entry['feature_importance'] = {
                    f: float(v) for f, v in zip(feature_names,
                                                rf.feature_importances_)}
                pred_df = df_test[['program_id', 'best_strategy']].copy()
                pred_df['predicted'] = pred
                proba = rf.predict_proba(X_test)
                classes = list(rf.classes_)
                pred_df['confidence'] = proba.max(axis=1)
                for s in STRATEGIES:
                    pred_df[f'p_{SHORT[s]}'] = proba[:, classes.index(s)] \
                        if s in classes else 0.0
                pred_df.to_csv(
                    os.path.join(OUT, f'predictions_rf_{set_name}.csv'), index=False)

        results['feature_sets'][set_name] = entry

    # ------------------------------------------------------------- persist
    name, acc, model, feature_names = best_overall
    print(f'\nbest model: {name} (test accuracy {acc:.3f}) -- refitting on all data')
    X_all = df[feature_names].values.astype(float)
    model.fit(X_all, y)
    with open(os.path.join(OUT, 'model.pkl'), 'wb') as fh:
        pickle.dump({'model': model, 'feature_names': feature_names,
                     'name': name}, fh)

    # also persist an 8-feature model, so the loop can be run under the exact
    # feature set the specification mandates
    spec_model = build_models()['RandomForest']
    spec_model.fit(df[REQUIRED_FEATURES].values.astype(float), y)
    with open(os.path.join(OUT, 'model_required8.pkl'), 'wb') as fh:
        pickle.dump({'model': spec_model, 'feature_names': REQUIRED_FEATURES,
                     'name': 'RandomForest[required_8]'}, fh)

    results['best_model'] = {'name': name, 'test_accuracy': float(acc),
                             'features': feature_names}
    results['class_counts'] = {s: int((y == s).sum()) for s in STRATEGIES}
    with open(os.path.join(OUT, 'metrics.json'), 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'wrote {OUT}/metrics.json, {OUT}/model.pkl')


if __name__ == '__main__':
    main()
