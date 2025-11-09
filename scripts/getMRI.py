import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import os
from imblearn.over_sampling import SMOTE
import argparse
  
def resample_per_phenotype(X,Yohe,phenotype):
    """
    For each phenotype, dynamically compute upsampling and downsampling
    strategy based on the current Y_train (one-vs-rest).
    Yohe: one-hot encoded labels of training fold.
    """
    Y = Yohe.idxmax(axis = 1)
    class_counts = Y.value_counts()
    negative_class_counts = class_counts.drop(index=phenotype, errors='ignore')
    max_negative_count = negative_class_counts.max()
    negative_class_n = len(negative_class_counts)
    
    positive_target = negative_class_n  * max_negative_count
    negative_target = max_negative_count
    
    sampling_strategy = {}
    for p, count in class_counts.items():
        if p == phenotype:
            sampling_strategy[p] = positive_target
        else:
            sampling_strategy[p] = negative_target

    # 打印当前策略（调试用）
    print(f"Sampling strategy: {sampling_strategy}")
    
    oversample = SMOTE(sampling_strategy=sampling_strategy, random_state=1)
    X_resample, Y_resample = oversample.fit_resample(X, Y)
    
    Y_resample_ohe = pd.get_dummies(Y_resample)


    return X_resample, Y_resample_ohe

def cross_validation_single_phenotype(X_train, Y_train, phenotype):
    """
    for each phenotype, perform cross validation to obtain MRI scores of training samples.
    """
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_all = Y_train['phenotype']
    mri_scores = pd.Series(index=X_train.index, dtype=float, name=phenotype)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_all), start=1):
        X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        Y_tr = Y_train.iloc[tr_idx]
        Y_tr_ohe = pd.get_dummies(Y_tr['phenotype'])
    
        X_resample, Y_resample_ohe = resample_per_phenotype(X_tr, Y_tr_ohe, phenotype)
    
        rf = RandomForestClassifier(oob_score=False, class_weight='balanced', random_state=42,n_jobs=-1)
        rf.fit(X_resample, Y_resample_ohe[phenotype])
    
        pred_proba = rf.predict_proba(X_val)[:, 1]
        mri_scores.iloc[val_idx] = pred_proba
        
    return mri_scores.to_frame() 

def cross_validation_MRI_phenotypes(expr_path,Y_train,phenotypes):
    cv_MRI_res = pd.DataFrame(index=Y_train.index)
    for p in phenotypes:
        expr_file = os.path.join(expr_path,f'{p}_expr_train.csv')
        if not os.path.exists(expr_file):
            print(f"[Warning] {expr_file} not found, skip {p}.")
            continue
        X_train_p = pd.read_csv(expr_file, index_col=0)
        cv_MRI_df = cross_validation_single_phenotype(X_train_p, Y_train, p)
        cv_MRI_res = cv_MRI_res.join(cv_MRI_df, how='left')
    return cv_MRI_res

def main(args):
    expr = pd.read_csv(args.Train_X, index_col=0)
    meta = pd.read_csv(args.Train_Y, index_col=0)

    if 'phenotype' not in meta.columns:
        if 'Condition_combined' in meta.columns:
            meta = meta.rename(columns={'Condition_combined': 'phenotype'})
        else:
            raise ValueError("Train_Y must contain 'phenotype' or 'Condition_combined' column.")

    meta = meta[['phenotype']].copy()
    meta['phenotype'] = meta['phenotype'].astype(str)

    X_train, X_test, Y_train, Y_test = train_test_split(
        expr, meta,
        test_size=0.3,
        random_state=42,
        stratify=meta['phenotype']
    )

    phenotypes = sorted(Y_train['phenotype'].unique())

    feature_n = pd.read_csv(args.FeatureN)
    if not {'phenotype', 'N'}.issubset(feature_n.columns):
        raise ValueError("FeatureN must contain 'phenotype' and 'N' columns.")

    train_expr_dir = os.path.join(args.Output, "input_train")
    os.makedirs(train_expr_dir, exist_ok=True)

    for p in phenotypes:
        rank_file = os.path.join(args.RankPath, f"{p}_result.csv")
        if not os.path.exists(rank_file):
            print(f"[Warning] rank file not found for {p}: {rank_file}, skip.")
            continue

        if p not in feature_n['phenotype'].values:
            print(f"[Warning] no N for {p} in FeatureN, skip.")
            continue

        feature_rank = pd.read_csv(rank_file, index_col=0)
        N = int(feature_n.loc[feature_n['phenotype'] == p, 'N'].values[0])

        selected_feats = [f for f in feature_rank.index[:N] if f in expr.columns]
        if len(selected_feats) == 0:
            print(f"[Warning] no overlapping features for {p}, skip.")
            continue

        train_expr_p = X_train.loc[:, selected_feats]

        train_expr_p.to_csv(os.path.join(train_expr_dir, f"{p}_expr_train.csv"))

    cv_MRI_res = cross_validation_MRI_phenotypes(train_expr_dir, Y_train, phenotypes)

    os.makedirs(args.Output, exist_ok=True)
    out_mri = os.path.join(args.Output, "MRI_level1_train_cv.csv")
    cv_MRI_res.to_csv(out_mri)
    print(f"Saved MRI OOF matrix to: {out_mri}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Construct MRI (level-1) with SMOTE oversampling for each phenotype"
    )
    parser.add_argument("--Train_X", type=str, required=True,
                        help="Expression matrix (samples x features)")
    parser.add_argument("--Train_Y", type=str, required=True,
                        help="Metadata with 'phenotype' or 'Condition_combined'")
    parser.add_argument("--RankPath", type=str, required=True,
                        help="Path to feature rank files: {phenotype}_result.csv")
    parser.add_argument("--FeatureN", type=str, required=True,
                        help="CSV with columns: phenotype, N")
    parser.add_argument("--Output", type=str, required=True,
                        help="Output directory")
    args = parser.parse_args()
    main(args)
  
