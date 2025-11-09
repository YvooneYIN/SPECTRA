import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import os

from sklearn.metrics import roc_auc_score

def select_MRI_panel_by_RFE(MRI_train, Y_train, min_features=1, n_splits=5, random_state=42):
    """
    在 MRI (level-1) 特征上做迭代特征消除，选出组合表现最好的 MRI 子集。

    MRI_train: DataFrame, 行为训练样本，列为各 phenotype 对应的 MRI 分数
    Y_train: DataFrame 或 Series, 含 'phenotype'
    返回:
        best_features: 表现最好的 MRI 特征名列表
        history: DataFrame, 记录每一步的特征数和宏平均 AUC
    """
    if isinstance(Y_train, pd.DataFrame):
        y = Y_train['phenotype'].astype(str)
    else:
        y = Y_train.astype(str)

    cols = list(MRI_train.columns)
    history = []

    while len(cols) >= min_features:
        X_cur = MRI_train[cols]

        # K-fold OOF 预测
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        oof_pred = pd.DataFrame(0.0, index=MRI_train.index, columns=sorted(y.unique()))

        for tr_idx, val_idx in skf.split(X_cur, y):
            X_tr, X_val = X_cur.iloc[tr_idx], X_cur.iloc[val_idx]
            y_tr = y.iloc[tr_idx]

            clf = RandomForestClassifier(
                n_estimators=500,
                class_weight='balanced',
                random_state=random_state,
                n_jobs=-1
            )
            clf.fit(X_tr, y_tr)
            proba = clf.predict_proba(X_val)
            oof_pred.iloc[val_idx] = proba

        # 计算宏平均 AUC
        y_ohe = pd.get_dummies(y)
        aucs = []
        for ph in oof_pred.columns:
            aucs.append(roc_auc_score(y_ohe[ph], oof_pred[ph]))
        macro_auc = np.mean(aucs)

        history.append({
            "n_features": len(cols),
            "macro_auc": macro_auc,
            "features": cols.copy()
        })

        # 用全数据训练一次拿 feature_importances_，删最小的 MRI
        clf_full = RandomForestClassifier(
            n_estimators=500,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        clf_full.fit(X_cur, y)
        importances = pd.Series(clf_full.feature_importances_, index=cols)

        drop_feat = importances.idxmin()
        print(f"[RFE] n={len(cols)}, macro_auc={macro_auc:.4f}, drop {drop_feat}")
        cols.remove(drop_feat)

    hist_df = pd.DataFrame(history)
    # 选 macro_auc 最大的那一行
    best_row = hist_df.iloc[hist_df['macro_auc'].idxmax()]
    best_features = best_row['features']

    print(f"[RFE] Best MRI subset size={len(best_features)}, macro_auc={best_row['macro_auc']:.4f}")
    return best_features, hist_df



def SPECTRA_cv(MRI_train_sel, Y_train, n_splits=10, random_state=42):
    """
    在选定的 MRI 组合上做多分类 RF 的 K 折交叉验证，
    返回每个训练样本的 OOF 预测概率 (SPECTRA level-2 scores)。
    """
    if isinstance(Y_train, pd.DataFrame):
        y = Y_train['phenotype'].astype(str)
    else:
        y = Y_train.astype(str)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    classes = sorted(y.unique())
    oof_pred = pd.DataFrame(index=MRI_train_sel.index, columns=classes, dtype=float)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(MRI_train_sel, y), start=1):
        X_tr, X_val = MRI_train_sel.iloc[tr_idx], MRI_train_sel.iloc[val_idx]
        y_tr = y.iloc[tr_idx]

        rf = RandomForestClassifier(
            n_estimators=500,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        rf.fit(X_tr, y_tr)
        proba = rf.predict_proba(X_val)
        oof_pred.iloc[val_idx] = proba

        print(f"[SPECTRA CV] Fold {fold} done.")

    return oof_pred


def SPECTRA_test(MRI_train_sel, Y_train, MRI_test_sel, random_state=42):
    """
    用全部训练集 (选定 MRI 特征) 拟合最终 SPECTRA 模型，
    返回测试集的预测概率。
    """
    if isinstance(Y_train, pd.DataFrame):
        y = Y_train['phenotype'].astype(str)
    else:
        y = Y_train.astype(str)

    rf = RandomForestClassifier(
        n_estimators=500,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(MRI_train_sel, y)
    proba = rf.predict_proba(MRI_test_sel)

    return pd.DataFrame(proba, index=MRI_test_sel.index, columns=rf.classes_)


def main(args):
    # 1. 读取训练集 MRI (level-1 OOF scores)
    mri_train = pd.read_csv(args.MRI_TRAIN, index_col=0)

    # 2. 读取训练集表型信息
    meta_train = pd.read_csv(args.Train_Y, index_col=0)
    if 'phenotype' not in meta_train.columns:
        if 'Condition_combined' in meta_train.columns:
            meta_train = meta_train.rename(columns={'Condition_combined': 'phenotype'})
        else:
            raise ValueError("Train_Y 必须包含 'phenotype' 或 'Condition_combined' 列。")
    y_train = meta_train.loc[mri_train.index, 'phenotype'].astype(str)

    os.makedirs(args.Output, exist_ok=True)

    # 3. 基于 RFE 在 MRI 上选择最优组合
    best_mri_cols, rfe_history = select_MRI_panel_by_RFE(
        mri_train,
        y_train,
        min_features=args.min_features,
        n_splits=args.rfe_cv,
        random_state=42
    )

    mri_train_sel = mri_train[best_mri_cols]
    rfe_history.to_csv(os.path.join(args.Output, "SPECTRA_MRI_RFE_history.csv"))
    pd.Series(best_mri_cols, name="selected_MRI").to_csv(
        os.path.join(args.Output, "SPECTRA_selected_MRI_panel.csv")
    )
    print(f"[SPECTRA] Selected {len(best_mri_cols)} MRI features.")

    # 4. 在选定 MRI 上做 SPECTRA 的 K 折交叉验证（训练集 OOF 预测）
    spectra_oof = SPECTRA_cv(
        mri_train_sel,
        y_train,
        n_splits=args.spectra_cv,
        random_state=42
    )
    spectra_oof.to_csv(os.path.join(args.Output, "SPECTRA_train_OOF.csv"))
    print("[SPECTRA] Saved training OOF predictions.")

    # 5. 如果提供了测试集 MRI，就在同一 MRI 子集上输出测试集预测
    if args.MRI_TEST is not None:
        mri_test = pd.read_csv(args.MRI_TEST, index_col=0)
        # 测试集可能多列，这里严格按选定 MRI 子集取列
        mri_test_sel = mri_test[best_mri_cols]
        spectra_test = SPECTRA_test(
            mri_train_sel,
            y_train,
            mri_test_sel,
            random_state=42
        )
        spectra_test.to_csv(os.path.join(args.Output, "SPECTRA_test_pred.csv"))
        print("[SPECTRA] Saved test set predictions.")

    print("[SPECTRA] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SPECTRA: build level-2 model on MRI features with RFE-based MRI panel selection"
    )
    parser.add_argument(
        "--MRI_TRAIN",
        type=str,
        required=True,
        help="Path to MRI level-1 OOF scores for training samples (rows=samples, cols=MRI features)."
    )
    parser.add_argument(
        "--Train_Y",
        type=str,
        required=True,
        help="Path to training metadata with 'phenotype' or 'Condition_combined'."
    )
    parser.add_argument(
        "--MRI_TEST",
        type=str,
        required=False,
        default=None,
        help="(Optional) Path to MRI features for test samples; if provided, outputs SPECTRA_test_pred."
    )
    parser.add_argument(
        "--Output",
        type=str,
        required=True,
        help="Output directory."
    )
    parser.add_argument(
        "--min_features",
        type=int,
        default=2,
        help="Minimum number of MRI features to keep during RFE."
    )
    parser.add_argument(
        "--rfe_cv",
        type=int,
        default=5,
        help="CV folds used inside MRI RFE selection."
    )
    parser.add_argument(
        "--spectra_cv",
        type=int,
        default=10,
        help="CV folds for final SPECTRA OOF training."
    )

    args = parser.parse_args()
    main(args)
