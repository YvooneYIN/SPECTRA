import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc,roc_curve,recall_score,precision_score,precision_recall_curve,f1_score,accuracy_score,roc_auc_score,matthews_corrcoef,cohen_kappa_score
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch
import matplotlib.ticker as mtick
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

import seaborn as sns

from scipy.stats import gaussian_kde
from scipy.stats import mannwhitneyu
from scipy.stats import pearsonr
from scipy.stats import MonteCarloMethod
from scipy.stats import wilcoxon, kendalltau, ranksums
from scipy.stats import sem
from statsmodels.stats.multitest import multipletests
from imblearn.over_sampling import SMOTE

from sklearn.metrics import auc,roc_curve,recall_score,precision_score,precision_recall_curve,f1_score,accuracy_score,roc_auc_score,matthews_corrcoef,cohen_kappa_score
import argparse

#####RANK MICROBES########
def subsample(perc,X_train,y_train):
	n_samples = int(perc * X_train.shape[0])
	indices = np.random.choice(range(X_train.shape[0]), size=n_samples, replace=False)
	X_subset = X_train[indices]
	y_subset = y_train[indices]
	kf = KFold(n_splits=5, shuffle=True, random_state=42)
	coef_res = []
	for train_index, test_index in kf.split(X_subset):
	    X_train_fold, X_val_fold = X_subset[train_index], X_subset[test_index]
	    y_train_fold, y_val_fold = y_subset[train_index], y_subset[test_index]
	    logistic = LogisticRegressionCV(cv=3, 
																	    penalty = 'l1', 
																	    scoring = 'cohen_kappa_score', 
																	    solver= 'saga',
																	    class_weight = 'balanced',
																	    multi_class = 'multinomial',
																	    random_state=0)
	    
	    logistic.fit(X_train_fold, y_train_fold)
			coef_res.append(logistic.coef_.copy())
	coef_res = np.array(coef_res)
	coef_mean_per_class = coef_res.mean(axis=0)
	return coef_mean_per_class

def repeat_subsampling(n_repeats, perc, X_train, y_train):
    coef_results = []
    for _ in range(n_repeats):
        coef_mean_per_class = subsample(perc, X_train, y_train)
        coef_results.append(coef_mean_overall)
    coef_results = np.array(coef_results)
    return coef_results.transpose(2, 1, 0)

#######Finding MSigs#########
def scoring(pred_proba,target):
    fprs,tprs,thresholds = roc_curve(target,pred_proba)
    youden_index = tprs - fprs
    max_index = youden_index.argmax()
    best_threshold = thresholds[max_index]
    best_youden = youden_index[max_index]

    pred_label =  (pred_proba >= best_threshold).astype(int)

    FP = ((pred_label==1) & (target==0)).sum()
    TN = ((pred_label==0) & (target==0)).sum() 
    spe = TN / float(FP + TN) 
    sen = recall_score(target,pred_label,zero_division =0)
    precision = precision_score(target, pred_label,zero_division = 0)
    accuracy = accuracy_score(target, pred_label)
    f1 = f1_score(target, pred_label)
    mcc = matthews_corrcoef(target, pred_label)
    kappa = cohen_kappa_score(target,pred_label)
    precisions, recalls, _  = precision_recall_curve(target,pred_proba)
    aupr = auc(recalls,precisions)
    auc_score = roc_auc_score(target,pred_proba)
    scores = {
        'auc':auc_score,
        'aupr':aupr,
        'Sensitivity': sen,
        'Specificity': spe,
        'Precision': precision,
        'F1': f1,
        'Accuracy': accuracy,
        'MCC': mcc,
        'kappa':kappa,
        'threshold':best_threshold,
        'youden':best_youden
    }
    return scores

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def mutest(target_phenotype,scores_phenotype_df):
    """
    perform mann whitney U test for scores in target phenotype and others
    scores_phenotype_df: predicted scores (DataFrame)
    """
    phenotypes = np.unique(scores_phenotype_df['phenotype'])
    scores_dict = {p : scores_phenotype_df[scores_phenotype_df.iloc[:, 1] == p]['scores'].values.flatten() for p in phenotypes}
    
    p_values = {}
    for p in phenotypes:
        if p == target_phenotype:
            continue 
        pval = mannwhitneyu(scores_dict[target_phenotype], scores_dict[p]).pvalue
        p_values[p]=pval
    return p_values

def resample_per_phenotype(X,Yohe,phenotype):
    """
    For each phenotype, dynamically compute upsampling and downsampling strategy based on the current Y_train.
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

def FSfit_oversample(Xtrain,Xtest,YtrainOhe,Ytest,feature_rank,N,phenotype,clf):
    """
    Feature selection and model fitting function.

    Parameters:
    - Xtrain: Training set features (DataFrame)
    - Xtest: Testing set features (DataFrame)
    - YtrainOhe: One-hot encoded training labels (DataFrame)
    - Ytest: Testing labels (DataFrame)
    - feature_rank: Feature importance ranking (Pandas Series or DataFrame)
    - N: Number of top features to use
    - phenotype: Target phenotype
    - clf: Classifier object (e.g., RandomForestClassifier)

    Returns:
    - sorted_phenotypes: List of phenotypes sorted by median scores
    - pvals: Dictionary of p-values for each phenotype
    - clf: Trained classifier
    """
    if not hasattr(feature_rank, 'index'):
        raise TypeError("feature_rank should be a Pandas Series or DataFrame with an index.")

    if phenotype not in YtrainOhe.columns:
            raise ValueError(f"Phenotype '{phenotype}' is not a column in YtrainOhe.")
    ##Oversample
    X_resample, Y_resample_ohe = resample_per_phenotype(Xtrain,YtrainOhe,phenotype)
    
    #Feature selection
    Xtrain_topN = X_resample[feature_rank.index[:N]]
    Xtest_topN = Xtest[feature_rank.index[:N]]
    
    clf.fit(Xtrain_topN, Y_resample_ohe[phenotype])
    
    pred_proba = clf.predict_proba(Xtest_topN)[:, 1]
    scores = pd.DataFrame(pred_proba, index=Ytest.index, columns=["scores"])
    
    scores_phenotype_df = pd.concat([scores,Ytest['phenotype']],axis = 1)
    #phenotypes = np.unique(Ytest['phenotype'])
    #scores_dict = {p : scores_phenotype_df[scores_phenotype_df.iloc[:, 1] == p]['scores'].values.flatten() for p in phenotypes}
    
       
    pvals = mutest(phenotype,scores_phenotype_df)
    return scores_phenotype_df,pvals


def FindbestN_cv(X_train, Y_train,feature_rank,phenotype,clf,Nlist):
    """
    Perform cross-validation to find the best N across folds.

    Parameters:
    - X_train, Y_train: Training features and labels.
    - feature_rank: Feature ranking (DataFrame).
    - phenotype: Target phenotype for analysis.
    - clf: Classifier object (e.g., RandomForestClassifier).
    - Nlist: List of feature counts (N values) to test (default: range(1, 200)).

    Returns:
    - cv_results: A dictionary where keys are fold indices, and values are dictionaries of fold-specific results.
    - cv_Ns: A list of the best N values found across all folds.
    """
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}
    cv_Ns = []
    cv_scores_df = pd.DataFrame()
    for fold,(train_idx, val_idx) in enumerate(inner_cv.split(X_train, Y_train),start =1):
        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        Y_train_fold, Y_val_fold = Y_train.iloc[train_idx], Y_train.iloc[val_idx]
    
        Y_train_fold_ohe = pd.get_dummies(Y_train_fold['phenotype'])
    
        fold_res,scores_df = FindbestN_fold(Nlist,X_train_fold,X_val_fold,Y_train_fold_ohe,Y_val_fold,feature_rank,phenotype,clf)
        scores_df['fold'] = fold
        scores_df['satisified'] = ['yes' if x in fold_res.keys() else 'no' for x in scores_df.index]
        cv_scores_df = pd.concat([cv_scores_df,scores_df],axis = 0)
        if fold_res:
            best_N_fold = fold_res.keys()
            print(f'find the best N: {best_N_fold} in fold {fold}!')
            cv_Ns.extend(best_N_fold)
            
        else:
            print(f'can not find the best N in fold {fold}!')
            
        cv_results[fold] = fold_res if fold_res else {}
    return cv_results,cv_Ns,cv_scores_df

def FindbestN_fold(Nlist,X_train_fold,X_val_fold,Y_train_fold_ohe,Y_val_fold,feature_rank,phenotype,clf):
    """
    Search for the best number of features (N) in a single fold in cross validation.

    Parameters:
    - Nlist: List of feature counts (N values) to test.
    - X_train_fold, X_val_fold: Training and testing feature sets in a single fold in inner cross validation(DataFrames).
    - Y_train_fold_ohe: One-hot encoded training labels (DataFrame).
    - Y_val_fold: Testing labels (DataFrame).
    - feature_rank: Feature ranking (DataFrame).
    - phenotype: The target phenotype.
    - clf: Classifier object (e.g., RandomForestClassifier).

    Returns:
    - Ns_res: A dictionary where keys are N values, and values are lists of [sorted_phenotypes, pvals].
    - best_N: the first N value that satisfies the conditions.
    """
    
    fold_res = {}
    scores_df = pd.DataFrame()
    phenotypes = Y_train_fold_ohe.columns
    for N in Nlist:
        #print(f'computing for N: {N}')
        #scores_phenotype_df,pvals = FSfit(X_train_fold,X_val_fold,Y_train_fold_ohe,Y_val_fold,feature_rank,N,phenotype,clf)
        scores_phenotype_df,pvals = FSfit_oversample(X_train_fold,X_val_fold,Y_train_fold_ohe,Y_val_fold,feature_rank,N,phenotype,clf)
        scores_phenotype_df['phenotype_ohe'] = scores_phenotype_df['phenotype'].apply(lambda x:1 if x == phenotype else 0)
        scores = scoring(scores_phenotype_df["scores"],scores_phenotype_df['phenotype_ohe'])
        scores = pd.DataFrame(scores,index = [N])
        scores_df = pd.concat([scores_df,scores],axis = 0)
        
        scores_dict = {p : scores_phenotype_df[scores_phenotype_df.iloc[:, 1] == p]['scores'].values.flatten() for p in phenotypes}
    
        scores_medians = {p : np.median(scores_dict[p]) for p in scores_dict}
        sorted_phenotypes = sorted(list(phenotypes), key=lambda x : scores_medians[x], reverse=True)
        
 
        if (phenotype == sorted_phenotypes[0]) and all(pval < 0.05 for p,pval in pvals.items()):
            fold_res[N] = [scores_phenotype_df,pvals]
            #break
    return fold_res,scores_df


def FindbestN_val(Xtrain,Xval,YtrainOhe,Yval,feature_rank,phenotype,clf,Nlist):
    """
    Search for the best number of features (N) on the validation dataset.

    Parameters:
    - Xtrain, Xtest: Training and testing feature sets (DataFrames).
    - YtrainOhe: One-hot encoded training labels (DataFrame).
    - Yval: Validation labels (DataFrame), with a column named "phenotype"
    - feature_rank: Feature ranking (DataFrame).
    - phenotype: The target phenotype.
    - clf: Classifier object (e.g., RandomForestClassifier).
    - Nlist: List of feature counts (N values) to test.

    Returns:
    - Ns_res: A dictionary where keys are N values, and values are lists of [sorted_phenotypes, pvals].
    - sorted_Ns: A sorted list of N values, ranked by the number of satisfied p-values (descending).
    """
    
    val_results = {}
    phenotypes = YtrainOhe.columns
    satisfied_p_n = {}
    scores_df = pd.DataFrame()
    for N in Nlist:
        #print(f'computing for N: {N}')
        #scores_phenotype_df,pvals = FSfit(Xtrain,Xval,YtrainOhe,Yval,feature_rank,N,phenotype,clf)
        scores_phenotype_df,pvals = FSfit_oversample(Xtrain,Xval,YtrainOhe,Yval,feature_rank,N,phenotype,clf)
        scores_phenotype_df['phenotype_ohe'] = scores_phenotype_df['phenotype'].apply(lambda x:1 if x == phenotype else 0)
        scores = scoring(scores_phenotype_df["scores"],scores_phenotype_df['phenotype_ohe'])
        scores = pd.DataFrame(scores,index = [N])
        scores_df = pd.concat([scores_df,scores],axis = 0)
        
        
        scores_dict = {p : scores_phenotype_df[scores_phenotype_df.iloc[:, 1] == p]['scores'].values.flatten() for p in phenotypes}
    
        scores_medians = {p : np.median(scores_dict[p]) for p in scores_dict}
        sorted_phenotypes = sorted(list(phenotypes), key=lambda x : scores_medians[x], reverse=True)
        
        if (phenotype == sorted_phenotypes[0]):
            satisfied_p = sum(pval < 0.05 for p, pval in pvals.items())
            satisfied_p_n[N] = satisfied_p
        
        val_results[N] = [scores_phenotype_df,pvals]

    sorted_Ns = scores_df.sort_values('auc', ascending=False)
        
    return val_results,sorted_Ns,scores_df


def main(args):
    X = pd.read_csv(args.Train_X,index_col=0)
    Y = pd.read_csv(args.Train_Y,index_col=0)
    Y.rename(columns={'Condition_combined':'phenotype'},inplace=True)


    Y = Y.loc[:,['phenotype']]
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=42,stratify=Y['phenotype'])
    X_val,X_test,Y_val,Y_test = train_test_split(X_test,Y_test,test_size=0.5,random_state=42,stratify=Y_test['phenotype'])
    test_scores_df = pd.DataFrame()
    phenotypes = np.unique(Y_train['phenotype'])
    Msigs = {}
    for phenotype in phenotypes:
        feature_rank = pd.read_csv(f'{args.RankPath}/{phenotype}_result.csv', index_col=0)
        rf = RandomForestClassifier(oob_score=True, class_weight='balanced', random_state=42)  
        fold_results,cv_Ns,cv_scores_df = FindbestN_cv(X_train, Y_train,feature_rank,phenotype,rf,range(10,100))
        cv_scores_df['N'] = cv_scores_df.index
        #cv_scores_df.to_csv(os.path.join(f'{args.Output}','cv',f'{phenotype}_res.csv'))
        
        satisfied_df = cv_scores_df[cv_scores_df['satisified'] == 'yes']
        cv_Ns = np.unique(satisfied_df.index)

        Y_train_ohe = pd.get_dummies(Y_train['phenotype'])
        val_results,sorted_Ns,val_scores_df = FindbestN_val(X_train,X_val,Y_train_ohe,Y_val,feature_rank,phenotype,rf,cv_Ns)
        #val_scores_df.to_csv(os.path.join(f'{args.Output}','val',f'{phenotype}_res.csv'))
        
        #internal test
        best_N = sorted_Ns[0]
        Msigs[phenotype] = feature_rank.index[:N]
    
  
  df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in Msigs.items()]))
  return df

