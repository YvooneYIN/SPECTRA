import pandas as pd
import numpy as np
import sys
import warnings
import os

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc,roc_curve,recall_score,precision_score,precision_recall_curve,f1_score,accuracy_score,roc_auc_score,matthews_corrcoef,cohen_kappa_score

import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu
from imblearn.over_sampling import SMOTE
import argparse


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
	scores_dict = {p: scores_phenotype_df[scores_phenotype_df['phenotype'] == p]['scores'].values.flatten() for p in phenotypes}

	
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

def compute_MRI_using_MSig_panel(Xtrain,Xtest,YtrainOhe,Ytest,msig_rank,N,phenotype,clf):
	"""
	Msig selection and model fitting function.

	Parameters:
	- Xtrain: Training set (DataFrame)
	- Xtest: Testing set (DataFrame)
	- YtrainOhe: One-hot encoded training labels (DataFrame)
	- Ytest: Testing labels (DataFrame)
	- msig_rank: msig ranking (Pandas Series or DataFrame)
	- N: Number of top Msig to use
	- phenotype: Target phenotype
	- clf: Classifier object (e.g., RandomForestClassifier)

	Returns:
	- scores_phenotype_df: Df of phenotypes sorted by median scores
	- pvals: Dictionary of p-values for each phenotype
	"""
	if not hasattr(msig_rank, 'index'):
		raise TypeError("msig_rank should be a Pandas Series or DataFrame with an index.")

	if phenotype not in YtrainOhe.columns:
			raise ValueError(f"Phenotype '{phenotype}' is not a column in YtrainOhe.")
	##Oversample
	X_resample, Y_resample_ohe = resample_per_phenotype(Xtrain,YtrainOhe,phenotype)
	
	Xtrain_msig = X_resample[msig_rank.index[:N]]
	Xtest_msig = Xtest[msig_rank.index[:N]]
	
	clf.fit(Xtrain_msig, Y_resample_ohe[phenotype])
	
	pred_proba = clf.predict_proba(Xtest_msig)[:, 1]
	scores = pd.DataFrame(pred_proba, index=Ytest.index, columns=["scores"])
	
	scores_phenotype_df = pd.concat([scores,Ytest['phenotype']],axis = 1) 
	pvals = mutest(phenotype,scores_phenotype_df)
	return scores_phenotype_df,pvals


def select_MSig_panel_size_cv(X_train, Y_train,msig_rank,phenotype,clf,Nlist):
	"""
	Perform cross-validation to find the best N across folds.

	Parameters:
	- X_train, Y_train: Training Msig and labels.
	- msig_rank: Msig ranking (DataFrame).
	- phenotype: Target phenotype for analysis.
	- clf: Classifier object (e.g., RandomForestClassifier).
	- Nlist: List of Msig counts (N values) to test (default: range(1, 200)).

	Returns:
	- cv_results: A dictionary where keys are fold indices, and values are dictionaries of fold-specific results.
	- cv_Ns: A list of the best N values found across all folds.
	"""
	inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
	cv_results = {}
	cv_Ns = []
	cv_scores_df = pd.DataFrame()
	y_label = Y_train['phenotype']
	for fold,(train_idx, val_idx) in enumerate(inner_cv.split(X_train, y_label),start =1):
		X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
		Y_train_fold, Y_val_fold = Y_train.iloc[train_idx], Y_train.iloc[val_idx]
	
		Y_train_fold_ohe = pd.get_dummies(Y_train_fold['phenotype'])
	
		fold_res,scores_df = select_MSig_panel_size_in_fold(Nlist,X_train_fold,X_val_fold,Y_train_fold_ohe,Y_val_fold,msig_rank,phenotype,clf)
		scores_df['fold'] = fold
		scores_df['satisified'] = ['yes' if x in fold_res.keys() else 'no' for x in scores_df.index]
		cv_scores_df = pd.concat([cv_scores_df,scores_df],axis = 0)
		if fold_res:
			best_Ns_in_fold = list(fold_res.keys())
			print(f'find suitable MSig panel sizes {best_Ns_in_fold} for {phenotype} in fold {fold}!')
			cv_Ns.extend(best_Ns_in_fold)
			
		else:
			print(f'cannot find suitable MSig panel size for {phenotype} in fold {fold}!')
			
		cv_results[fold] = fold_res if fold_res else {}
	return cv_results,cv_Ns,cv_scores_df

def select_MSig_panel_size_in_fold(Nlist,X_train_fold,X_val_fold,Y_train_fold_ohe,Y_val_fold,msig_rank,phenotype,clf):
	"""
	Search for the best number of Msig (N) in a single fold in cross validation.

	Parameters:
	- Nlist: List of Msig counts (N values) to test.
	- X_train_fold, X_val_fold: Training and testing Msig sets in a single fold in inner cross validation(DataFrames).
	- Y_train_fold_ohe: One-hot encoded training labels (DataFrame).
	- Y_val_fold: Testing labels (DataFrame).
	- msig_rank: MSig ranking (DataFrame).
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
		scores_phenotype_df,pvals = compute_MRI_using_MSig_panel(X_train_fold,X_val_fold,Y_train_fold_ohe,Y_val_fold,msig_rank,N,phenotype,clf)
		scores_phenotype_df['phenotype_ohe'] = scores_phenotype_df['phenotype'].apply(lambda x:1 if x == phenotype else 0)
		scores = scoring(scores_phenotype_df["scores"],scores_phenotype_df['phenotype_ohe'])
		scores = pd.DataFrame(scores,index = [N])
		scores_df = pd.concat([scores_df,scores],axis = 0)
		
		scores_dict = {p: scores_phenotype_df[scores_phenotype_df['phenotype'] == p]['scores'].values.flatten()for p in phenotypes}
	
		scores_medians = {p : np.median(scores_dict[p]) for p in scores_dict}
		sorted_phenotypes = sorted(list(phenotypes), key=lambda x : scores_medians[x], reverse=True)
		
 
		if (phenotype == sorted_phenotypes[0]) and all(pval < 0.05 for p,pval in pvals.items()):
			fold_res[N] = [scores_phenotype_df,pvals]
			#break
	return fold_res,scores_df


def validate_MSig_panel_size(Xtrain,Xval,YtrainOhe,Yval,msig_rank,phenotype,clf,Nlist):
	"""
	Search for the best number of Msig (N) on the validation dataset.

	Parameters:
	- Xtrain, Xtest: Training and testing Msig sets (DataFrames).
	- YtrainOhe: One-hot encoded training labels (DataFrame).
	- Yval: Validation labels (DataFrame), with a column named "phenotype"
	- msig_rank: Msig ranking (DataFrame).
	- phenotype: The target phenotype.
	- clf: Classifier object (e.g., RandomForestClassifier).
	- Nlist: List of Msig counts (N values) to test.

	Returns:
	- Ns_res: A dictionary where keys are N values, and values are lists of [sorted_phenotypes, pvals].
	- sorted_Ns: A sorted list of N values, ranked by the number of satisfied p-values (descending).
	"""
	
	val_results = {}
	phenotypes = YtrainOhe.columns
	scores_df = pd.DataFrame()
	for N in Nlist:
		scores_phenotype_df,pvals = compute_MRI_using_MSig_panel(Xtrain,Xval,YtrainOhe,Yval,msig_rank,N,phenotype,clf)
		scores_phenotype_df['phenotype_ohe'] = scores_phenotype_df['phenotype'].apply(lambda x:1 if x == phenotype else 0)
		scores = scoring(scores_phenotype_df["scores"],scores_phenotype_df['phenotype_ohe'])
		scores = pd.DataFrame(scores,index = [N])
		scores_df = pd.concat([scores_df,scores],axis = 0)   
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
	for phenotype in phenotypes:
		msig_rank = pd.read_csv(f'{args.RankPath}/{phenotype}_result.csv', index_col=0)
		rf = RandomForestClassifier(oob_score=True, class_weight='balanced', random_state=42)  
		fold_results,cv_Ns,cv_scores_df = select_MSig_panel_size_cv(X_train, Y_train,msig_rank,phenotype,rf,range(10,11))
		cv_scores_df['N'] = cv_scores_df.index
		cv_scores_df.to_csv(os.path.join(f'{args.Output}','1CV',f'{phenotype}_res.csv'))
		
		satisfied_df = cv_scores_df[cv_scores_df['satisified'] == 'yes']
		cv_Ns = np.unique(satisfied_df.index)

		Y_train_ohe = pd.get_dummies(Y_train['phenotype'])
		val_results,sorted_Ns,val_scores_df = validate_MSig_panel_size(X_train,X_val,Y_train_ohe,Y_val,msig_rank,phenotype,rf,cv_Ns)
		val_scores_df.to_csv(os.path.join(f'{args.Output}','2Validation',f'{phenotype}_res.csv'))
		
		#internal test
		best_N = sorted_Ns.index[0]
		scores_phenotype_df,pvals = compute_MRI_using_MSig_panel(X_train,X_test,Y_train_ohe,Y_test,msig_rank,best_N,phenotype,rf)
		scores_phenotype_df['target'] = scores_phenotype_df['phenotype'].apply(lambda x:1 if x == phenotype else 0)
		scores_phenotype_df.to_csv(os.path.join(f'{args.Output}','3Test',f'{phenotype}_probs.csv'))
		
		scores = scoring(scores_phenotype_df["scores"],scores_phenotype_df['target'])
		test_scores = pd.DataFrame(scores,index = [best_N])
		test_scores.to_csv(os.path.join(f'{args.Output}','3Test',f'{phenotype}_res.csv'))
		test_scores_df = pd.concat([test_scores_df,test_scores],axis = 0)
	test_scores_df.to_csv(os.path.join(f'{args.Output}','3Test',f'summary_test.csv'))



if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Msig Selection with oversample')
	parser.add_argument('--Train_X', type=str, required=True, help='Train_X : Expression file for training and internal validation/test')
	parser.add_argument('--Train_Y', type=str, required=True, help='Train_Y : Metadata file for training and internal validation/test')
	parser.add_argument('--RankPath', type=str, required=True, help='RankPath: Path to Msig rank dfs')
	parser.add_argument('--Output',type = str,help = 'output dir: output path')
	args = parser.parse_args()
