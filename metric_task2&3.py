from sklearn import metrics
import torch
import numpy as np


#Calculating Confusion Matrix
def confusion_matrix(truth, pred):
	pred= torch.flatten(pred, start_dim=0, end_dim=1).detach().cpu().numpy()
	truth= torch.flatten(truth, start_dim=0, end_dim=1).detach().cpu().numpy()
	TP=0
	TN=0
	FP=0
	FN=0
	con_mat= metrics.confusion_matrix(truth, pred)
	TP = np.diag(con_mat)
	FP = con_mat[:].sum(axis=0) - np.diag(con_mat)
	FN = con_mat[:].sum(axis=1) - np.diag(con_mat)
	TN = con_mat[:].sum() - (FP+FN+TP)

	return TP.sum(),TN.sum(),FP.sum(),FN.sum()

	
#Calculating F1-score and jacard score
def evaluation_metrics(truth, pred):
	pred= torch.flatten(pred, start_dim=0, end_dim=1).detach().cpu().numpy()
	truth= torch.flatten(truth, start_dim=0, end_dim=1).detach().cpu().numpy()
	f1  = metrics.f1_score(truth, pred,average = 'micro')
	jacc = metrics.jaccard_score(truth ,pred, average = 'micro')
	return f1,jacc
