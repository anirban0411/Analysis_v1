import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
from xgboost import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import uproot
import seaborn
import ROOT
import array
import math
import operator
from array import array
#%jsroot on
#ROOT.gStyle.SetOptStat(0)


### Get the root file with the tree name
file_sig = uproot.open("training_root_files/WH_all_mA_2018_bdt_v1.root:tree")
file_bkg = uproot.open("training_root_files/TTbar_2018_bdt_v1.root:tree")

outputFile = ROOT.TFile.Open("compressed/compressed_BDT_all_mA_out_2018.root", "RECREATE")

outputTree1 = ROOT.TTree("Tree_sig_test", "Tree_sig_test")
outputTree2 = ROOT.TTree("Tree_sig_train", "Tree_sig_train")
outputTree3 = ROOT.TTree("Tree_bkg_test", "Tree_bkg_test")
outputTree4 = ROOT.TTree("Tree_bkg_train", "Tree_bkg_train")

var1 = array('f', [ 0 ])
var2 = array('f', [ 0 ])
var3 = array('f', [ 0 ])
var4 = array('f', [ 0 ])
var1w = array('f', [ 0 ])
var2w = array('f', [ 0 ])
var3w = array('f', [ 0 ])
var4w = array('f', [ 0 ])

outputTree1.Branch("branch_sig_test", var1, "branch_sig_test/F");
outputTree2.Branch("branch_sig_train", var2, "branch_sig_train/F");
outputTree3.Branch("branch_bkg_test", var3, "branch_bkg_test/F");
outputTree4.Branch("branch_bkg_train", var4, "branch_bkg_train/F");
outputTree1.Branch("branch_sig_test_weight", var1w, "branch_sig_test_weight/F");
outputTree2.Branch("branch_sig_train_weight", var2w, "branch_sig_train_weight/F");
outputTree3.Branch("branch_bkg_test_weight", var3w, "branch_bkg_test_weight/F");
outputTree4.Branch("branch_bkg_train_weight", var4w, "branch_bkg_train_weight/F");

### Store the branches as pandas dataframe
sig = file_sig.arrays(["evt_wgt","leppt","lepeta","b1eta","b2eta","pT1_by_mbb","pT2_by_mbb","Njets","pho1eta","pho2eta","pT1_by_mgg","pT2_by_mgg","leadbdeepjet","subleadbdeepjet","leadgMVA","subleadgMVA","delphi_gg","delphi_bb","delphi_bbgg","delphi_ggMET","yout"], library="pd")
bkg = file_bkg.arrays(["evt_wgt","leppt","lepeta","b1eta","b2eta","pT1_by_mbb","pT2_by_mbb","Njets","pho1eta","pho2eta","pT1_by_mgg","pT2_by_mgg","leadbdeepjet","subleadbdeepjet","leadgMVA","subleadgMVA","delphi_gg","delphi_bb","delphi_bbgg","delphi_ggMET","yout"], library="pd")

sig_corr = file_sig.arrays(["leppt","lepeta","b1eta","b2eta","pT1_by_mbb","pT2_by_mbb","Njets","pho1eta","pho2eta","pT1_by_mgg","pT2_by_mgg","leadbdeepjet","subleadbdeepjet","leadgMVA","subleadgMVA","delphi_gg","delphi_bb","delphi_bbgg","delphi_ggMET"], library="pd")
bkg_corr = file_bkg.arrays(["leppt","lepeta","b1eta","b2eta","pT1_by_mbb","pT2_by_mbb","Njets","pho1eta","pho2eta","pT1_by_mgg","pT2_by_mgg","leadbdeepjet","subleadbdeepjet","leadgMVA","subleadgMVA","delphi_gg","delphi_bb","delphi_bbgg","delphi_ggMET"], library="pd")
#print(sig)
#print(bkg)

### name of the variables used in training, will be used in feature importance plot
name_Var = ["leppt","lepeta","b1eta","b2eta","pT1_by_mbb","pT2_by_mbb","Njets","pho1eta","pho2eta","pT1_by_mgg","pT2_by_mgg","leadbdeepjet","subleadbdeepjet","leadgMVA","subleadgMVA","delphi_gg","delphi_bb","delphi_bbgg","delphi_ggMET"]

### Store the data as numpy array from the pandas dataframe
ntuple_datasetSig = sig.to_numpy()
ntuple_datasetBkg = bkg.to_numpy()
#print(ntuple_datasetSig.shape)
#print(ntuple_datasetBkg.shape)


XSig = ntuple_datasetSig[:,1:20]   ### pick elements from 0 to 19 
YSig = ntuple_datasetSig[:,20]
XBkg = ntuple_datasetBkg[:,1:20]
YBkg = ntuple_datasetBkg[:,20]
#print(XSig)
#print(YSig)
#print(XBkg)
#print(YBkg)

seed = 7
test_size = 0.33
X_trainSig, X_testSig, Y_trainSig, Y_testSig = train_test_split(XSig, YSig, test_size=test_size, random_state=seed, shuffle=1)
X_trainBkg, X_testBkg, Y_trainBkg, Y_testBkg = train_test_split(XBkg, YBkg, test_size=test_size, random_state=seed, shuffle=1)
print("X_trainSig shape: ",X_trainSig.shape)
#print(X_trainBkg)
#print(Y_trainSig)
#print(Y_trainBkg)
X_train = np.vstack((X_trainSig, X_trainBkg))
X_test = np.vstack((X_testSig,X_testBkg))
Y_train = np.hstack((Y_trainSig,Y_trainBkg))
Y_test = np.hstack((Y_testSig,Y_testBkg))
#print(X_train)
#print(X_test)
#print(Y_train)
#print(Y_test)

# fit model no training data
model = XGBClassifier(learning_rate=0.3, max_depth=6, eval_metric=auc, gamma=0.1, objective="binary:logistic")
gbm = model.fit(X_train, Y_train)
print(model)

# make predictions for test data
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)
y_pred_proba_train = model.predict_proba(X_train)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(Y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

############################################ ROC Curve

# Compute micro-average ROC curve and ROC area
fpr, tpr, _ = roc_curve(Y_test, y_pred)
roc_auc = auc(fpr, tpr)
gbm.get_booster().feature_names = name_Var
xgb.plot_importance(gbm)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()

##################################################
# Plot correlation matrix for signal and backgrounds
correlation_sig = sig_corr.corr()
seaborn.heatmap(correlation_sig).set(title = "sig correlation")
plt.show()

correlation_bkg = bkg_corr.corr()
seaborn.heatmap(correlation_bkg).set(title = "bkg correlation")
plt.show()

##################################################

roc_auc1 = auc(fpr, tpr)                                                                                                                                                                                  
plt.plot(fpr, tpr, lw=1, label='Test ROC (area = %0.2f)'%(roc_auc1))                                                                                                                                      
fpr2, tpr2, thresholds2 = roc_curve(Y_test, y_pred)                                                                                                                                            
roc_auc2 = auc(fpr2, tpr2)                                                                                                                                                                                  
plt.plot(fpr2, tpr2, lw=1, label='Train ROC (area = %0.2f)'%(roc_auc2))                                                                                                                                     
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')                                                                                                                                         
plt.xlim([-0.05, 1.05])                                                                                                                                                                                     
plt.ylim([-0.05, 1.05])                                                                                                                                                                                     
plt.xlabel('False Positive Rate')                                                                                                                                                                           
plt.ylabel('True Positive Rate')                                                                                                                                                                            
plt.title('Receiver operating characteristic')                                                                                                                                                              
plt.legend(loc="lower right")                                                                                                                                                                               
plt.grid()                                                                                                                                                                                                  
plt.show()

#################################################
# Plot the output BDT distribution of Signal and Background
y_pred_proba_real = y_pred_proba[:,1]
y_pred_sig = []
y_pred_bkg = []
for i in range (len(Y_test)):
    if Y_test[i]==1:
        y_pred_sig.append(y_pred_proba_real[i])
        var1[0] = y_pred_proba_real[i]
        var1w[0] = (59730*0.01)/(327965) + (59730*0.01)/(327967) + (59730*0.01)/(306201) + (59730*0.01)/(327956) + (59730*0.01)/(320042) + (59730*0.01)/(313111) + (59730*0.01)/(237890) + (59730*0.01)/(298271) + (59730*0.01)/(266598) 
        outputTree1.Fill()
    else:
        y_pred_bkg.append(y_pred_proba_real[i])
        var3[0] = y_pred_proba_real[i]
        var3w[0] = (59730*4.078)/(1.40e+06) + (59730*365.34)/(1.62e+08) + (59730*88.29)/(0.47e+08)
        outputTree3.Fill()

plt.hist([y_pred_sig, y_pred_bkg],color=['g','r'],histtype='step',bins=40, label=["Signal test", "Background test"])
plt.yscale("log")
plt.legend()
plt.show()

y_pred_proba_real_train = y_pred_proba_train[:,1]
y_pred_sig_train = []
y_pred_bkg_train = []
for i in range (len(Y_train)):
    if Y_train[i]==1:
        y_pred_sig_train.append(y_pred_proba_real_train[i])
        var2[0] = y_pred_proba_real_train[i]
        var2w[0] = (59730*0.01)/(655930) + (59730*0.01)/(655935) + (59730*0.01)/(612402) + (59730*0.01)/(655912) + (59730*0.01)/(640085) + (59730*0.01)/(626223) + (59730*0.01)/(475781) + (59730*0.01)/(596542) + (59730*0.01)/(533196) 
        outputTree2.Fill()
    else:
        y_pred_bkg_train.append(y_pred_proba_real_train[i])
        var4[0] = y_pred_proba_real_train[i]
        var4w[0] = (59730*4.078)/(2.8e+06) + (59730*365.34)/(3.24e+08) + (59730*88.29)/(0.94e+08)
        outputTree4.Fill()

plt.hist([y_pred_sig_train, y_pred_bkg_train],color=['g','r'],histtype='step',bins=40, label=["Signal train", "Background train"])
plt.yscale("log")
plt.legend()
plt.show()

outputFile.cd()
outputTree1.Write()
outputTree2.Write()
outputTree3.Write()
outputTree4.Write()
outputFile.Close()

##################################################

##################################################

# Plot the different BDT input variables
plt.rcParams['text.usetex'] = True
plt.rcParams["figure.figsize"] = [7.50, 7.50]
plt.rcParams["figure.autolayout"] = True

Nbins=[30,30,30,30,10,5,20,30,30,10,5,20,20,40,40,40,40,40,40]
Range=[[0,300],[-3,3],[-3,3],[-3,3],[0,10],[0,5],[0,20],[-3,3],[-3,3],[0,10],[0,5],[0,1],[0,1],[-1,1],[-1,1],[0,4],[0,4],[0,4],[0,4]]
Ncols = 4
Nrows = 5
xAxisLabel = ["leppt","lepeta","b1eta","b2eta","pT1/Mbb","pT2/Mbb","Njets","pho1eta","pho2eta","pT1/Mgg","pT2/Mgg","leaddeepjet","subleaddeepjet","leadPhoMVA","subleadPhoMVA","delphi_gg","delphi_bb","delphi_bbgg","delphi_ggMET"]

figure, axis = plt.subplots(Nrows, Ncols)
counter=0
for i in range(Nrows):
    for j in range(Ncols):
        if (i+j+counter< len(Nbins)):
            axis[i,j].hist([X_testSig[:,i+j+counter], X_testBkg[:,i+j+counter]], color=['g','r'], histtype='stepfilled', bins=Nbins[i+j+counter], label=["Signal", "Background"], density=[True,True], range=Range[i+j+counter], alpha = 0.5)
            axis[i,j].set_xlabel(xAxisLabel[i+j+counter], fontsize=20)
            axis[i,j].set_ylabel("AU")
            axis[i,j].legend()
    counter = counter+Ncols-1
#plt.hist([X_testSig[:,2], X_testBkg[:,2],],color=['g','r'],histtype='step',bins=60, label=["Signal", "Background"], density=[True,True], range=[0,60])
plt.subplots_adjust(left=0.1, bottom=0.12, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
plt.show()
