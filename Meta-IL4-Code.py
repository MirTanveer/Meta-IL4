#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import RepeatedStratifiedKFold


# Import Data
file_positive= pd.read_csv("/path")
file_negative= pd.read_csv("/path")

#Merging Files
csv_file_list = [file_negative, file_positive]
list_of_dataframes = []
for filename in csv_file_list:
    list_of_dataframes.append((filename))
merged_df1 = pd.concat(list_of_dataframes)
X = merged_df1.iloc[:,1:-1].values
y = merged_df1.iloc[:,-1].values

#Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Using SMOTE-ENN
#from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
sm=SMOTEENN(random_state=100)
X_sm, y_sm= sm.fit_resample(X_train,y_train)
#counter = Counter(y)
#print(counter)
#counter1=Counter(y_sm)
#print(counter1)

def evaluate_model_test(model, X_test, y_test):
    from sklearn import metrics

    # Predict Test Data 
    y_pred = model.predict(X_test)

    # Calculate accuracy, precision, recall, f1-score, and kappa score
    acc = metrics.accuracy_score(y_test, y_pred)
    prec = metrics.precision_score(y_test, y_pred)
    rec = metrics.recall_score(y_test, y_pred)
    f1 = metrics.f1_score(y_test, y_pred)

    # Calculate area under curve (AUC)
    y_pred_proba = model.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    
    #MCC
    mcc=matthews_corrcoef(y_test, model.predict(X_test))
    
    # Display confussion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)
    total=sum(sum(cm))
    
    #accuracy=(cm[0,0]+cm[1,1])/total
    sen = cm[0,0]/(cm[0,0]+cm[0,1])
    spec= cm[1,1]/(cm[1,0]+cm[1,1])

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'mcc':mcc,
            'fpr': fpr, 'tpr': tpr, 'auc': auc, 'cm': cm, 'sen': sen, 'spec':spec}


def evaluate_model_train(model, X_train, y_train):
    conf_matrix_list_of_arrays = []
    mcc_array=[]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)  
    lst_accu = []
    
    prec_train=np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='precision'))
    recall_train=np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='recall'))
    f1_train=np.mean(cross_val_score(model, X_train, y_train, cv=5, scoring='f1'))
    
    for train_index, test_index in cv.split(X_train, y_train): 
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index] 
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index] 
        model.fit(X_train_fold, y_train_fold) 
        lst_accu.append(model.score(X_test_fold, y_test_fold))
        acc=np.mean(lst_accu)
        
        conf_matrix = confusion_matrix(y_test_fold, model.predict(X_test_fold))
        conf_matrix_list_of_arrays.append(conf_matrix)
        cm = np.mean(conf_matrix_list_of_arrays, axis=0)
        mcc_array.append(matthews_corrcoef(y_test_fold, model.predict(X_test_fold)))
        mcc=np.mean(mcc_array, axis=0)
        
        
    total=sum(sum(cm))
    accuracy=(cm[0,0]+cm[1,1])/total
    sensitivity = cm[0,0]/(cm[0,0]+cm[0,1])
    specificity = cm[1,1]/(cm[1,0]+cm[1,1])
    
    
    return {'prec_train': prec_train, 'recall_train': recall_train, 'f1_train': f1_train, 'cm': cm, 'mcc': mcc,'acc':acc,
           'sen':sensitivity,'spec':specificity, 'acc':acc}

def roc_curve(model1, model2, model3, model4, X_test, y_test):
    from sklearn import metrics
    y_pred1= (model1.predict_proba(X_test))[:,1]
    y_pred2= (model2.predict_proba(X_test))[:,1]
    y_pred3= (model3.predict_proba(X_test))[:,1]
    y_pred4= (model4.predict_proba(X_test))[:,1]
    
    x1="RF"
    x2="XGB"
    x3="LGBM"
    x4="ETC"
    
    fpr1, tpr1,threshold  = metrics.roc_curve(y_test,  y_pred1)
    auc1 = metrics.roc_auc_score(y_test, y_pred1)
   
    
    fpr2, tpr2,threshold  = metrics.roc_curve(y_test,  y_pred2)
    auc2 = metrics.roc_auc_score(y_test, y_pred2)
    
    
    fpr3, tpr3,threshold  = metrics.roc_curve(y_test,  y_pred3)
    auc3 = metrics.roc_auc_score(y_test, y_pred3)

    
    fpr4, tpr4,threshold  = metrics.roc_curve(y_test,  y_pred4)
    auc4 = metrics.roc_auc_score(y_test, y_pred4)
    
    
    #create ROC curve
    plt.figure(figsize=(8,8)) 
    plt.plot(fpr1,tpr1,label= f"{x1} (AUC={auc1: .3})")
    plt.plot(fpr2,tpr2,label= f"{x2} (AUC={auc2: .3})")
    plt.plot(fpr3,tpr3,label= f"{x3} (AUC={auc3: .3})")
    plt.plot(fpr4,tpr4,label= f"{x4} (AUC={auc4: .3})")
    
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.legend(loc=4, fontsize=14)







#Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=146, criterion='gini', max_depth=None, min_samples_split=2,
                                    min_samples_leaf=4, min_weight_fraction_leaf=0.0, max_features='sqrt', 
                                    max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, 
                                    n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None,
                                    ccp_alpha=0.0, max_samples=None)


# Evaluate Model on Training data
train_eval = evaluate_model_train(classifier, X_train, y_train)
print("Confusion Matrix is: ", train_eval['cm'])
print ('Accuracy : ', train_eval['acc'])
print('Sensitivity : ', train_eval['sen'])
print('Specificity : ', train_eval['spec'])
print("Mean of Matthews Correlation Coefficient is: ", train_eval['mcc'])
print("The Precision value is: ", train_eval['prec_train'])
print("The Recall value is: ", train_eval['recall_train'])
print("The F1 score is: ", train_eval['f1_train'])


# Evaluate Model on Testing data
dtc_eval = evaluate_model_test(classifier, X_test, y_test)

# Print result
print('Accuracy:', dtc_eval['acc'])
print('Precision:', dtc_eval['prec'])
print('Recall:', dtc_eval['rec'])
print('F1 Score:', dtc_eval['f1'])
print('Area Under Curve:', dtc_eval['auc'])
print('Sensitivity : ', dtc_eval['sen'])
print('Specificity : ', dtc_eval['spec'])
print('MCC Score : ', dtc_eval['mcc'])
print('Confusion Matrix:\n', dtc_eval['cm'])


#XGB
# Training XGBoost on the Training set
from xgboost import XGBClassifier
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Evaluate Model on Training data
train_eval = evaluate_model_train(xgb, X_train, y_train)
print("Confusion Matrix is: ", train_eval['cm'])
print ('Accuracy : ', train_eval['acc'])
print('Sensitivity : ', train_eval['sen'])
print('Specificity : ', train_eval['spec'])
print("Mean of Matthews Correlation Coefficient is: ", train_eval['mcc'])
print("The Precision value is: ", train_eval['prec_train'])
print("The Recall value is: ", train_eval['recall_train'])
print("The F1 score is: ", train_eval['f1_train'])



# Evaluate Model on Testing data
dtc_eval = evaluate_model_test(xgb, X_test, y_test)

# Print result
print('Accuracy:', dtc_eval['acc'])
print('Precision:', dtc_eval['prec'])
print('Recall:', dtc_eval['rec'])
print('F1 Score:', dtc_eval['f1'])
print('Area Under Curve:', dtc_eval['auc'])
print('Sensitivity : ', dtc_eval['sen'])
print('Specificity : ', dtc_eval['spec'])
print('MCC Score : ', dtc_eval['mcc'])
print('Confusion Matrix:\n', dtc_eval['cm'])


#LGBM
import lightgbm as lgb
lgbm = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, n_estimators=250)

# Evaluate Model on Training data
train_eval = evaluate_model_train(lgbm, X_train, y_train)
print("Confusion Matrix is:\n", train_eval['cm'])
print ('Accuracy : ', train_eval['acc'])
print('Sensitivity : ', train_eval['sen'])
print('Specificity : ', train_eval['spec'])
print("Matthews Correlation Coefficient is: ", train_eval['mcc'])
print("Precision value is: ", train_eval['prec_train'])
print("Recall value is: ", train_eval['recall_train'])
print("F1 score is: ", train_eval['f1_train'])


# Evaluate Model on Testing data
dtc_eval = evaluate_model_test(lgbm, X_test, y_test)
print('Accuracy:', dtc_eval['acc'])
print('Precision:', dtc_eval['prec'])
print('Recall:', dtc_eval['rec'])
print('F1 Score:', dtc_eval['f1'])
print('Area Under Curve:', dtc_eval['auc'])
print('Sensitivity : ', dtc_eval['sen'])
print('Specificity : ', dtc_eval['spec'])
print('MCC Score : ', dtc_eval['mcc'])
print('Confusion Matrix:\n', dtc_eval['cm'])


#ExtraTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
etc=ExtraTreesClassifier(n_estimators=800, criterion='gini', max_depth=None, min_samples_split=2)
etc.fit(X_train, y_train)


# Evaluate Model on Training data
train_eval = evaluate_model_train(etc, X_train, y_train)
print("Confusion Matrix is:\n", train_eval['cm'])
print ('Accuracy : ', train_eval['acc'])
print('Sensitivity : ', train_eval['sen'])
print('Specificity : ', train_eval['spec'])
print("Matthews Correlation Coefficient is: ", train_eval['mcc'])
print("Precision value is: ", train_eval['prec_train'])
print("Recall value is: ", train_eval['recall_train'])
print("F1 score is: ", train_eval['f1_train'])



# Evaluate Model on Testing data
dtc_eval=evaluate_model_test(etc, X_test, y_test)
print('Accuracy:', dtc_eval['acc'])
print('Precision:', dtc_eval['prec'])
print('Recall:', dtc_eval['rec'])
print('F1 Score:', dtc_eval['f1'])
print('Area Under Curve:', dtc_eval['auc'])
print('Sensitivity : ', dtc_eval['sen'])
print('Specificity : ', dtc_eval['spec'])
print('MCC Score : ', dtc_eval['mcc'])
print('Confusion Matrix:\n', dtc_eval['cm'])


#Ensemble Learning
# GaussainProcessClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
gclf = GaussianProcessClassifier()
# Defining meta-classifier
from mlxtend.classifier import StackingClassifier
clf_stack = StackingClassifier(classifiers =[etc, lgbm, xgb,classifier], meta_classifier = gclf, use_probas = True, use_features_in_secondary = True)


# Evaluate Meta-Model on Training data
train_eval = evaluate_model_train(clf_stack, X_train, y_train)
print("Confusion Matrix is:\n", train_eval['cm'])
print ('Accuracy : ', train_eval['acc'])
print('Sensitivity : ', train_eval['sen'])
print('Specificity : ', train_eval['spec'])
print("Matthews Correlation Coefficient is: ", train_eval['mcc'])
print("Precision value is: ", train_eval['prec_train'])
print("Recall value is: ", train_eval['recall_train'])
print("F1 score is: ", train_eval['f1_train'])


# Evaluate Meta-Model on Testing data
dtc_eval = evaluate_model_test(clf_stack, X_test, y_test)
print('Accuracy:', dtc_eval['acc'])
print('Precision:', dtc_eval['prec'])
print('Recall:', dtc_eval['rec'])
print('F1 Score:', dtc_eval['f1'])
print('Area Under Curve:', dtc_eval['auc'])
print('Sensitivity : ', dtc_eval['sen'])
print('Specificity : ', dtc_eval['spec'])
print('MCC Score : ', dtc_eval['mcc'])
print('Confusion Matrix:\n', dtc_eval['cm'])

