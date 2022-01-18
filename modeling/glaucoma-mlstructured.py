import numpy as np
import csv
import sklearn
import sklearn.model_selection
import sklearn.linear_model
import sklearn.impute
import sklearn.preprocessing
import sklearn.ensemble
import sklearn.metrics
import sys
from collections import Counter

print('Scikit Learn version:', sklearn.__version__)
print('Python version:', sys.version)
print('Numpy version:', np.__version__)

# load data from file and convert to numpy arrays
csv.field_size_limit(1310720)
embedding_dimension = 300
width = 1000
extrinsic_file = 'extrinsicdatabalanced.csv'

print('loading CSV from', extrinsic_file)
with open(extrinsic_file, 'r') as f:
    r = csv.reader(f)
    i = 0
    structuredArray = []
    outputArray = []
    tokenArray = []
    for row in r:
        i += 1
        if '' in row:
            continue 
        structuredinputs = np.array([float(j) for j in row[3:-1]])
        output = np.array([int(row[-1])])
        
        structuredArray.append(structuredinputs)
        outputArray.append(output)

outputArray = np.array(outputArray).reshape((len(outputArray),))
structuredArray = np.array(structuredArray)
print('Structured Array:', structuredArray.shape)
print('Output Array:', outputArray.shape)

# separate into training, validation, and holdout sets matching what was used for unstructured/combo models
Xholdout = structuredArray[0:500]
Xvalidation = structuredArray[500:900]
Xtrain = structuredArray[900:]
Yholdout = outputArray[0:500]
Yvalidation = outputArray[500:900]
Ytrain = outputArray[900:]

print(Xtrain.shape, Ytrain.shape)
print(Xholdout.shape, Yholdout.shape)
print(Xvalidation.shape, Yvalidation.shape)

# grid search
# l1/LASSO classifier hyperparameter tuning
Cvals = [0.00001, 0.001, 0.1, 1, 10, 1000, 100000]
l1classifiers = {}
for C in Cvals:
    # train model
    l1classifier = sklearn.linear_model.LogisticRegression(solver='saga', penalty = 'l1', C=C, max_iter=5000).fit(Xtrain, Ytrain)
    l1classifiers[C] = l1classifier

    # score model on train and validation
    train_accuracy = l1classifier.score(Xtrain, Ytrain)
    validation_accuracy = l1classifier.score(Xvalidation, Yvalidation)
    validation_auc = sklearn.metrics.roc_auc_score(Yvalidation, l1classifier.predict_proba(Xvalidation)[:,1])
    print('L1, C:', f'{C:5}', 'Training accuracy:', f'{train_accuracy:6.4f}', 'Validation accuracy:', f'{validation_accuracy:6.4f}', 'Validation AUC:', f'{validation_auc:6.4f}')

# l2/RIDGE classifier hyperparameter tuning
Cvals = [0.00001, 0.001, 0.1, 1, 10, 1000, 100000]
l2classifiers = {}
for C in Cvals:
    # train model
    l2classifier = sklearn.linear_model.LogisticRegression(solver='saga', penalty = 'l2', C=C, max_iter=5000).fit(Xtrain, Ytrain)
    l2classifiers[C] = l2classifier

    # score model on train and validation
    train_accuracy = l2classifier.score(Xtrain, Ytrain)
    validation_accuracy = l2classifier.score(Xvalidation, Yvalidation)
    validation_auc = sklearn.metrics.roc_auc_score(Yvalidation, l2classifier.predict_proba(Xvalidation)[:,1])
    print('L2, C:', f'{C:5}', 'Training accuracy:', f'{train_accuracy:6.4f}', 'Validation accuracy:', f'{validation_accuracy:6.4f}', 'Validation AUC:', f'{validation_auc:6.4f}')

# elasticnet classifier hyperparameter tuning
Cvals = [0.00001, 0.001, 0.1, 1, 10, 1000, 100000]
l1ratios = [0.1, 0.25, 0.5, 0.75, 0.9]
en_classifiers = {}
for C in Cvals:
    for l1ratio in l1ratios:
        # train model
        en_classifier = sklearn.linear_model.LogisticRegression(solver='saga', penalty = 'elasticnet', l1_ratio = l1ratio, C=C, max_iter=5000).fit(Xtrain, Ytrain)
        en_classifiers[(C,l1ratio)] = en_classifier

        # score model on train and validation
        train_accuracy = en_classifier.score(Xtrain, Ytrain)
        validation_accuracy = en_classifier.score(Xvalidation, Yvalidation)
        validation_auc = sklearn.metrics.roc_auc_score(Yvalidation, en_classifier.predict_proba(Xvalidation)[:,1])
        print('ElasticNet: (C =', f'{C:5}',', l1ratio =', f'{l1ratio:4.2f}', 'Training accuracy:', f'{train_accuracy:6.4f}', 'Validation accuracy:', f'{validation_accuracy:6.4f}', 'Validation AUC:', f'{validation_auc:6.4f}')
	
# Gradient Boosted Tree hyperparameter tuning
num_estimators = [25, 50, 100, 200, 800]
max_depths = [2, 3, 6, 12]
min_samples_splits = [2, 6, 20]
min_samples_leafs = [1, 3, 10]
gbt_classifiers = {}
for num_estimator in num_estimators:
    for max_depth in max_depths:
        for min_samples_split in min_samples_splits:
            for min_samples_leaf in min_samples_leafs:
                # train model
                gbt_classifier = sklearn.ensemble.GradientBoostingClassifier(n_estimators = num_estimator, max_depth = max_depth, max_features = 'sqrt', min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf).fit(Xtrain, Ytrain)
                gbt_classifiers[(num_estimator,max_depth, min_samples_split, min_samples_leaf)] = gbt_classifier

                # score model on train and validation
                train_accuracy = gbt_classifier.score(Xtrain, Ytrain)
                validation_accuracy = gbt_classifier.score(Xvalidation, Yvalidation)
                validation_auc = sklearn.metrics.roc_auc_score(Yvalidation, gbt_classifier.predict_proba(Xvalidation)[:,1])
                print('GBTrees num_estimator =', f'{num_estimator:5}',', max_depth =', f'{max_depth:2}', ', min_samples_split =', f'{min_samples_split:2}', ', min_samples_leaf =', f'{min_samples_leaf:2}','Training accuracy:', f'{train_accuracy:6.4f}', 'Validation accuracy:', f'{validation_accuracy:6.4f}', 'Validation AUC:', f'{validation_auc:6.4f}')

# for best models, tune F1 on validation set
models = {
    'l1classifier': l1classifiers[0.1],
    'l2classifier': l2classifiers[0.1],
    'en_classifier': en_classifiers[(0.1, 0.25)],
    'gbt_classifier': gbt_classifiers[(25, 3, 6, 1)]
}
for model in models:
    f1_scores = Counter()
    for threshold in [i*0.05 for i in range(1,20)]:
        Ypredictions = (models[model].predict_proba(Xvalidation)[:,1] >= threshold)
        f1 = sklearn.metrics.f1_score(Yvalidation, Ypredictions)
        f1_scores[threshold] = f1
  
    best = f1_scores.most_common()[0]
    holdoutf1 = sklearn.metrics.f1_score(Yholdout, (models[model].predict_proba(Xholdout)[:,1] >= best[0]))
    print('for', model, 'best validation f1 achieved at threshold =', f'{best[0]:4.2f}', 'of', f'{best[1]:6.4f}', 'resulting in f1 on holdout set of:', f'{holdoutf1:6.4f}')

# for best models, score on the holdout set -- accuracy & AUC
print('L1 classifier on Holdout set')
print(models['l1classifier'].score(Xholdout,Yholdout))
print(sklearn.metrics.roc_auc_score(Yholdout, l1classifiers[0.1].predict_proba(Xholdout)[:,1]))

print('\nL2 classifier on Holdout set')
print(models['l2classifier'].score(Xholdout,Yholdout))
print(sklearn.metrics.roc_auc_score(Yholdout, l2classifiers[0.1].predict_proba(Xholdout)[:,1]))

print('\nElasticnet classifier on Holdout set')
print(models['en_classifier'].score(Xholdout,Yholdout))
print(sklearn.metrics.roc_auc_score(Yholdout, en_classifiers[(0.1, 0.25)].predict_proba(Xholdout)[:,1]))

print('\nGBTree on Holdout set')
print(models['gbt_classifier'].score(Xholdout,Yholdout))
print(sklearn.metrics.roc_auc_score(Yholdout, gbt_classifiers[(25, 3, 6, 1)].predict_proba(Xholdout)[:,1]))

