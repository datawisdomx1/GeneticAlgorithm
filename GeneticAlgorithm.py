#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 17:02:13 2021

@author: nitinsinghal
"""

# Genetic algorithm - Santander Customer Satisfaction Kaggle problem

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from genetic_selection import GeneticSelectionCV
import warnings
warnings.filterwarnings("ignore")

# Load the data

train_data = pd.read_csv('./Santander Customer Satisfaction - TRAIN.csv')
test_data = pd.read_csv('./Santander Customer Satisfaction - TEST-Without TARGET.csv')

# Perform EDA - see the data types, content, statistical properties
print(train_data.describe())
print(train_data.info())
print(train_data.head(5))

print(test_data.describe())
print(test_data.info())
print(test_data.head(5))

# Check for % of classification categories
print('traintarget0 count: ', (train_data['TARGET']==0).sum())
print('traintarget1 count: ', (train_data['TARGET']==1).sum())
print('traintarget0 pct: %.2f' %((train_data['TARGET']==0).sum()/train_data['TARGET'].count() *100))
print('traintarget1 pct: %.2f' %((train_data['TARGET']==1).sum()/train_data['TARGET'].count() *100))

emptycols = train_data.sum()==0
ec = emptycols[emptycols==True].index
print(ec)

# Perform data wrangling - remove duplicate values and clean null values
train_data.drop_duplicates(inplace=True)
test_data.drop_duplicates(inplace=True)

train_data.fillna(0, inplace=True)
test_data.fillna(0, inplace=True)

X_train_minor = train_data[train_data['TARGET']==1]
X_train_major = train_data[train_data['TARGET']==0]

# Increase Majority class in multiples of 3008, as Minority class has 3008 samples
X_train_major_1 = X_train_major.head(6016)
X_train_combined_1 = X_train_minor.append(X_train_major_1, ignore_index=True)

# Setup the traing and test X, y datasets
X_train = X_train_combined_1.iloc[:,1:-1].values
y_train = X_train_combined_1.iloc[:,-1].values
X_test = test_data.iloc[:,1:].values

# Scale the X data (input feature data used for training the model)
# This is done as some columns have large numeric vaues, while most have 0 or 1
# This prevents giving large scale values higher weights in the model's
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Using RandomForestClassifier as the estimator for GeneticSearch FeatureSelection
estimator = RandomForestClassifier()

selector = GeneticSelectionCV(estimator,
                              cv=5,
                              verbose=1,
                              scoring="accuracy",
                              max_features=65,
                              n_population=50,
                              crossover_proba=0.5,
                              mutation_proba=0.2,
                              n_generations=5,
                              crossover_independent_proba=0.5,
                              mutation_independent_proba=0.05,
                              tournament_size=3,
                              n_gen_no_change=10,
                              caching=True,
                              n_jobs=-1)

selector = selector.fit(X_train,y_train)

print(selector.support_)
print('cv score:', selector.generation_scores_[-1])

df_X_train = pd.DataFrame(X_train)
selected_features = list(df_X_train.columns[selector.support_])
print("No of Selected Features:  ", len(selected_features))
print("Selected Features:  ", selected_features)

X_train_selfeat = df_X_train.filter(items=selected_features)

rf_paramgrid = {'n_estimators' : [300,500],
                'min_samples_split' : [10,20],
                'min_samples_leaf' : [2,3],
                'max_features' : ['auto','log2']}

# Do hyperparameter tuning using GA
cv = EvolutionaryAlgorithmSearchCV(estimator=RandomForestClassifier(),
                                   params=rf_paramgrid,
                                   scoring="accuracy",
                                   cv=StratifiedKFold(n_splits=2),
                                   verbose=1,
                                   population_size=50,
                                   gene_mutation_prob=0.10,
                                   gene_crossover_prob=0.5,
                                   tournament_size=3,
                                   generations_number=5,
                                   n_jobs=1)

cv.fit(X_train_selfeat, y_train)
cv.best_score_, cv.best_params_

#Construct a Random Forest Classifier on data
clf=RandomForestClassifier(**cv.best_params_)
clf.set_params(**cv.best_params_)
RF_text = clf.fit(X_train_selfeat,y_train)
print("Accuracy score (training): {0:.6f}".format(clf.score(X_train_selfeat,y_train)))


# Using GradientBoostingClassifier as the estimator for GeneticSearch FeatureSelection
estimator = GradientBoostingClassifier()

selector = GeneticSelectionCV(estimator,
                              cv=5,
                              verbose=1,
                              scoring="accuracy",
                              max_features=26,
                              n_population=65,
                              crossover_proba=0.5,
                              mutation_proba=0.2,
                              n_generations=5,
                              crossover_independent_proba=0.5,
                              mutation_independent_proba=0.05,
                              tournament_size=3,
                              n_gen_no_change=10,
                              caching=True,
                              n_jobs=-1)

selector = selector.fit(X_train,y_train)

print(selector.support_)
print('cv score:', selector.generation_scores_[-1])

df_X_train = pd.DataFrame(X_train)
selected_features = list(df_X_train.columns[selector.support_])
print("No of Selected Features:  ", len(selected_features))
print("Selected Features:  ", selected_features)

X_train_selfeat = df_X_train.filter(items=selected_features)
gb_paramgrid = {'n_estimators' : [200,300],
                'learning_rate' : [0.01, 0.1],
                'max_features' : ['auto','log2']}

# Do hyperparameter tuning using GA
cv = EvolutionaryAlgorithmSearchCV(estimator=GradientBoostingClassifier(),
                                   params=gb_paramgrid,
                                   scoring="accuracy",
                                   cv=StratifiedKFold(n_splits=2),
                                   verbose=1,
                                   population_size=50,
                                   gene_mutation_prob=0.10,
                                   gene_crossover_prob=0.5,
                                   tournament_size=3,
                                   generations_number=5,
                                   n_jobs=1)

cv.fit(X_train_selfeat, y_train)
cv.best_score_, cv.best_params_

#Construct a GradientBoostingClassifier on data
clf=GradientBoostingClassifier()
clf.set_params(**cv.best_params_)
GB_text = clf.fit(X_train_selfeat,y_train)
print("Accuracy score (training): {0:.6f}".format(clf.score(X_train_selfeat,y_train)))


#Feature selection - Filter Type - KBest
for i in (25, 50):
    kbest_selector = SelectKBest(k=i)
    kbestfeatures = kbest_selector.fit_transform(X_train,y_train)
    feature_names_out = kbest_selector.get_support(indices=True)
    print('KBest Features: ', feature_names_out)
    
    print('KBest Results k = ', i)
    #Construct a Random Forest Classifier on data
    clf=RandomForestClassifier()
    RF_text = clf.fit(X_train_selfeat,y_train)
    print("Accuracy score (training): {0:.6f}".format(clf.score(X_train_selfeat,y_train)))
    
    #Construct a GradientBoostingClassifier on data
    clf=GradientBoostingClassifier()
    GB_text = clf.fit(X_train_selfeat,y_train)
    print("Accuracy score (training): {0:.6f}".format(clf.score(X_train_selfeat,y_train)))
    

# 3. Build 4 models with 2 classifiers and Top25, Top50 merged features
# Do hyperparameter tuning using GA

# Merged Top 25 features from RandomForest T25, GradientBoosting T25, KBest25 
# Took top 8 in each removing common one's and adding others
selected_features25 = [1,	26,	30,	36,	58,	63,	75,	92,	119,	128,	
                       12,	46,	61,	70,	89,	167,	182,	184,	207,	24,	27,	29,	31,	32,	33]

# Merged Top 50 features from RandomForest T50, GradientBoosting T50, KBest 50 
# Took top 20 in each removing common one's and adding others
selected_features50 = [1,	6,	12,	20,	22,	24,	27,	29,	33,	37,	43,	45,	65,	70,	71,	75,	102, 127,
                       128 ,	132,	136,	138,	155,	165,	38,	47,	66,	131,	149,	162,	
                       164,	174,	176,	182,	184,	192,	197,	203,	242,	294,	
                       312,	316,	327,	337,	49,	63,	76,	88,	90,	93]

# 3. Use RandomForestClassifier
df_X_train = pd.DataFrame(X_train)
X_train_selfeat = df_X_train.filter(items=selected_features50)

# Hyperparameter tuning using RandomForestClassifier
rf_paramgrid = {'n_estimators' : [300,500],
                'min_samples_split' : [10,20],
                'min_samples_leaf' : [2,3],
                'max_features' : ['auto','log2']}

cv = EvolutionaryAlgorithmSearchCV(estimator=RandomForestClassifier(),
                                   params=rf_paramgrid,
                                   scoring="accuracy",
                                   cv=StratifiedKFold(n_splits=2),
                                   verbose=1,
                                   population_size=50,
                                   gene_mutation_prob=0.10,
                                   gene_crossover_prob=0.5,
                                   tournament_size=3,
                                   generations_number=5,
                                   n_jobs=1)

cv.fit(X_train_selfeat, y_train)
cv.best_score_, cv.best_params_

#Construct a Random Forest Classifier on data
clf=RandomForestClassifier(**cv.best_params_)
clf.set_params(**cv.best_params_)
RF_text = clf.fit(X_train_selfeat,y_train)
print("Accuracy score (training): {0:.6f}".format(clf.score(X_train_selfeat,y_train)))

# 3. Use GradientBoostingClassifier
df_X_train = pd.DataFrame(X_train)
X_train_selfeat = df_X_train.filter(items=selected_features50)

gb_paramgrid = {'n_estimators' : [200,300],
                'learning_rate' : [0.01, 0.1],
                'max_features' : ['auto','log2']}

# Do hyperparameter tuning using GA
cv = EvolutionaryAlgorithmSearchCV(estimator=GradientBoostingClassifier(),
                                   params=gb_paramgrid,
                                   scoring="accuracy",
                                   cv=StratifiedKFold(n_splits=2),
                                   verbose=1,
                                   population_size=50,
                                   gene_mutation_prob=0.10,
                                   gene_crossover_prob=0.5,
                                   tournament_size=3,
                                   generations_number=5,
                                   n_jobs=1)

cv.fit(X_train_selfeat, y_train)
cv.best_score_, cv.best_params_

#Construct a GradientBoostingClassifier on data
clf=GradientBoostingClassifier()
clf.set_params(**cv.best_params_)
GB_text = clf.fit(X_train_selfeat,y_train)
print("Accuracy score (training): {0:.6f}".format(clf.score(X_train_selfeat,y_train)))


# 4. Now build 4 models using the 2 classifiers with best hyperparameters, 
# with merged top 25 and 50 features
# Can use all training data - but too large. Will only use 10% of the data

# Increase training sample to all data now

# Setup the traing and test X, y datasets
X_train = train_data.iloc[:,1:-1].values
y_train = train_data.iloc[:,-1].values
X_test = test_data.iloc[:,1:].values

# Scale the X data (input feature data used for training the model)
# This is done as some columns have large numeric vaues, while most have 0 or 1
# This prevents giving large scale values higher weights in the model's
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

df_X_train = pd.DataFrame(X_train)
X_train_selfeat = df_X_train.filter(items=selected_features25)
df_X_test = pd.DataFrame(X_test)
X_test_selfeat = df_X_test.filter(items=selected_features25)

# Top25 features and hyperparameters
#Construct a Random Forest Classifier on data
clf=RandomForestClassifier(n_estimators= 500, min_samples_split= 20, min_samples_leaf= 2, max_features= 'auto')
RF_text = clf.fit(X_train_selfeat,y_train)
print("Accuracy score (training): {0:.6f}".format(clf.score(X_train_selfeat,y_train)))
rf_predictions = clf.predict_proba(X_test_selfeat)
rf_predictions1 = rf_predictions[:,1]

# Output the predicted y values into a csv file, submit in kaggle santander competition and check score
df_result = pd.DataFrame()
df_result.index = test_data['ID']
df_result['TARGET'] = rf_predictions1
df_result.to_csv('./Prob2_RFPredTop25MergedTuned.csv')


#Construct a GradientBoostingClassifier on data
clf=GradientBoostingClassifier(n_estimators= 300, learning_rate= 0.1, max_features= 'log2')
GB_text = clf.fit(X_train_selfeat,y_train)
print("Accuracy score (training): {0:.6f}".format(clf.score(X_train_selfeat,y_train)))
gb_predictions = clf.predict_proba(X_test_selfeat)
gb_predictions1 = gb_predictions[:,1]

# Output the predicted y values into a csv file, submit in kaggle santander competition and check score
df_result = pd.DataFrame()
df_result.index = test_data['ID']
df_result['TARGET'] = gb_predictions1
df_result.to_csv('./Prob2_GBPredTop25MergedTuned.csv')


# Top50 features and hyperparameters
df_X_train = pd.DataFrame(X_train)
X_train_selfeat = df_X_train.filter(items=selected_features50)
df_X_test = pd.DataFrame(X_test)
X_test_selfeat = df_X_test.filter(items=selected_features50)

#Construct a Random Forest Classifier on data
clf=RandomForestClassifier(n_estimators= 500, min_samples_split= 10, min_samples_leaf= 3, max_features= 'auto')
RF_text = clf.fit(X_train_selfeat,y_train)
print("Accuracy score (training): {0:.6f}".format(clf.score(X_train_selfeat,y_train)))
rf_predictions = clf.predict_proba(X_test_selfeat)
rf_predictions1 = rf_predictions[:,1]

# Output the predicted y values into a csv file, submit in kaggle santander competition and check score
df_result = pd.DataFrame()
df_result.index = test_data['ID']
df_result['TARGET'] = rf_predictions1
df_result.to_csv('./Prob2_RFPredTop50MergedTuned.csv')

#Construct a GradientBoostingClassifier on data
clf=GradientBoostingClassifier(n_estimators= 300, learning_rate= 0.1, max_features= 'log2')
GB_text = clf.fit(X_train_selfeat,y_train)
print("Accuracy score (training): {0:.6f}".format(clf.score(X_train_selfeat,y_train)))
gb_predictions = clf.predict_proba(X_test_selfeat)
gb_predictions1 = gb_predictions[:,1]

# Output the predicted y values into a csv file, submit in kaggle santander competition and check score
df_result = pd.DataFrame()
df_result.index = test_data['ID']
df_result['TARGET'] = gb_predictions1
df_result.to_csv('./Prob2_GBPredTop50MergedTuned.csv')


# Base model results without any GA feature selection, hyperparameter tuning or minority sampling
clf=RandomForestClassifier()
RF_text = clf.fit(X_train,y_train)
print('Base RandomForestClassifier Results: ')
print("Accuracy score (training): {0:.6f}".format(clf.score(X_train,y_train)))
rf_predictions = clf.predict_proba(X_test)
rf_predictions1 = rf_predictions[:,1]

# Output the predicted y values into a csv file, submit in kaggle santander competition and check score
df_result = pd.DataFrame()
df_result.index = test_data['ID']
df_result['TARGET'] = rf_predictions1
df_result.to_csv('./Prob2_BaseRFPred.csv')

clf=GradientBoostingClassifier()
gb_text = clf.fit(X_train,y_train)
print('Base GradientBoostingClassifier Results: ')
print("Accuracy score (training): {0:.6f}".format(clf.score(X_train,y_train)))
gb_predictions = clf.predict_proba(X_test)
gb_predictions1 = gb_predictions[:,1]

# Output the predicted y values into a csv file, submit in kaggle santander competition and check score
df_result = pd.DataFrame()
df_result.index = test_data['ID']
df_result['TARGET'] = gb_predictions1
df_result.to_csv('./Prob2_BaseGBPred.csv')

