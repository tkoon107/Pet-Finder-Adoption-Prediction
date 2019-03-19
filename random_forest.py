import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import export_graphviz
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn import metrics
os.chdir(r"D:\Project Data\Pet Finder")

#Load and preprocess dataset

#encode state variable
train = pd.read_csv(r"./train/train.csv").drop(axis=1, columns ="Description")
test = pd.read_csv(r"./test/test.csv").drop(axis=1, columns ="Description")

 
len(train['RescuerID'].unique())

train['RescuerID'].value_counts().describe()


adoption_count_dict = dict(train['RescuerID'].value_counts().apply(lambda x:">40" if x > 40 else "<40"))

train['RescuerID_adoption_count'] = train['RescuerID'].apply(lambda x: adoption_count_dict[x])


train.describe()

len(train['RescuerID'].value_counts().apply(lambda x:">40" if x > 40 else "<40").values)
lambda x: if x > 40 ">40" else "<40"



#Select features
feature_list = ['Type',  'Age', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
       'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
       'Sterilized', 'Health', 'Quantity', 'Fee', 'State', 'RescuerID',
       'VideoAmt', 'PhotoAmt']

le = LabelEncoder()

onehotencoder = OneHotEncoder(categorical_features = ['State'])
onehotencoder.fit(['State'])

hashEncoder = HashingEncoder(cols = ['RescuerID'])
hashEncoder.fit_transom(train_X)


new_cat_features = enc.transform(cat_features)

one_hot_test = onehotencoder.fit_transform(train).toarray()

train['RescuerID'] = le.fit_transform(train['RescuerID'])

train['RescuerID']

pd.factorize(train['RescuerID'])



train_X = train[feature_list]
train_y = train['AdoptionSpeed']

random_grid = {'n_estimators': [int(x) for x in np.linspace(start = 100, stop = 500, num = 5)],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4],
               'bootstrap': [True, False]}

#skf = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

classifier = RandomForestClassifier()

param_search = RandomizedSearchCV(estimator = classifier, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1)

param_search.fit(train_X, train_y)



fitted_classifier = classifier.fit(train_X, train_y)

fitted_classifier.oob_score_
cross_val_score(fitted_classifier, train_X, train_y, cv=5, scoring='f1_macro')



predictions = classifier.predict(train_y)




for (train_index,test_index) in skf.split(train_X,train_y):
    print((train_index, test_index))


i = 1
for train_index,test_index in skf.split(train_X,train_y):
    print('{} of KFold {}'.format(i,kf.n_splits))
    xtr,xvl = train_X.loc[train_index],train_X.loc[test_index]
    ytr,yvl = train_y.loc[train_index],train_y.loc[test_index]
    
    #model
    classifier.fit(train_X, train_y)
    score = roc_auc_score(yvl,lr.predict(xvl))
    print('ROC AUC score:',score)
    cv_score.append(score)    
    pred_test = lr.predict_proba(x_test)[:,1]
    pred_test_full +=pred_test
    i+=1








test_X = test[feature_list]

#program and fit model
classifier = RandomForestClassifier(n_estimators=10, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0., min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)
classifier.fit(train_X, train_y)

#predictions
classifier.predict()


#important features
classifier.feature_importances_
classifier.decision_path

# Export as dot file
export_graphviz(classifier, out_file='tree.dot', 
                feature_names = feature_list,
                class_names = train['AdoptionSpeed'].unique(),
                rounded = True, proportion = False, 
                precision = 2, filled = True)


from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)




dot_data = export_graphviz(clf, out_file=None) 
graph = export_graphviz.Source(dot_data) 
graph.render("iris") 
