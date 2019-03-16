import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

os.chdir(r"D:\Project Data\Pet Finder")
train = pd.read_csv(r"./train/train.csv").drop(axis=1, columns ="Description")
test = pd.read_csv(r"./test/test.csv").drop(axis=1, columns ="Description")

#Preprocess dataset

classifier = RandomForestClassifier(n_estimators=10, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0., min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None)




#Fit model
