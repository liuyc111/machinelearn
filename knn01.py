# %%

from scikit_learn import model_selection
from sklearn import datasets
import joblib
from scikit_learn import KNNclassifier
import os
from collections import Counter
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

# %%
os.makedirs('outputs', exist_ok=True)
joblib.dump(KNNclassifier.KNNClassifier, 'outputs/model.joblib')
# %%
new_model = joblib.load("outputs/model.joblib")
mm = new_model(3)
iris = datasets.load_iris()
iris.keys()
x = iris.data
y = iris.target
x_trainiris, y_trainiris, x_testiris, y_testiris = model_selection.train_test_split(
    x, y, 0.8,None)
mm.fit(x_trainiris, y_trainiris)
sum((mm.predict(x_testiris)==y_testiris)/len(y_testiris))

# %%
