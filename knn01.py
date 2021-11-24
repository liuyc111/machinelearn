# %%
from collections import Counter
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

raw_data_x = [[3.393533211, 2.331273381],
              [3.110073483, 1.781539638],
              [1.343808831, 3.368360954],
              [3.582294042, 4.679179110],
              [2.280362439, 2.866990263],
              [7.423436942, 4.696522875],
              [5.745051997, 3.533989803],
              [9.172168622, 2.511101045],
              [7.792783481, 3.424088941],
              [7.939820817, 0.791637231]]
raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

X_train = np.array(raw_data_x)
Y_train = np.array(raw_data_y)

# %%
import os
from scikit_learn import KNNclassifier
import joblib
os.makedirs('outputs', exist_ok=True)
joblib.dump(KNNclassifier.KNNClassifier, 'outputs/model.joblib')
# %%
new_model = joblib.load("outputs/model.joblib")
mm=new_model(3)
mm.fit(X_train,Y_train)
mm.predict( [[3.393533211, 2.331273381],
              [3.110073483, 1.781539638],
              [7.939820817, 0.791637231]])

