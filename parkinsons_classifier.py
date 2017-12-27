## 'Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection', 
## Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM. 
## BioMedical Engineering OnLine 2007, 6:23 (26 June 2007)

# The dataset is composed of a range of biomedical voice measurements from 31
# people, 23 with PD. Each column is a particular voice measure and each row
# corresponds to a voice recording. The status column is set to 0 for healthy
# and 1 for PD.

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
import pickle

df = pd.read_csv('parkinsons.data.txt')
# Name needs to be dropped from dataset
df.drop(['name'], 1, inplace=True)

X = np.array(df.drop(['status'], 1))
X = scale(X)
y = np.array(df['status'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fine tuning to determine best parameters for KNeighborsClassifier
##hyperparameters = {'n_neighbors':[3, 4, 5],
##               'weights':['uniform', 'distance'],
##               'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
##               'leaf_size':[1, 5, 10, 20, 30],
##               'p':[1, 2, 3]}                                                     
##clf = GridSearchCV(KNeighborsClassifier(), hyperparameters, cv=10)
##print(clf.best_params_)

# Best parameters chosen from GridSearch
##clf = KNeighborsClassifier(n_neighbors=4, weights='distance', p=1)
##clf.fit(X_train, y_train)

# Saved estimator in .pickle file 
##with open('parkinsons_classifier.pickle', 'wb') as file:
##    pickle.dump(clf, file)
pickle_in = open('parkinsons_classifier.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
print('Accuracy:', '{}%'.format(accuracy*100))

# Can compare predicted results and observed data for testing group
# or predict unique featuresets
attributes = X_test
print(y_test)
prediction = clf.predict(attributes)
print(prediction)
    

