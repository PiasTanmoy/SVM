# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 11:33:04 2019

@author: Pias Tanmoy
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, 1:4].values
y = dataset.iloc[:, 4].values

X

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
X[:, 0] = label_encoder.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.svm import SVC
classifier = SVC(kernel='poly', random_state=0)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


for i in range(y.shape[0]):
    if (y[i] == 0):
        color = 'b'
        m = '^'
    else:
        color = 'r'
        m = 'o'
        
    x0 = X[i, 0]
    x1 = X[i, 1]
    x2 = X[i, 2]
    
    ax.scatter(x0, x1, x2, c=color, marker=m)
        
ax.set_xlabel('Gender')
ax.set_ylabel('Age')
ax.set_zlabel('Salary')

plt.show()

    


# =============================================================================
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# 
# x1 = X[:, 0]
# x2 =X[:, 1]
# x3 =X[:, 2]
# 
# ax.scatter(x1, x2, x3, c='r', marker='o')
# 
# ax.set_xlabel('Gender')
# ax.set_ylabel('Age')
# ax.set_zlabel('Salary')
# 
# plt.show()
# 
# for c, m, zlow, zhigh in [('r', 'o', 0), ('b', '^', 1)]:
#     xs = randrange(n, 23, 32)
#     ys = randrange(n, 0, 100)
#     zs = randrange(n, zlow, zhigh)
#     ax.scatter(xs, ys, zs, c=c, marker=m)
#     
# 
# 
# =============================================================================

