# Importing the libraries
import pandas as pd
import numpy as np
#import category_encoders as ce

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\vikash\Desktop\DSA\ibank\1526115457_cust_Bank.csv')
dataset = dataset[['age','job','marital','education','housing','loan','y']]
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values


"""# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X[0] = sc.fit_transform(X[0])
#X_test = sc.transform(X_test)"""


#nominal variable : job
#binary variable: marital, housing, loan
#ordinal variable : education


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])
X[:, 5] = labelencoder_X.fit_transform(X[:, 5])
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

onehotencoder = OneHotEncoder(categorical_features='all')

X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)



# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

uniqueValues, occurCount = np.unique(y_pred, return_counts=True)
# Zip both the arrays
listOfUniqueValues = zip(uniqueValues, occurCount)

print('Unique Values along with occurrence Count')
# Iterate over the zip object
for elem in listOfUniqueValues:
    print(elem[0], ' Occurs : ', elem[1], ' times')

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(report)
print('Accuracy for LR: ' + str(accuracy*100) + '%')
"""correct_predictions = cm[0][0]+cm[1][1]
wrong_predictions = cm[0][1]+cm[1][0]
total_predictions = cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0]
accuracy = correct_predictions / total_predictions * 100"""


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()


import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_circles
X,y = make_circles(90, factor=0.2, noise=0.1)
#noise = standard deviation of Gaussian noise added in data.
#factor = scale factor between the two circles
plt.scatter(X[:,0],X[:,1], c=y, s=50, cmap='seismic')
plt.show()