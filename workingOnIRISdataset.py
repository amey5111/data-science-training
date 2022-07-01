import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

logr = LogisticRegression(random_state=0)
s = svm.SVC()
nb = GaussianNB()
dt = DecisionTreeClassifier(random_state=0)
rf = RandomForestClassifier(random_state=1)
gb = GradientBoostingClassifier(n_estimators=10)
nn = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(5,2),random_state=0)


x_df = pd.read_csv("IRIS.csv")
# print(x_train)
y_df = pd.read_csv("IRIS.csv")

x=x_df.drop(columns= ["species"],axis=1)
print(x)

y=y_df['species']
print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.3)
# x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.3)
# x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.3)
# x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.3)
# x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.3)
# x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.3)
# x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.3)

logr.fit(x_train,y_train)
s.fit(x_train,y_train)
nb.fit(x_train,y_train)
rf.fit(x_train,y_train)
dt.fit(x_train,y_train)
nn.fit(x_train,y_train)
gb.fit(x_train,y_train)

y_pred =logr.predict(x_test)
ys_pred =s.predict(x_test)
ynb_pred =nb.predict(x_test)
yrf_pred =rf.predict(x_test)
ydt_pred =dt.predict(x_test)
ynn_pred =nn.predict(x_test)
ygb_pred =gb.predict(x_test)

print("log",accuracy_score(y_test,y_pred))
print("svm",accuracy_score(y_test,ys_pred))
print("navies bayes",accuracy_score(y_test,ynb_pred))
print("random forest",accuracy_score(y_test,yrf_pred))
print("decision tree",accuracy_score(y_test,ydt_pred))
print("neural network",accuracy_score(y_test,ynn_pred))
print("gradient boosting",accuracy_score(y_test,ygb_pred))



#      sepal_length  sepal_width  petal_length  petal_width
# 0             5.1          3.5           1.4          0.2
# 1             4.9          3.0           1.4          0.2
# 2             4.7          3.2           1.3          0.2
# 3             4.6          3.1           1.5          0.2
# 4             5.0          3.6           1.4          0.2
# ..            ...          ...           ...          ...
# 145           6.7          3.0           5.2          2.3
# 146           6.3          2.5           5.0          1.9
# 147           6.5          3.0           5.2          2.0
# 148           6.2          3.4           5.4          2.3
# 149           5.9          3.0           5.1          1.8
#
# [150 rows x 4 columns]
# 0         Iris-setosa
# 1         Iris-setosa
# 2         Iris-setosa
# 3         Iris-setosa
# 4         Iris-setosa
#             ...
# 145    Iris-virginica
# 146    Iris-virginica
# 147    Iris-virginica
# 148    Iris-virginica
# 149    Iris-virginica
# Name: species, Length: 150, dtype: object
# log 0.9777777777777777
# svm 0.9777777777777777
# navies bayes 1.0
# random forest 0.9777777777777777
# decision tree 0.9777777777777777
# neural network 0.24444444444444444
# gradient boosting 0.9777777777777777