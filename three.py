import pandas as pd
df = pd.read_csv("IRIS.csv")
#print(df)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logr=LogisticRegression()
x=df.drop(['species'], axis=1)
print(x)
y=df['species']
print(y)

x_train, x_test, y_train, y_test=train_test_split(x,y,random_state=0,test_size=0.3)

train=logr.fit(x_train,y_train)
y_pred=logr.predict(x_test)
print(accuracy_score(y_test,y_pred))