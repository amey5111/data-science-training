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
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer


rf = RandomForestClassifier(random_state=1)

dc = ['greek', 'southern_us', 'filipino', 'indian', 'jamaican', 'spanish', 'italian', 'mexican', 'chinese', 'british',
      'thai', 'vietnamese', 'cajun_creole', 'brazilian', 'french', 'japanese', 'irish', 'korean', 'moroccan', 'russian']

df = pd.read_json("What'sCoockingTrain.json")
# print(df.describe())

print(df.columns.values)

print(df['cuisine'].unique())

x = df['ingredients']

y = df['cuisine'].apply(dc.index)

df['all_ingredients'] = df['ingredients'].map(";".join)

cv = CountVectorizer()

X = cv.fit_transform(df['all_ingredients'].values)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, test_size=0.3)

train = rf.fit(X_train, y_train)

y_p = rf.predict(X_test)
# print(accuracy_score(y_test,y_p),normalise=False)
print("mse", mean_squared_error(y_test, y_p))
print("acc", accuracy_score(y_test, y_p))
# output-----
# Mean Squared  Error: 15.31174055141205
# accuracy score :  0.7531215955752953
