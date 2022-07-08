from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor

df = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
# df.head()
# df.info()
# df.describe()
df['source'] = 'train'
test['source'] = 'test'
data = pd.concat([df, test])
data = data.sample(n=100000)
# Dropping Unwanted Features
# only 30% data is available
data.drop('Product_Category_3', axis=1, inplace=True)
data.drop('User_ID', axis=1, inplace=True)
data.drop('Product_ID', axis=1, inplace=True)
# Filling Missing Values
data['Product_Category_2'].fillna(
    data['Product_Category_2'].mean(), inplace=True)
# Replacing
data['Age'] = data['Age'].apply(lambda x: str(x).replace('55+', '55'))
data['Stay_In_Current_City_Years'] = data['Stay_In_Current_City_Years'].apply(
    lambda x: str(x).replace('4+', '4'))

# Label Encoding
le = LabelEncoder()
data['Gender'] = le.fit_transform(data['Gender'])
le = LabelEncoder()
data['Age'] = le.fit_transform(data['Age'])
le = LabelEncoder()
data['City_Category'] = le.fit_transform(data['City_Category'])

# Converting Datatype to int
data['Stay_In_Current_City_Years'] = data['Stay_In_Current_City_Years'].astype(
    'int')
print(data.shape)
# Separating Train and Test
train = data.loc[data['source'] == 'train']
test = data.loc[data['source'] == 'test']
train.drop('source', axis=1, inplace=True)
test.drop('source', axis=1, inplace=True)

X = train.drop("Purchase", axis=1)
Y = train["Purchase"]

"""Feature Selection"""
selector = ExtraTreesRegressor()
selector.fit(X, Y)
feature_imp = selector.feature_importances_
X.drop(['Gender', 'City_Category', 'Marital_Status'], axis=1, inplace=True)

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, random_state=0, test_size=0.2)
names = ['Linear Regression', "KNN", "Linear_SVM",
         "Gradient_Boosting", "Decision_Tree", "Random_Forest"]
regressors = [
    LinearRegression(),
    KNeighborsRegressor(n_neighbors=3),
    SVR(),
    GradientBoostingRegressor(n_estimators=100),
    DecisionTreeRegressor(max_depth=5),
    RandomForestRegressor(max_depth=5, n_estimators=100)]

scores = []
mean_score = []
for name, clf in zip(names, regressors):
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    mse = mean_squared_error(y_test, clf.predict(x_test))
    scores.append(score)
    mean_score.append(mse)

scores_df = pd.DataFrame()
scores_df['name           '] = names
scores_df['accuracy'] = scores
scores_df['Mean_squared_error'] = mean_score
print(scores_df.sort_values('accuracy', ascending=False))


'''
     name             accuracy  Mean_squared_error
3  Gradient_Boosting  0.632328        9.221056e+06
4      Decision_Tree  0.570245        1.077807e+07
5      Random_Forest  0.566460        1.087298e+07
1                KNN  0.498593        1.257506e+07
0  Linear Regression  0.128555        2.185547e+07
2         Linear_SVM  0.121373        2.203559e+07
'''
