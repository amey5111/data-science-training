import numpy
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

df = pd.read_csv("Walmart.csv")
print(df)
df['Date'] = pd.to_datetime(df['Date'])
df['days'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year
df['WeekOfYear'] = df.Date.dt.isocalendar().week
df.drop('Date', axis=1, inplace=True)

# print(df.isnull().all())

x = df.drop('Weekly_Sales', axis=1)
y = df['Weekly_Sales']

# Feature Extraction
pca = PCA(n_components=3)
fit = pca.fit(x)

# Splitting Data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=0, test_size=0.2)
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
3  Gradient_Boosting  0.908652        2.887160e+10
5      Random_Forest  0.713018        9.070438e+10
4      Decision_Tree  0.699240        9.505901e+10
1                KNN  0.297445        2.220513e+11
0  Linear Regression  0.168144        2.629183e+11
2         Linear_SVM -0.026239        3.243555e+11
'''
