import pandas as pd
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import numpy as np

df = pd.read_csv("winequalityN.csv")


# print(df.isnull().sum())
df = df.fillna(df.median())
# print(df.info())

df['quality'] = pd.cut(df['quality'], 2, labels=['1', '2'])

x = df.drop(["type", "citric acid", "alcohol", "pH", "density", "quality"],
            axis=1)
y = df["quality"]

# Feature Selection
best_features = SelectKBest(score_func=chi2, k="all")
fit = best_features.fit(x, y)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(x.columns)
features_scores = pd.concat([df_columns, df_scores], axis=1)
features_scores.columns = ["Attributes", "Score"]
# print(features_scores)
X_Train, X_Test, Y_Train, Y_Test = train_test_split(
    x, y, test_size=0.3, random_state=0)
names = ['Logistic Regression ', "GradientBoostingClasifier", "RandomForestClassifier",
         "Decision_Tree_Classifier", "SVC", "MLPClassifier", "MultinomialClassifier"]
regressors = [
    LogisticRegression(random_state=45),
    GradientBoostingClassifier(n_estimators=12),
    RandomForestClassifier(random_state=2),
    DecisionTreeClassifier(random_state=42),
    svm.SVC(),
    MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=2), MultinomialNB()]

scores = []
mean_score = []
for name, clf in zip(names, regressors):
    clf.fit(X_Train, Y_Train)
    score = accuracy_score(Y_Test, clf.predict(X_Test))
    mse = 1-score
    scores.append(score)
    mean_score.append(mse)

scores_df = pd.DataFrame()
scores_df['name           '] = names
scores_df['accuracy'] = scores
scores_df['Mean_squared_error'] = mean_score
print(scores_df.sort_values('accuracy', ascending=False))

# RESULTS
'''
             name             accuracy  Mean_squared_error
2     RandomForestClassifier  0.879487            0.120513
3   Decision_Tree_Classifier  0.824103            0.175897
0       Logistic Regression   0.808205            0.191795
1  GradientBoostingClasifier  0.807179            0.192821
4                        SVC  0.807179            0.192821
5              MLPClassifier  0.807179            0.192821
6      MultinomialClassifier  0.770769            0.229231
'''
