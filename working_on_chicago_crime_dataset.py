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
from sklearn.ensemble import ExtraTreesRegressor
import pandas as pd
from sklearn.decomposition import PCA
data = pd.concat([pd.read_csv('../input/crimes-in-chicago/Chicago_Crimes_2001_to_2004.csv', error_bad_lines=False),
                 pd.read_csv('../input/crimes-in-chicago/Chicago_Crimes_2005_to_2007.csv', error_bad_lines=False)], ignore_index=True)
data = pd.concat([data, pd.read_csv(
    '../input/crimes-in-chicago/Chicago_Crimes_2008_to_2011.csv', error_bad_lines=False)], ignore_index=True)
data = pd.concat([data, pd.read_csv(
    '../input/crimes-in-chicago/Chicago_Crimes_2012_to_2017.csv', error_bad_lines=False)], ignore_index=True)
data.head()

data = data.sample(n=100000)
data.shape
data.drop(['Unnamed: 0', 'ID', 'Description', 'Location Description', 'Date', 'Block', 'Location', 'FBI Code',
          'Case Number', 'X Coordinate', 'Y Coordinate', 'Community Area', 'Updated On', 'IUCR'], inplace=True, axis=1)
nan_value = float("NaN")
data.replace("", nan_value, inplace=True)
data.dropna(subset=['Ward', 'District', 'Latitude', 'Longitude'], inplace=True)
data.drop_duplicates()
le = LabelEncoder()
data['Arrest'] = le.fit_transform(data['Arrest'])
data['Primary Type'] = le.fit_transform(data['Primary Type'])
x = data.drop('Arrest', axis=1)
y = data['Arrest']
pca = PCA(n_components=3)
fit = pca.fit(x)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=2)

names = ['Logistic Regression ', "GradientBoostingClasifier",
         "RandomForestClassifier", "Decision_Tree_Classifier", "SVC", "MLPClassifier"]
regressors = [
    LogisticRegression(random_state=45),
    GradientBoostingClassifier(n_estimators=12),
    RandomForestClassifier(random_state=2),
    DecisionTreeClassifier(random_state=42),
    svm.SVC(),
    MLPClassifier(solver='lbfgs', alpha=1e-5,
                  hidden_layer_sizes=(5, 2), random_state=2)
]

scores = []
mean_score = []
for name, clf in zip(names, regressors):
    clf.fit(x_train, y_train)
    score = accuracy_score(y_test, clf.predict(x_test))
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
2     RandomForestClassifier  0.848780            0.151220
1  GradientBoostingClasifier  0.831171            0.168829
3   Decision_Tree_Classifier  0.795062            0.204938
0       Logistic Regression   0.715949            0.284051
4                        SVC  0.715949            0.284051
5              MLPClassifier  0.715949            0.284051
'''
