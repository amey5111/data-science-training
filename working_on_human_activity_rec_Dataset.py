import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
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

train_path = "EBMAMRS/ml/Datasets/UCI HAR Dataset/train/"
test_path = "GauthaEBMAMRS/ml/Datasets/UCI HAR Dataset/test/"
features_path = "EBMAMRS/ml/Datasets/UCI HAR Dataset/features.txt"

features = []
with open(features_path) as f:
    features = [line.split()[1] for line in f.readlines()]
re = []
for i, f in enumerate(features):
    for j in range(i+1, len(features)):
        if features[i] == features[j] and features[i] not in re:
            re.append(features[i])
for i, f in enumerate(features):
    features[i] = ''.join(e for e in f if e not in ['(', ')', '-', ','])

train = pd.read_csv(train_path + "X_train.txt",
                    delim_whitespace=True, header=None)
train.columns = features
train['subject'] = pd.read_csv(
    train_path + 'subject_train.txt', header=None, squeeze=True)
test = pd.read_csv(test_path + "X_test.txt",
                   delim_whitespace=True, header=None)
test.columns = features
test['subject'] = pd.read_csv(
    test_path + 'subject_test.txt', header=None, squeeze=True)
y_train = pd.read_csv(train_path + 'y_train.txt',
                      names=['Activity'], squeeze=True)
y_test = pd.read_csv(test_path + 'y_test.txt',
                     names=['Activity'], squeeze=True)

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
    clf.fit(train, y_train)
    score = accuracy_score(y_test, clf.predict(test))
    mse = 1-score
    scores.append(score)
    mean_score.append(mse)

scores_df = pd.DataFrame()
scores_df['name           '] = names
scores_df['accuracy'] = scores
scores_df['Mean_squared_error'] = mean_score
print(scores_df.sort_values('accuracy', ascending=False))


# RESULTS:
'''
             name             accuracy  Mean_squared_error
0       Logistic Regression   0.952494            0.047506
4                        SVC  0.930777            0.069223
2     RandomForestClassifier  0.920937            0.079063
1  GradientBoostingClasifier  0.896166            0.103834
3   Decision_Tree_Classifier  0.855786            0.144214
5              MLPClassifier  0.182219            0.817781
'''
