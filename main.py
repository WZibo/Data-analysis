# Part 1
print("Part 1")
import numpy as np
import pandas as pd

# load data
churn_df = pd.read_csv('bank.data.csv')

# check data info
print(churn_df.info())
# check the unique values for each column
churn_df.nunique()
print(churn_df.nunique())
# check missing values
churn_df.isnull().sum()
print(churn_df.isnull().sum())

# Get target variable
y = churn_df['Exited']

# understand Numerical feature
# 'CreditScore', 'Age', 'Tenure', 'NumberOfProducts', 'Balance', 'EstimatedSalary'
print(churn_df[['CreditScore', 'Age', 'Tenure', 'NumOfProducts','Balance', 'EstimatedSalary']].describe())

# check the feature distribution
import matplotlib.pyplot as plt
import seaborn as sns
# boxplot for numerical feature
_,axss = plt.subplots(2, 3, figsize=[20, 10])
sns.boxplot(x='Exited', y='CreditScore', data=churn_df, ax=axss[0][0])
sns.boxplot(x='Exited', y='Age', data=churn_df, ax=axss[0][1])
sns.boxplot(x='Exited', y='Tenure', data=churn_df, ax=axss[0][2])
sns.boxplot(x='Exited', y='NumOfProducts', data=churn_df, ax=axss[1][0])
sns.boxplot(x='Exited', y='Balance', data=churn_df, ax=axss[1][1])
sns.boxplot(x='Exited', y='EstimatedSalary', data=churn_df, ax=axss[1][2])

# understand categorical feature
_,axss = plt.subplots(2, 2, figsize=[20, 10])
sns.countplot(x='Exited', hue='Geography', data=churn_df, ax=axss[0][0])
sns.countplot(x='Exited', hue='Gender', data=churn_df, ax=axss[0][1])
sns.countplot(x='Exited', hue='HasCrCard', data=churn_df, ax=axss[1][0])
sns.countplot(x='Exited', hue='IsActiveMember', data=churn_df, ax=axss[1][1])

# create a boxplot for the 'CreditScore' feature
plt.boxplot(churn_df['CreditScore'])
# set the title and axis labels
plt.title('CreditScore Boxplot')
plt.xlabel('CreditScore')
plt.ylabel('Value')

# show the plot
plt.show()


# Part 2
print("\nPart 2")
# Get feature space by dropping useless feature
to_drop = ['RowNumber', 'CustomerId', 'Surname', 'Exited']
X = churn_df.drop(to_drop, axis=1)

# Split data into training and testing
from sklearn import model_selection

# Reserve 25% for testing
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, stratify=y, random_state=1)
print('training data has ' + str(X_train.shape[0]) + ' observation with ' + str(X_train.shape[1]) + ' features')
print('test data has ' + str(X_test.shape[0]) + ' observation with ' + str(X_test.shape[1]) + ' features\n')

cat_cols = X.columns[X.dtypes == 'object']
num_cols = X.columns[(X.dtypes == 'float64') | (X.dtypes == 'int64')]

# One hot encoding
from sklearn.preprocessing import OneHotEncoder


def OneHotEncoding(df, enc, categories):
  transformed = pd.DataFrame(enc.transform(df[categories]).toarray(), columns=enc.get_feature_names_out(categories))
  return pd.concat([df.reset_index(drop=True), transformed], axis=1).drop(categories, axis=1)


categories = ['Geography']
enc_ohe = OneHotEncoder()
enc_ohe.fit(X_train[categories])

X_train = OneHotEncoding(X_train, enc_ohe, categories)
X_test = OneHotEncoding(X_test, enc_ohe, categories)

# Ordinal encoding
from sklearn.preprocessing import OrdinalEncoder

categories = ['Gender']
enc_oe = OrdinalEncoder()
enc_oe.fit(X_train[categories])

X_train[categories] = enc_oe.transform(X_train[categories])
X_test[categories] = enc_oe.transform(X_test[categories])

# Standardize/Normalize Data
# Scale the data, using standardization

# for example, use training data to train the standardscaler to get mean and std
# apply mean and std to both training and testing data.
# fit_transform does the training and applying, transform only does applying.
# Because we can't use any info from test, and we need to do the same modification
# to testing data as well as training data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train[num_cols])
X_train[num_cols] = scaler.transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])


# Part3
print("\npart 3")
# Model Training and Result Evaluation
# build models
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Logistic Regression
classifier_logistic = LogisticRegression()
# K Nearest Neighbors
classifier_KNN = KNeighborsClassifier()
# Random Forest
classifier_RF = RandomForestClassifier()


# Use Grid Search to Find Optimal Hyperparameters
from sklearn.model_selection import GridSearchCV

# helper function for printing out grid search results
def print_grid_search_metrics(gs):
    print("Best score: " + str(gs.best_score_))
    print("Best parameters set:")
    best_parameters = gs.best_params_
    for param_name in sorted(best_parameters.keys()):
        print(param_name + ':' + str(best_parameters[param_name]))


# Find Optimal Hyperparameters - LogisticRegression
# Penalty is chose from L1 or L2
# C is the 1/lambda value(weight) for L1 and L2
parameters = {
    'penalty': ('l2', 'l1'),
    'C': (0.01, 0.05, 0.1, 0.2, 1)
}
Grid_LR = GridSearchCV(LogisticRegression(solver='liblinear'), parameters, cv=5)
Grid_LR.fit(X_train, y_train)
# the best hyperparameter combination
# C = 1/lambda
print_grid_search_metrics(Grid_LR)
# best model
best_LR_model = Grid_LR.best_estimator_
y_pred_best = best_LR_model.predict(X_test)
accuracy_best = best_LR_model.score(X_test, y_test)


# Find Optimal Hyperparameters: KNN
# Choose k
parameters = {
    'n_neighbors': [1, 3, 5, 7, 9]
}
Grid_KNN = GridSearchCV(KNeighborsClassifier(), parameters, cv=5)
Grid_KNN.fit(X_train, y_train)
# best k
print_grid_search_metrics(Grid_KNN)
best_KNN_model = Grid_KNN.best_estimator_


# Find Optimal Hyperparameters: Random Forest
# Choose the number of trees
parameters = {
    'n_estimators': [60, 80, 100],
    'max_depth': [1, 5, 10]
}
Grid_RF = GridSearchCV(RandomForestClassifier(), parameters, cv=5)
Grid_RF.fit(X_train, y_train)
# best number of tress
print_grid_search_metrics(Grid_RF)
# best random forest
best_RF_model = Grid_RF.best_estimator_


# Model Evaluation - Confusion Matrix (TNR, TPR, Accuracy)
from sklearn.metrics import confusion_matrix

# calculate accuracy, TNR and TPR, [[tn, fp],[]]
def cal_evaluation(classifier, cm):
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    accuracy = (tp + tn) / (tp + fp + fn + tn + 0.0)
    TNR = tn / (tn + fp + 0.0)
    TPR = tp / (tp + fn + 0.0)
    print(classifier)
    print("Accuracy is: " + str(accuracy))
    print("TNR is: " + str(TNR))
    print("TPR is: " + str(TPR))
    print()


# print out confusion matrices
def draw_confusion_matrices(confusion_matrices):
    class_names = ['Not', 'Churn']
    for cm in confusion_matrices:
        classifier, cm = cm[0], cm[1]
        cal_evaluation(classifier, cm)

# Confusion matrix, accuracy, TNR and TPR for random forest and logistic regression
confusion_matrices = [
    ("\nRandom Forest", confusion_matrix(y_test, best_RF_model.predict(X_test))),
    ("Logistic Regression", confusion_matrix(y_test, best_LR_model.predict(X_test))),
    ("K nearest neighbor", confusion_matrix(y_test, best_KNN_model.predict(X_test)))
]
draw_confusion_matrices(confusion_matrices)

# Model Evaluation - ROC & AUC     ROC of RF Model
from sklearn.metrics import roc_curve

# Use predict_proba to get the probability results of Random Forest
y_pred_rf = best_RF_model.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
best_RF_model.predict_proba(X_test)

# ROC curve of Random Forest result
import matplotlib.pyplot as plt
from sklearn import metrics

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='RF')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve - RF model')
plt.legend(loc='best')
plt.show()

# AUC score
print("AUC of Random Forest:", metrics.auc(fpr_rf, tpr_rf))

# Random Forest Model - Feature Importance Discussion
X_RF = X.copy()
X_RF = OneHotEncoding(X_RF, enc_ohe, ['Geography'])
X_RF['Gender'] = enc_oe.transform(X_RF[['Gender']])

# check feature importance of random forest for feature selection
forest = RandomForestClassifier()
forest.fit(X_RF, y)
importance = forest.feature_importances_
indices = np.argsort(importance)[::-1]

# Print the feature ranking
print("\nFeature importance ranking by Random Forest Model:")
for ind in range(X.shape[1]):
  print("{0} : {1}".format(X_RF.columns[indices[ind]],round(importance[indices[ind]], 4)))
