from io import StringIO
from tkinter import Image

import numpy as np
from skimage.metrics import mean_squared_error
from sklearn import tree
import pandas as pd
import pydotplus as pydotplus
from matplotlib import pyplot as plt
from sklearn import metrics, __all__  # Import scikit-learn metrics module for accuracy calculation
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz

df = pd.read_csv('penguins.csv')
df = pd.DataFrame(df)

# features = df['bill_length_mm', 'flipper_length_mm']

species_bill_length_avg = df.groupby('species')['bill_length_mm'].transform('mean')
species_flipper_length_avg = df.groupby('species')['flipper_length_mm'].transform('mean')

df['bill_length_mm'] = df['bill_length_mm'].fillna(species_bill_length_avg)

# 用相应species的平均值填充flipper_length_mm的缺失值
df['flipper_length_mm'] = df['flipper_length_mm'].fillna(species_flipper_length_avg)

X = df.drop(['species', 'island', 'sex', 'bill_depth_mm', 'body_mass_g', 'year', 'rowid'], axis=1)
y = df['species']

feature_cols = ['bill_length_mm', 'flipper_length_mm']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)  # 70% training and 30% test

max_depth_values = 20
mse_results = []
train_accuracies = [[] for _ in range(max_depth_values)]
val_accuracies = [[] for _ in range(max_depth_values)]

dummy_clf = DummyClassifier(strategy='most_frequent')

# 训练哑分类器
dummy_clf.fit(X_train, y_train)

# 在测试数据上进行预测
y_pred = dummy_clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)

print(f"Baseline accuracy: {accuracy}")

encoder = LabelEncoder()
y_test_encoded = encoder.fit_transform(y_test)

kf = KFold(n_splits=10, random_state=63, shuffle=True)

for max_depth in range(max_depth_values):
    # Instantiate a DecisionTreeRegressor with max_depth
    dtc = DecisionTreeClassifier(criterion='entropy',max_depth=max_depth+1)
    # fit the model on the training set
    # Loop over cross-validation splits. Note that we perform cross-validation on our training data Xtr.
    # We keep our testing data Xtest aside
    for train_index, val_index in kf.split(X_train):
        Xtrain, Xval = X_train.iloc[train_index], X_train.iloc[val_index]
        ytrain, yval = y_train.iloc[train_index], y_train.iloc[val_index]

        # Fit the classifier on Xtrain and Ytrain
        dtc.fit(Xtrain, ytrain)

        # Make predictions on Xtrain and assign to a variable pred_train
        pred_train = dtc.predict(Xtrain)

        # Make predictions on Xval and assign to a variable pred_val
        pred_val = dtc.predict(Xval)

        # Calculate the accuracy of the predictions on the training set and save in the variable train_accuracies
        train_accuracies[max_depth].append(accuracy_score(ytrain, pred_train))

        # Do the same for the predictions on the validation set
        val_accuracies[max_depth].append(accuracy_score(yval, pred_val))


# Calculate the mean and standard deviation for each depth across splits
train_accuracy_mean = np.mean(train_accuracies, axis=1)
train_accuracy_stdev = np.std(train_accuracies, axis=1)
val_accuracy_mean = np.mean(val_accuracies, axis=1)
val_accuracy_stdev = np.std(val_accuracies, axis=1)

# The arrays of means and standard deviation should have shape (max_d, ). The following will generate an error if not.
assert (np.shape(train_accuracy_mean) == (max_depth_values,))
assert (np.shape(train_accuracy_stdev) == (max_depth_values,))
assert (np.shape(val_accuracy_mean) == (max_depth_values,))
assert (np.shape(val_accuracy_stdev) == (max_depth_values,))

# Plotting mean accuracy on training set
plt.plot(range(1, max_depth_values+1), train_accuracy_mean, label='Training Set', marker='o')

# Plotting mean accuracy on validation set
plt.plot(range(1, max_depth_values+1), val_accuracy_mean, label='Validation Set', marker='o')

# Adding labels and title
plt.xlabel('Max Depth')
plt.ylabel('Mean Accuracy')
plt.title('Mean Accuracy vs Max Depth for Decision Tree Classifier')
plt.legend()

# Display the plot
plt.show()

dtc = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_impurity_decrease=0.2)

# Split dataset into training set and test set

dtc.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = dtc.predict(X_test)

plt.figure(1, figsize=(6, 5))
tree.plot_tree(dtc, feature_names=feature_cols, filled=True, class_names=['0', '1', '2'])

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
ps = precision_score(y_test, y_pred, average='macro')
rs = recall_score(y_test, y_pred, average='macro')
f_one_score = f1_score(y_test, y_pred, average='macro')

# Print out the predicted genders

print(f"Accuracy: {accuracy}")
print(f"Confusion matrix: {cm}")
print(f"Precision score: {ps}")
print(f"Recall score: {rs}")
print(f"F1 score: {f_one_score}")
