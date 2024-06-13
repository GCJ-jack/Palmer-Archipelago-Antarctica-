import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.metrics import mean_squared_error
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

df = pd.read_csv('penguins.csv')
print(f"Data Shape: {df.shape}")
print(f"Data head: {df.head()}")
print(f"Data column: {df.columns}")
print(f"Data information: {df.info()}")
print(f"Data description: {df.describe()}")
print(f"Miss data: {df.isna().sum()}")

df = pd.DataFrame(df)
df = df.drop(['rowid', 'year'], axis=1)

# print(df.head())
# print(df.info)

h = .02

# 计算每个species在bill_length_mm和flipper_length_mm上的平均值
species_bill_length_avg = df.groupby('species')['bill_length_mm'].transform('mean')
species_flipper_length_avg = df.groupby('species')['flipper_length_mm'].transform('mean')
# print(df['species'].unique())
# print(df['island'].unique())
# print(species_bill_length_avg.values)
df['bill_length_mm'] = df['bill_length_mm'].fillna(species_bill_length_avg)

df['flipper_length_mm'] = df['flipper_length_mm'].fillna(species_flipper_length_avg)

palette_colors = {"Adelie": "red", "Chinstrap": "yellow", "Gentoo": "blue"}

island_colors = {"Torgersen": "blue", "Biscoe": "green", "Dream": "red"}

# Plotting the gender ratio within the penguin dataset
gender_distribution = df.groupby(['species', 'sex']).size().unstack()

# Creating a bar plot to show the gender ratio within each penguin species
gender_distribution.plot(kind='bar', stacked=True, color=['blue', 'pink'])
plt.title('Gender Distribution within Each Penguin Species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(title='Gender')
plt.show()

sns.pairplot(data=df, hue='species')
plt.show()
plt.savefig('scatter.png', format='png', dpi=300)


df['species'] = df['species'].astype('category')
df['species'] = df['species'].cat.codes
df['species'].unique()

numeric_columns = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
numeric_df = df[numeric_columns]

# 计算相关系数矩阵
corr_matrix = numeric_df.corr()

# 使用热图可视化相关系数矩阵
plt.figure(figsize=(8, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.savefig('corr.png', format='png', dpi=300)
plt.show()

# dummy baseline


X = df.drop(['species', 'island', 'sex', 'bill_depth_mm', 'body_mass_g'], axis=1)
y = df['species']
# defining the class

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

x_min, x_max = X_train.iloc[:, 0].min() - .5, X_train.iloc[:, 0].max() + .5
y_min, y_max = X_train.iloc[:, 1].min() - .5, X_train.iloc[:, 1].max() + .5

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Create an imputer object with a mean filling strategy
# 选择特征和标签

dummy_clf = DummyClassifier(strategy='most_frequent', random_state=42)

dummy_clf.fit(X_train, y_train)

dummy_predictions = dummy_clf.predict(X_test)

dummy_accuracy = accuracy_score(y_test, dummy_predictions)

cm = confusion_matrix(y_test, dummy_predictions)
ps = precision_score(y_test, dummy_predictions, average='macro')
rs = recall_score(y_test, dummy_predictions, average='macro')
f_one_score = f1_score(y_test, dummy_predictions, average='macro')

print(f"Baseline (Dummy) Accuracy: {dummy_accuracy:.4f}")
print(f"Baseline (Dummy) Confusion matrix: {cm}")
print(f"Baseline (Dummy) Precision score: {ps}")
print(f"Baseline (Dummy) Recall score: {rs}")
print(f"Baseline (Dummy) F1 score: {f_one_score}")

plt.figure(1, figsize=(3, 3))
plt.set_cmap(plt.cm.Paired)

############################################

# Set a variable max_k to 30
# Instantiate KFold with 5 splits.
kf = KFold(n_splits=5, random_state=10, shuffle=True)
# Set the parameter random_state to help you reproduce your results if needed.

# Set a variable max_k to 30
max_k = 30

# Initialize two lists to store the
# training accuracies and validation accuracies
# (these need to store max_k*5 accuracies)
training_accuracy = []
validation_accuracy = []

# Loop over the values of k:
for k in range(1, max_k + 1):

    # Instantiate a k-nn classifier (Use the sklearn classifier)
    # with the current value of k
    knn_classifier = KNeighborsClassifier(n_neighbors=k + 1)

    # Initialize variables to store accuracies for each split
    train_accuracies_split = []
    val_accuracies_split = []

    # Loop over the cross-validation splits:
    for train_index, val_index in kf.split(X_train):
        Xtrain, Xval = X_train.iloc[train_index], X_train.iloc[val_index]
        ytrain, yval = y_train.iloc[train_index], y_train.iloc[val_index]

        # fit the model on the current split of data
        knn_classifier.fit(Xtrain, ytrain)

        # make predictions
        pred_train = knn_classifier.predict(Xtrain)
        pred_val = knn_classifier.predict(Xval)

        # calculate training and validation accuracy and store
        train_accuracies_split.append(accuracy_score(ytrain, pred_train))
        val_accuracies_split.append(accuracy_score(yval, pred_val))

    # Append mean accuracies for the current k
    training_accuracy.append(np.mean(train_accuracies_split))
    validation_accuracy.append(np.mean(val_accuracies_split))
    print(
        f"K = {k}, Training Accuracy: {np.mean(train_accuracies_split):.4f}, Validation Accuracy: {np.mean(val_accuracies_split):.4f}")

assert (np.shape(training_accuracy) == (max_k,))
assert (np.shape(validation_accuracy) == (max_k,))

# Plotting the mean training and validation accuracies
plt.plot(range(1, max_k + 1), training_accuracy, label='Training Accuracy')
plt.plot(range(1, max_k + 1), validation_accuracy, label='Validation Accuracy')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.title('Mean Training and Validation Accuracies vs. k')
plt.legend()
plt.show()
plt.savefig('meta parameter.png', format='png', dpi=300)

knn = KNeighborsClassifier(n_neighbors=25)

# neighbours = {'n_neighbors': np.arange(1, 5)}
# knn_cv = GridSearchCV(knn, neighbours, cv=5)

# 使用处理过的训练集训练模型
knn.fit(X_train, y_train)

# 使用模型进行预测
predictions = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)
ps = precision_score(y_test, predictions, average='macro')
rs = recall_score(y_test, predictions, average='macro')
f_one_score = f1_score(y_test, predictions, average='macro')

# Print out the predicted genders

print(f"Accuracy: {accuracy}")
print(f"Confusion matrix: {cm}")
print(f"Precision score: {ps}")
print(f"Recall score: {rs}")
print(f"F1 score: {f_one_score}")


# 现在使用具有特征名称的DataFrame进行预测
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
# 将Z转换为正确的形状，然后绘制contourf等
Z = Z.reshape(xx.shape)

plt.figure(1, figsize=(6, 5))
# plt.set_cmap(plt.cm.Paired)
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

# Plot training points
plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, cmap=cmap_bold)
plt.xlabel('culmen_length')
plt.ylabel('flipper_length')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()
