import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load the Titanic dataset
data = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Explore the first few rows of the data
print(data.head())

# Get summary statistics of the data
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Plot the distribution of the target variable 'Survived'
sns.countplot(x='Survived', data=data)
plt.show()

# Plot the distribution of 'Age' by 'Survived'
sns.boxplot(x='Survived', y='Age', data=data)
plt.show()

# Plot the distribution of 'Fare' by 'Survived'
sns.boxplot(x='Survived', y='Fare', data=data)
plt.show()

# Plot the distribution of 'Pclass' by 'Survived'
sns.countplot(x='Survived', hue='Pclass', data=data)
plt.show()

# Split the data into training and testing sets
X = data.drop('Survived', axis=1)
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Fit a Logistic Regression model to the training data
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model's performance
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()
