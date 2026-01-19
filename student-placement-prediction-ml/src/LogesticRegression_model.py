import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

dataset = pd.read_csv('./Placement_Prediction_data.csv')  # Replace with your CSV file name

# Drop 'StudentId' column as it's not needed
dataset.drop(['StudentId'], axis=1, inplace=True)

# Check missing values
dataset.isnull().sum()
#  Label encoding for categorical columns
encoder = LabelEncoder()
columns_to_encode = ['Internship','Hackathon','PlacementStatus']
for column in columns_to_encode:
    dataset[column] = encoder.fit_transform(dataset[column])
#  Remove outliers (example based on CGPA, 12th & 10th percentages)
# Adjust thresholds based on your dataset
dataset = dataset[dataset['CGPA'] <= 10]
dataset = dataset[dataset['12th Percentage'] <= 100]
dataset = dataset[dataset['10th Percentage'] <= 100]


#  Split features and target

x = dataset.loc[:, dataset.columns != 'PlacementStatus']
y = dataset.loc[:, 'PlacementStatus']

# Standardize features
sc = StandardScaler()
x_scaled = sc.fit_transform(x)
x_scaled = pd.DataFrame(x_scaled, columns=x.columns)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.18, random_state=0)

#  Logistic Regression

lr = LogisticRegression()
lr.fit(x_train, y_train)

# Predictions
y_pred = lr.predict(x_test)


#  Evaluation

accuracy = accuracy_score(y_test, y_pred)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nAccuracy:", accuracy * 100, "%")