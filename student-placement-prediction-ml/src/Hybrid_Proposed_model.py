import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load CSV dataset
df = pd.read_csv("Placement_Prediction_data.csv")  # replace with your CSV file path

# Step 2: Encode categorical variables
categorical_cols = ['Internship', 'Hackathon']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Encode target variable
le_target = LabelEncoder()
df['PlacementStatus'] = le_target.fit_transform(df['PlacementStatus'])

# Step 3: Split features and target
X = df.drop(columns=['StudentId', 'PlacementStatus'])
y = df['PlacementStatus']

# Step 4: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Define base models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(eval_metric='logloss')

# Step 6: Create Voting (Hybrid) Classifier
hybrid_model = VotingClassifier(estimators=[('rf', rf), ('xgb', xgb)], voting='soft')
hybrid_model.fit(X_train, y_train)

# Step 7: Make predictions and evaluate
y_pred = hybrid_model.predict(X_test)

print("Hybrid Model Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
