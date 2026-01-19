import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load CSV file
# Make sure your CSV has the same column names as in your example
df = pd.read_csv("Placement_Prediction_data.csv")

# Step 3: Encode categorical variables
categorical_cols = ['Internship', 'Hackathon']
le_dict = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le  # Save encoder if needed later

# Encode target variable
le_target = LabelEncoder()
df['PlacementStatus'] = le_target.fit_transform(df['PlacementStatus'])  # NotPlaced=0, Placed=1

# Step 4: Split features and target
X = df.drop(columns=['StudentId', 'PlacementStatus'])
y = df['PlacementStatus']

# Step 5: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))