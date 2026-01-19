import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ==========================
# Load Dataset
# ==========================
dataset = pd.read_csv('./Placement_Prediction_data.csv')

# Drop unnecessary column
dataset.drop(['StudentId'], axis=1, inplace=True)

# Encode categorical columns
encoder = LabelEncoder()
columns_to_encode = ['Internship', 'Hackathon', 'PlacementStatus']

for col in columns_to_encode:
    dataset[col] = encoder.fit_transform(dataset[col])

# Features & Target
X = dataset.drop('PlacementStatus', axis=1)
y = dataset['PlacementStatus']

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ==========================
# Models
# ==========================
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC()
}

# ==========================
# Training & Evaluation
# ==========================
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append([name, round(acc * 100, 2)])

# ==========================
# Results Table
# ==========================
results_df = pd.DataFrame(results, columns=["Algorithm", "Accuracy (%)"])
print("\nModel Comparison:\n")
print(results_df.sort_values(by="Accuracy (%)", ascending=False))
