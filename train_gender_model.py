import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import joblib

# Check if the file exists
if not os.path.exists("voice.csv"):
    raise FileNotFoundError("The file 'voice.csv' was not found in the current directory.")

# Load voice gender dataset
df = pd.read_csv("voice.csv")
if df.empty:
    raise ValueError("The file 'voice.csv' is empty. Please provide a valid dataset.")

print(df.info())
print(df.head())
print(df.isnull().sum())
print(df["label"].value_counts())

X = df.drop("label", axis=1)
y = df["label"].map({"male": 0, "female": 1})

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
gender_model = grid_search.best_estimator_

# Save
joblib.dump(gender_model, "model.pkl")
print("Gender model saved as model.pkl ✅")

y_pred = gender_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Male", "Female"]))



test_sample = X_test[0].reshape(1, -1)
print("Predicted gender:", "Female" if gender_model.predict(test_sample)[0] == 1 else "Male")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Male", "Female"])
disp.plot()
