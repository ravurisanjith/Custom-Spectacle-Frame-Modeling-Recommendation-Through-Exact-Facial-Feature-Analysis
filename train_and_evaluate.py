import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

data_path = 'face_shape_features.csv'

def main():
    # Load CSV
    df = pd.read_csv(data_path)

    # Identify and drop any object-type columns except the label 'face_shape'
    drop_cols = [col for col in df.columns if df[col].dtype == 'object' and col != 'face_shape']
    if drop_cols:
        print(f"Dropping non-numeric columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    # Separate features and label
    X = df.drop('face_shape', axis=1)
    y = df['face_shape']

    # Encode labels into integers
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Train/test split with stratified classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model and hyperparameter grid for Random Forest
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }

    # GridSearchCV to find the best hyperparameters
    grid_search = GridSearchCV(rfc, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best hyperparameters: {grid_search.best_params_}")

    y_pred = best_model.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Save model, scaler, and label encoder
    joblib.dump(best_model, 'face_shape_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    print("Model, scaler, and label encoder saved.")

if __name__ == "__main__":
    main()
