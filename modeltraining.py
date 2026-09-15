from preprocessing import run_preprocessing

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def run_model_training():

    # ==========================================
    # 1. Get PREPROCESSED data
    # ==========================================

    preprocessing_results = run_preprocessing()

    df = preprocessing_results["data"].copy()

    # ==========================================
    # 2. Target column
    # ==========================================

    target_column = "PlacementStatus"

    if target_column not in df.columns:
        raise ValueError(
            "PlacementStatus column not found."
        )

    # ==========================================
    # 3. Separate X and y
    # ==========================================

    X = df.drop(
        columns=[
            "PlacementStatus",
            "Salary Package",
            "StudentID",
            "IsAnomaly"
        ],
        errors="ignore"
    )

    y = df["PlacementStatus"]

    # ==========================================
    # 4. Convert target to numeric if necessary
    # ==========================================

    if y.dtype == "object":

        unique_values = y.unique()

        mapping = {
            value: index
            for index, value in enumerate(unique_values)
        }

        y = y.map(mapping)

    # ==========================================
    # 5. Train-Test Split
    # ==========================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # ==========================================
    # 6. Create Logistic Regression Model
    # ==========================================

    model = LogisticRegression(
        max_iter=1000
    )

    # ==========================================
    # 7. Train Model
    # ==========================================

    model.fit(
        X_train,
        y_train
    )

    # ==========================================
    # 8. Prediction
    # ==========================================

    y_pred = model.predict(X_test)

    # ==========================================
    # 9. Accuracy
    # ==========================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # ==========================================
    # 10. Classification Report
    # ==========================================

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    # ==========================================
    # 11. Return results
    # ==========================================

    return {

        "target_column": target_column,

        "total_rows": len(df),

        "total_features": X.shape[1],

        "training_rows": len(X_train),

        "testing_rows": len(X_test),

        "accuracy": round(
            accuracy * 100,
            2
        ),

        "classification_report": report
    }