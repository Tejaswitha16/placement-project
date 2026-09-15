import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler


DATASET_PATH = r"D:\ML\placement_predict_50k Dataset (3)(in).csv"


FEATURES = [
    "SGPA_Sem1",
    "SGPA_Sem2",
    "SGPA_Sem3",
    "SGPA_Sem4",
    "SGPA_Sem5",
    "SGPA_Sem6",
    "SGPA_Sem7",
    "SGPA_Sem8",
    "CGPA",
    "AttendancePercent",
    "Internships",
    "Projects",
    "Workshops",
    "Certifications",
    "Publications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "CodingTestScore",
    "MockInterviewScore",
    "ExtraCurricular"
]


# ==========================================================
# TRAIN LOGISTIC REGRESSION MODEL
# ==========================================================

def train_logistic_model(regularization="none"):

    df = pd.read_csv(DATASET_PATH)

    X = df[FEATURES].copy()

    y = df["PlacementStatus"].copy()

    # Handle missing values
    X = X.fillna(X.median())

    # Standardization
    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ======================================================
    # MODEL SELECTION
    # ======================================================

    if regularization == "l2":

        # L2 Regularization
        model = LogisticRegression(
            penalty="l2",
            C=1.0,
            max_iter=1000
        )

        model_name = "Logistic Regression"

        regularization_name = "L2 Regularization"

    elif regularization == "lasso":

        # L1 / Lasso Regularization
        model = LogisticRegression(
            penalty="l1",
            C=1.0,
            solver="liblinear",
            max_iter=1000
        )

        model_name = "Logistic Regression"

        regularization_name = "L1 (Lasso) Regularization"

    else:

        # No Regularization
        model = LogisticRegression(
            penalty=None,
            max_iter=1000
        )

        model_name = "Logistic Regression"

        regularization_name = "None"

    # ======================================================
    # TRAIN MODEL
    # ======================================================

    model.fit(
        X_train,
        y_train
    )

    # ======================================================
    # PREDICTION
    # ======================================================

    y_pred = model.predict(
        X_test
    )

    # ======================================================
    # PERFORMANCE METRICS
    # ======================================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

    matrix = confusion_matrix(
        y_test,
        y_pred
    )

    # ======================================================
    # RETURN RESULTS
    # ======================================================

    return {
        "model": model,
        "scaler": scaler,

        "model_name": model_name,

        "regularization": regularization_name,

        "target_column": "PlacementStatus",

        "total_rows": len(df),

        "total_features": len(FEATURES),

        "training_rows": len(X_train),

        "testing_rows": len(X_test),

        "accuracy": round(
            accuracy * 100,
            2
        ),

        "precision": round(
            precision * 100,
            2
        ),

        "recall": round(
            recall * 100,
            2
        ),

        "f1": round(
            f1 * 100,
            2
        ),

        "confusion_matrix": matrix.tolist()
    }


# ==========================================================
# PLACEMENT PREDICTION
# ==========================================================

def predict_placement(
    values,
    regularization="none"
):

    results = train_logistic_model(
        regularization
    )

    model = results["model"]

    scaler = results["scaler"]

    input_data = pd.DataFrame(
        [values],
        columns=FEATURES
    )

    input_data = input_data.fillna(
        input_data.median()
    )

    input_scaled = scaler.transform(
        input_data
    )

    prediction = model.predict(
        input_scaled
    )[0]

    probability = model.predict_proba(
        input_scaled
    )[0][1]

    results["prediction"] = int(
        prediction
    )

    results["probability"] = round(
        float(probability * 100),
        2
    )

    return results


# ==========================================================
# RUN LOGISTIC REGRESSION
# ==========================================================

def run_logistic_regression(
    regularization="none"
):

    return train_logistic_model(
        regularization
    )