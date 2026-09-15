import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


DATASET_PATH = r"C:\Users\hp\Documents\ML\ML\placement_predict_50k Dataset (3)(in).csv"


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
# TRAIN LINEAR REGRESSION MODEL
# ==========================================================

def train_model(regularization="none"):

    df = pd.read_csv(DATASET_PATH)

    X = df[FEATURES].copy()
    y = df["Salary Package"].copy()

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
        random_state=42
    )

    # ======================================================
    # MODEL SELECTION
    # ======================================================

    if regularization == "ridge":

        model = Ridge(
            alpha=1.0
        )

        model_name = "Ridge Regression"
        regularization_name = "L2 Regularization"

    elif regularization == "lasso":

        model = Lasso(
            alpha=0.01,
            max_iter=10000
        )

        model_name = "Lasso Regression"
        regularization_name = "L1 Regularization"

    else:

        model = LinearRegression()

        model_name = "Linear Regression"
        regularization_name = "None"

    # ======================================================
    # TRAIN MODEL
    # ======================================================

    model.fit(
        X_train,
        y_train
    )

    # Prediction
    y_pred = model.predict(
        X_test
    )

    # ======================================================
    # PERFORMANCE METRICS
    # ======================================================

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = mse ** 0.5

    r2 = r2_score(
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

        "target_column": "Salary Package",

        "total_rows": len(df),

        "total_features": len(FEATURES),

        "training_rows": len(X_train),

        "testing_rows": len(X_test),

        "mae": round(
            mae,
            4
        ),

        "mse": round(
            mse,
            4
        ),

        "rmse": round(
            rmse,
            4
        ),

        "r2": round(
            r2,
            4
        )
    }


# ==========================================================
# SALARY PREDICTION
# ==========================================================

def predict_salary(
    values,
    regularization="none"
):

    results = train_model(
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

    results["prediction"] = round(
        float(prediction),
        2
    )

    return results


# ==========================================================
# RUN LINEAR REGRESSION
# ==========================================================

def run_linear_regression(
    regularization="none"
):

    return train_model(
        regularization
    )