import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


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


def train_tree_model(model_type):

    # ==========================================================
    # LOAD DATA
    # ==========================================================

    df = pd.read_csv(DATASET_PATH)


    # ==========================================================
    # FEATURES AND TARGET
    # ==========================================================

    X = df[FEATURES].copy()

    y = df["PlacementStatus"].copy()


    # ==========================================================
    # HANDLE MISSING VALUES
    # ==========================================================

    X = X.fillna(X.median())


    # ==========================================================
    # SCALING
    # ==========================================================

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)


    # ==========================================================
    # TRAIN TEST SPLIT
    # ==========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )


    # ==========================================================
    # SELECT MODEL
    # ==========================================================

    if model_type == "id3":

        model = DecisionTreeClassifier(
            criterion="entropy",
            random_state=42
        )

        model_name = "ID3 Decision Tree"


    elif model_type == "random_forest":

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        model_name = "Random Forest"


    elif model_type == "adaboost":

        model = AdaBoostClassifier(
            n_estimators=100,
            random_state=42
        )

        model_name = "AdaBoost"


    elif model_type == "gradient_boosting":

        model = GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        )

        model_name = "Gradient Boosting"


    elif model_type == "xgboost":

        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss"
        )

        model_name = "XGBoost"


    elif model_type == "lightgbm":

        model = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=-1,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        )

        model_name = "LightGBM"


    else:

        raise ValueError(
            "Invalid tree model selected."
        )


    # ==========================================================
    # TRAIN MODEL
    # ==========================================================

    model.fit(
        X_train,
        y_train
    )


    # ==========================================================
    # PREDICTION
    # ==========================================================

    y_pred = model.predict(X_test)


    # ==========================================================
    # PERFORMANCE METRICS
    # ==========================================================

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


    # ==========================================================
    # CONFUSION MATRIX
    # ==========================================================

    matrix = confusion_matrix(
        y_test,
        y_pred
    )


    # ==========================================================
    # FEATURE IMPORTANCE
    # ==========================================================

    feature_importance = []


    if hasattr(
        model,
        "feature_importances_"
    ):

        for feature, importance in zip(
            FEATURES,
            model.feature_importances_
        ):

            feature_importance.append({

                "feature": feature,

                "importance": round(
                    float(importance),
                    4
                )

            })


        # Highest importance first

        feature_importance.sort(
            key=lambda x: x["importance"],
            reverse=True
        )


    # ==========================================================
    # RETURN RESULTS
    # ==========================================================

    return {

        "model": model,

        "scaler": scaler,

        "model_name": model_name,

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

        "confusion_matrix": matrix.tolist(),

        "feature_importance": feature_importance

    }


def run_tree_model(model_type):

    return train_tree_model(
        model_type
    )