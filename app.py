from flask import Flask, render_template, request

from load_data import load_data, get_data_summary
from placement_eda import run_eda
from preprocessing import run_preprocessing
from modeltraining import run_model_training

from linearregression import (
    run_linear_regression,
    predict_salary,
    FEATURES as LINEAR_FEATURES
)

from logisticregression import (
    run_logistic_regression,
    predict_placement,
    FEATURES as LOGISTIC_FEATURES
)

from treebasedmodels import (
    run_tree_model,
    FEATURES as TREE_FEATURES
)


app = Flask(__name__)


# ==========================================================
# HOME
# ==========================================================

@app.route("/")
def index():

    error = None
    summary = None

    try:

        data = load_data()

        summary = get_data_summary(
            data
        )

    except Exception as e:

        error = f"Error: {e}"

    return render_template(
        "index.html",
        active="data-loading",
        summary=summary,
        error=error
    )


# ==========================================================
# DATA LOADING
# ==========================================================

@app.route("/data-loading")
def data_loading():

    error = None
    summary = None

    try:

        data = load_data()

        summary = get_data_summary(
            data
        )

    except Exception as e:

        error = f"Error: {e}"

    return render_template(
        "index.html",
        active="data-loading",
        summary=summary,
        error=error
    )


# ==========================================================
# EDA
# ==========================================================

@app.route("/eda")
def eda_page():

    error = None
    results = None

    try:

        results = run_eda()

    except Exception as e:

        error = f"Error: {e}"

    return render_template(
        "eda.html",
        active="eda",
        results=results,
        error=error
    )


# ==========================================================
# PREPROCESSING
# ==========================================================

@app.route("/preprocessing")
def preprocessing_page():

    error = None
    results = None

    try:

        results = run_preprocessing()

    except Exception as e:

        error = f"Error: {e}"

    return render_template(
        "preprocessing.html",
        active="preprocessing",
        results=results,
        error=error
    )


# ==========================================================
# MODEL TRAINING
# ==========================================================

@app.route("/model-training")
def model_training_page():

    error = None
    results = None

    try:

        results = run_model_training()

    except Exception as e:

        error = f"Error: {e}"

    return render_template(
        "modeltraining.html",
        active="model-training",
        results=results,
        error=error
    )


# ==========================================================
# LINEAR REGRESSION
# ==========================================================

@app.route("/linear-regression")
def linear_regression_page():

    return render_template(
        "linearregression.html",
        active="linear-regression",
        results=None,
        selected_model=None,
        error=None,
        features=LINEAR_FEATURES
    )


# ==========================================================
# LINEAR REGRESSION - NO REGULARIZATION
# ==========================================================

@app.route("/linear-regression/without-regularization")
def linear_regression_without():

    error = None
    results = None

    try:

        results = run_linear_regression(
            "none"
        )

    except Exception as e:

        error = f"Error: {e}"

    return render_template(
        "linearregression.html",
        active="linear-regression",
        results=results,
        selected_model="none",
        error=error,
        features=LINEAR_FEATURES
    )


# ==========================================================
# LINEAR REGRESSION - RIDGE / L2
# ==========================================================

@app.route("/linear-regression/with-regularization")
def linear_regression_with():

    error = None
    results = None

    try:

        results = run_linear_regression(
            "ridge"
        )

    except Exception as e:

        error = f"Error: {e}"

    return render_template(
        "linearregression.html",
        active="linear-regression",
        results=results,
        selected_model="ridge",
        error=error,
        features=LINEAR_FEATURES
    )


# ==========================================================
# LINEAR REGRESSION - LASSO / L1
# ==========================================================

@app.route("/linear-regression/lasso")
def linear_regression_lasso():

    error = None
    results = None

    try:

        results = run_linear_regression(
            "lasso"
        )

    except Exception as e:

        error = f"Error: {e}"

    return render_template(
        "linearregression.html",
        active="linear-regression",
        results=results,
        selected_model="lasso",
        error=error,
        features=LINEAR_FEATURES
    )


# ==========================================================
# SALARY PREDICTION
# ==========================================================

@app.route(
    "/predict-salary/<model_type>",
    methods=["POST"]
)
def predict_salary_route(model_type):

    error = None
    results = None

    try:

        values = []

        for feature in LINEAR_FEATURES:

            value = float(
                request.form.get(
                    feature,
                    0
                )
            )

            values.append(
                value
            )

        results = predict_salary(
            values,
            model_type
        )

    except Exception as e:

        error = f"Prediction Error: {e}"

    return render_template(
        "linearregression.html",
        active="linear-regression",
        results=results,
        selected_model=model_type,
        error=error,
        features=LINEAR_FEATURES
    )


# ==========================================================
# LOGISTIC REGRESSION
# ==========================================================

@app.route("/logistic-regression")
def logistic_regression_page():

    return render_template(
        "logisticregression.html",
        active="logistic-regression",
        results=None,
        selected_model=None,
        error=None,
        features=LOGISTIC_FEATURES
    )


# ==========================================================
# LOGISTIC REGRESSION - NO REGULARIZATION
# ==========================================================

@app.route("/logistic-regression/without-regularization")
def logistic_regression_without():

    error = None
    results = None

    try:

        results = run_logistic_regression(
            "none"
        )

    except Exception as e:

        error = f"Error: {e}"

    return render_template(
        "logisticregression.html",
        active="logistic-regression",
        results=results,
        selected_model="none",
        error=error,
        features=LOGISTIC_FEATURES
    )


# ==========================================================
# LOGISTIC REGRESSION - RIDGE / L2
# ==========================================================

@app.route("/logistic-regression/with-regularization")
def logistic_regression_with():

    error = None
    results = None

    try:

        results = run_logistic_regression(
            "l2"
        )

    except Exception as e:

        error = f"Error: {e}"

    return render_template(
        "logisticregression.html",
        active="logistic-regression",
        results=results,
        selected_model="l2",
        error=error,
        features=LOGISTIC_FEATURES
    )


# ==========================================================
# LOGISTIC REGRESSION - LASSO / L1
# ==========================================================

@app.route("/logistic-regression/lasso")
def logistic_regression_lasso():

    error = None
    results = None

    try:

        results = run_logistic_regression(
            "lasso"
        )

    except Exception as e:

        error = f"Error: {e}"

    return render_template(
        "logisticregression.html",
        active="logistic-regression",
        results=results,
        selected_model="lasso",
        error=error,
        features=LOGISTIC_FEATURES
    )


# ==========================================================
# PLACEMENT PREDICTION
# ==========================================================

@app.route(
    "/predict-placement/<model_type>",
    methods=["POST"]
)
def predict_placement_route(model_type):

    error = None
    results = None

    try:

        values = []

        for feature in LOGISTIC_FEATURES:

            value = float(
                request.form.get(
                    feature,
                    0
                )
            )

            values.append(
                value
            )

        results = predict_placement(
            values,
            model_type
        )

    except Exception as e:

        error = f"Prediction Error: {e}"

    return render_template(
        "logisticregression.html",
        active="logistic-regression",
        results=results,
        selected_model=model_type,
        error=error,
        features=LOGISTIC_FEATURES
    )


# ==========================================================
# TREE BASED MODELS - ID3
# ==========================================================

@app.route("/tree-based-models/id3")
def id3_model():

    try:

        results = run_tree_model(
            "id3"
        )

        return render_template(
            "treebasedmodels.html",
            active="tree-based-models",
            results=results,
            selected_model="id3",
            error=None
        )

    except Exception as e:

        return render_template(
            "treebasedmodels.html",
            active="tree-based-models",
            results=None,
            selected_model="id3",
            error=str(e)
        )


# ==========================================================
# TREE BASED MODELS - RANDOM FOREST
# ==========================================================

@app.route("/tree-based-models/random-forest")
def random_forest_model():

    try:

        results = run_tree_model(
            "random_forest"
        )

        return render_template(
            "treebasedmodels.html",
            active="tree-based-models",
            results=results,
            selected_model="random_forest",
            error=None
        )

    except Exception as e:

        return render_template(
            "treebasedmodels.html",
            active="tree-based-models",
            results=None,
            selected_model="random_forest",
            error=str(e)
        )


# ==========================================================
# TREE BASED MODELS - ADABOOST
# ==========================================================

@app.route("/tree-based-models/adaboost")
def adaboost_model():

    try:

        results = run_tree_model(
            "adaboost"
        )

        return render_template(
            "treebasedmodels.html",
            active="tree-based-models",
            results=results,
            selected_model="adaboost",
            error=None
        )

    except Exception as e:

        return render_template(
            "treebasedmodels.html",
            active="tree-based-models",
            results=None,
            selected_model="adaboost",
            error=str(e)
        )


# ==========================================================
# TREE BASED MODELS - GRADIENT BOOSTING
# ==========================================================

@app.route("/tree-based-models/gradient-boosting")
def gradient_boosting_model():

    try:

        results = run_tree_model(
            "gradient_boosting"
        )

        return render_template(
            "treebasedmodels.html",
            active="tree-based-models",
            results=results,
            selected_model="gradient_boosting",
            error=None
        )

    except Exception as e:

        return render_template(
            "treebasedmodels.html",
            active="tree-based-models",
            results=None,
            selected_model="gradient_boosting",
            error=str(e)
        )


# ==========================================================
# TREE BASED MODELS - XGBOOST
# ==========================================================

@app.route("/tree-based-models/xgboost")
def xgboost_model():

    try:

        results = run_tree_model(
            "xgboost"
        )

        return render_template(
            "treebasedmodels.html",
            active="tree-based-models",
            results=results,
            selected_model="xgboost",
            error=None
        )

    except Exception as e:

        return render_template(
            "treebasedmodels.html",
            active="tree-based-models",
            results=None,
            selected_model="xgboost",
            error=str(e)
        )


# ==========================================================
# TREE BASED MODELS - LIGHTGBM
# ==========================================================

@app.route("/tree-based-models/lightgbm")
def lightgbm_model():

    try:

        results = run_tree_model(
            "lightgbm"
        )

        return render_template(
            "treebasedmodels.html",
            active="tree-based-models",
            results=results,
            selected_model="lightgbm",
            error=None
        )

    except Exception as e:

        return render_template(
            "treebasedmodels.html",
            active="tree-based-models",
            results=None,
            selected_model="lightgbm",
            error=str(e)
        )


# ==========================================================
# TREE BASED MODELS MAIN PAGE
# ==========================================================

@app.route("/tree-based-models")
def tree_based_models_page():

    return render_template(
        "treebasedmodels.html",
        active="tree-based-models",
        results=None,
        selected_model=None,
        error=None
    )


# ==========================================================
# RUN APPLICATION
# ==========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        port=5001
    )