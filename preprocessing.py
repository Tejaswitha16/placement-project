import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    r"C:\Users\hp\Documents\ML\ML\placement_predict_50k Dataset (3)(in).csv"
)


# ==========================================================
# RUN PREPROCESSING
# ==========================================================

def run_preprocessing():

    # ==========================================================
    # 1. LOAD DATASET
    # ==========================================================

    df = pd.read_csv(DATASET_PATH)

    original_rows, original_columns = df.shape


    # ==========================================================
    # 2. MISSING VALUES BEFORE PREPROCESSING
    # ==========================================================

    missing_before_total = int(
        df.isnull().sum().sum()
    )

    missing_before = df.isnull().sum()
    missing_before = missing_before[
        missing_before > 0
    ]

    missing_before_data = [
        {
            "column": column,
            "count": int(count)
        }
        for column, count in missing_before.items()
    ]


    # ==========================================================
    # 3. DUPLICATE ROWS
    # ==========================================================

    duplicate_count = int(
        df.duplicated().sum()
    )

    df = df.drop_duplicates()

    rows_after_duplicates = len(df)

    duplicates_removed = (
        original_rows - rows_after_duplicates
    )


    # ==========================================================
    # 4. IDENTIFY NUMERICAL AND CATEGORICAL COLUMNS
    # ==========================================================

    numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()


    # ==========================================================
    # 5. HANDLE MISSING VALUES
    # ==========================================================

    numerical_imputed = []

    numerical_imputation_count = 0

    for column in numerical_columns:

        missing_count = int(
            df[column].isnull().sum()
        )

        if missing_count > 0:

            df[column] = df[column].fillna(
                df[column].median()
            )

            numerical_imputed.append(column)

            numerical_imputation_count += missing_count


    categorical_imputed = []

    categorical_imputation_count = 0

    for column in categorical_columns:

        missing_count = int(
            df[column].isnull().sum()
        )

        if missing_count > 0:

            mode_value = df[column].mode()

            if len(mode_value) > 0:

                df[column] = df[column].fillna(
                    mode_value[0]
                )

            else:

                df[column] = df[column].fillna(
                    "Unknown"
                )

            categorical_imputed.append(column)

            categorical_imputation_count += missing_count


    total_imputed = (
        numerical_imputation_count
        + categorical_imputation_count
    )


    # ==========================================================
    # 6. ONE-HOT ENCODING
    # ==========================================================

    encoded_columns = categorical_columns.copy()

    if len(categorical_columns) > 0:

        df = pd.get_dummies(
            df,
            columns=categorical_columns,
            drop_first=True
        )


    # ==========================================================
    # 7. CONVERT BOOLEAN TO INTEGER
    # ==========================================================

    bool_columns = df.select_dtypes(
        include=["bool"]
    ).columns

    bool_columns_count = len(
        bool_columns
    )

    for column in bool_columns:

        df[column] = df[column].astype(int)


    # ==========================================================
    # 8. STANDARD SCALING
    # ==========================================================

    final_numerical_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()


    # Do NOT scale target columns
    target_columns = [
        "PlacementStatus",
        "Salary Package"
    ]


    scaling_columns = [
        column
        for column in final_numerical_columns
        if column not in target_columns
    ]


    if len(scaling_columns) > 0:

        scaler = StandardScaler()

        df[scaling_columns] = scaler.fit_transform(
            df[scaling_columns]
        )


    # ==========================================================
    # 9. MISSING VALUES AFTER PREPROCESSING
    # ==========================================================

    missing_after = int(
        df.isnull().sum().sum()
    )


    # ==========================================================
    # 10. FINAL DATASET INFORMATION
    # ==========================================================

    final_rows, final_columns = df.shape


    # ==========================================================
    # 11. FINAL COLUMN NAMES
    # ==========================================================

    final_column_names = df.columns.tolist()


    # ==========================================================
    # 12. PREPROCESSED DATA PREVIEW
    # ==========================================================

    preview = df.head(10).round(3).to_html(
        classes="data-table",
        index=False
    )


    # ==========================================================
    # 13. PREPROCESSING STATUS
    # ==========================================================

    if missing_after == 0:

        preprocessing_status = "Preprocessing Completed Successfully"

    else:

        preprocessing_status = "Preprocessing Completed with Missing Values"


    # ==========================================================
    # 14. RETURN RESULTS TO FLASK
    # ==========================================================

    return {

        # ------------------------------------------------------
        # DATASET INFORMATION
        # ------------------------------------------------------

        "original_rows": original_rows,

        "original_columns": original_columns,

        "final_rows": final_rows,

        "final_columns": final_columns,


        # ------------------------------------------------------
        # DUPLICATES
        # ------------------------------------------------------

        "duplicate_count": duplicate_count,

        "duplicates_removed": duplicates_removed,

        "rows_after_duplicates": rows_after_duplicates,


        # ------------------------------------------------------
        # MISSING VALUES
        # ------------------------------------------------------

        "missing_before_total": missing_before_total,

        "missing_before": missing_before_data,

        "missing_after": missing_after,

        "total_imputed": total_imputed,

        "numerical_imputation_count":
            numerical_imputation_count,

        "categorical_imputation_count":
            categorical_imputation_count,


        # ------------------------------------------------------
        # COLUMN INFORMATION
        # ------------------------------------------------------

        "numerical_columns":
            numerical_columns,

        "categorical_columns":
            categorical_columns,

        "numerical_count":
            len(numerical_columns),

        "categorical_count":
            len(categorical_columns),


        # ------------------------------------------------------
        # ENCODING
        # ------------------------------------------------------

        "encoded_columns":
            encoded_columns,

        "encoded_count":
            len(encoded_columns),

        "bool_columns_count":
            bool_columns_count,


        # ------------------------------------------------------
        # SCALING
        # ------------------------------------------------------

        "scaling_columns":
            scaling_columns,

        "scaling_count":
            len(scaling_columns),


        # ------------------------------------------------------
        # TARGET
        # ------------------------------------------------------

        "target_columns":
            target_columns,


        # ------------------------------------------------------
        # FINAL DATASET
        # ------------------------------------------------------

        "final_column_names":
            final_column_names,

        "preview":
            preview,


        # ------------------------------------------------------
        # STATUS
        # ------------------------------------------------------

        "preprocessing_status":
            preprocessing_status,


        # ------------------------------------------------------
        # PROCESSED DATA
        # ------------------------------------------------------

        "data":
            df
    }