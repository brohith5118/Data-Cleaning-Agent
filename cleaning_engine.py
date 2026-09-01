import pandas as pd
import json
import os

from data_ingestion import load_data


# ============================================================
# LOAD CLEANING PLAN
# ============================================================

def load_cleaning_plan(filename="cleaning_plan.json"):

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Cleaning plan not found: {filename}"
        )

    with open(filename, "r", encoding="utf-8") as file:
        plan = json.load(file)

    return plan


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(df):

    before = len(df)

    df = df.drop_duplicates()

    removed = before - len(df)

    return df, {
        "operation": "remove_duplicates",
        "rows_affected": removed
    }


# ============================================================
# FILL MISSING VALUES USING MEAN
# ============================================================

def fill_missing_mean(df, column):

    if column not in df.columns:
        return df, {
            "operation": "fill_missing_mean",
            "column": column,
            "rows_affected": 0,
            "status": "column_not_found"
        }

    missing_before = df[column].isnull().sum()

    if not pd.api.types.is_numeric_dtype(df[column]):

        return df, {
            "operation": "fill_missing_mean",
            "column": column,
            "rows_affected": 0,
            "status": "not_numeric"
        }

    mean_value = df[column].mean()

    df[column] = df[column].fillna(mean_value)

    return df, {
        "operation": "fill_missing_mean",
        "column": column,
        "rows_affected": int(missing_before),
        "fill_value": float(mean_value)
    }


# ============================================================
# FILL MISSING VALUES USING MEDIAN
# ============================================================

def fill_missing_median(df, column):

    if column not in df.columns:
        return df, {
            "operation": "fill_missing_median",
            "column": column,
            "rows_affected": 0,
            "status": "column_not_found"
        }

    missing_before = df[column].isnull().sum()

    if not pd.api.types.is_numeric_dtype(df[column]):

        return df, {
            "operation": "fill_missing_median",
            "column": column,
            "rows_affected": 0,
            "status": "not_numeric"
        }

    median_value = df[column].median()

    df[column] = df[column].fillna(median_value)

    return df, {
        "operation": "fill_missing_median",
        "column": column,
        "rows_affected": int(missing_before),
        "fill_value": float(median_value)
    }


# ============================================================
# FILL MISSING VALUES USING MODE
# ============================================================

def fill_missing_mode(df, column):

    if column not in df.columns:
        return df, {
            "operation": "fill_missing_mode",
            "column": column,
            "rows_affected": 0,
            "status": "column_not_found"
        }

    missing_before = df[column].isnull().sum()

    mode = df[column].mode()

    if mode.empty:

        return df, {
            "operation": "fill_missing_mode",
            "column": column,
            "rows_affected": 0,
            "status": "no_mode_available"
        }

    mode_value = mode.iloc[0]

    df[column] = df[column].fillna(mode_value)

    return df, {
        "operation": "fill_missing_mode",
        "column": column,
        "rows_affected": int(missing_before),
        "fill_value": mode_value
    }


# ============================================================
# DROP ROWS WITH MISSING VALUES
# ============================================================

def drop_missing_rows(df, column):

    if column not in df.columns:
        return df, {
            "operation": "drop_missing_rows",
            "column": column,
            "rows_affected": 0,
            "status": "column_not_found"
        }

    before = len(df)

    df = df.dropna(subset=[column])

    removed = before - len(df)

    return df, {
        "operation": "drop_missing_rows",
        "column": column,
        "rows_affected": removed
    }


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(df, column):

    if column not in df.columns:
        return df, {
            "operation": "normalize_text",
            "column": column,
            "rows_affected": 0,
            "status": "column_not_found"
        }

    if not (
        pd.api.types.is_object_dtype(df[column])
        or pd.api.types.is_string_dtype(df[column])
    ):

        return df, {
            "operation": "normalize_text",
            "column": column,
            "rows_affected": 0,
            "status": "not_text"
        }

    before = df[column].copy()

    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    changed = (before != df[column]).sum()

    return df, {
        "operation": "normalize_text",
        "column": column,
        "rows_affected": int(changed)
    }


# ============================================================
# STANDARDIZE CATEGORIES
# ============================================================

def standardize_categories(df, column):

    if column not in df.columns:
        return df, {
            "operation": "standardize_categories",
            "column": column,
            "rows_affected": 0,
            "status": "column_not_found"
        }

    before = df[column].copy()

    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    changed = (before != df[column]).sum()

    return df, {
        "operation": "standardize_categories",
        "column": column,
        "rows_affected": int(changed)
    }


# ============================================================
# FLAG OUTLIERS
# ============================================================

def flag_outliers(df, column):

    if column not in df.columns:
        return df, {
            "operation": "flag_outliers",
            "column": column,
            "rows_affected": 0,
            "status": "column_not_found"
        }

    if not pd.api.types.is_numeric_dtype(df[column]):

        return df, {
            "operation": "flag_outliers",
            "column": column,
            "rows_affected": 0,
            "status": "not_numeric"
        }

    values = df[column]

    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_mask = (
        (values < lower_bound)
        | (values > upper_bound)
    )

    flag_column = f"{column}_outlier"

    df[flag_column] = outlier_mask

    count = outlier_mask.sum()

    return df, {
        "operation": "flag_outliers",
        "column": column,
        "flag_column": flag_column,
        "rows_affected": int(count),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound)
    }


# ============================================================
# REMOVE OUTLIERS
# ============================================================

def remove_outliers(df, column):

    if column not in df.columns:
        return df, {
            "operation": "remove_outliers",
            "column": column,
            "rows_affected": 0,
            "status": "column_not_found"
        }

    if not pd.api.types.is_numeric_dtype(df[column]):

        return df, {
            "operation": "remove_outliers",
            "column": column,
            "rows_affected": 0,
            "status": "not_numeric"
        }

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    before = len(df)

    df = df[
        (df[column] >= lower_bound)
        & (df[column] <= upper_bound)
    ]

    removed = before - len(df)

    return df, {
        "operation": "remove_outliers",
        "column": column,
        "rows_affected": int(removed),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound)
    }


# ============================================================
# FLAG INVALID VALUES
# ============================================================

def flag_invalid_values(df, column):

    if column not in df.columns:
        return df, {
            "operation": "flag_invalid_values",
            "column": column,
            "rows_affected": 0,
            "status": "column_not_found"
        }

    # Heart dataset rules
    rules = {

        "age": (0, 120),

        "sex": (0, 1),

        "cp": (0, 3),

        "trestbps": (50, 250),

        "chol": (50, 700),

        "fbs": (0, 1),

        "restecg": (0, 2),

        "thalach": (50, 250),

        "exang": (0, 1),

        "oldpeak": (-10, 10),

        "slope": (0, 2),

        "ca": (0, 4),

        "thal": (0, 3),

        "target": (0, 1)
    }

    if column not in rules:

        return df, {
            "operation": "flag_invalid_values",
            "column": column,
            "rows_affected": 0,
            "status": "no_rule_available"
        }

    minimum, maximum = rules[column]

    invalid_mask = (
        (df[column] < minimum)
        | (df[column] > maximum)
    )

    flag_column = f"{column}_invalid"

    df[flag_column] = invalid_mask

    count = invalid_mask.sum()

    return df, {
        "operation": "flag_invalid_values",
        "column": column,
        "flag_column": flag_column,
        "rows_affected": int(count),
        "valid_range": f"{minimum} - {maximum}"
    }


# ============================================================
# EXECUTE ONE OPERATION
# ============================================================

def execute_operation(df, operation, column):

    if operation == "remove_duplicates":

        return remove_duplicates(df)

    elif operation == "fill_missing_mean":

        return fill_missing_mean(df, column)

    elif operation == "fill_missing_median":

        return fill_missing_median(df, column)

    elif operation == "fill_missing_mode":

        return fill_missing_mode(df, column)

    elif operation == "drop_missing_rows":

        return drop_missing_rows(df, column)

    elif operation == "normalize_text":

        return normalize_text(df, column)

    elif operation == "standardize_categories":

        return standardize_categories(df, column)

    elif operation == "flag_outliers":

        return flag_outliers(df, column)

    elif operation == "remove_outliers":

        return remove_outliers(df, column)

    elif operation == "flag_invalid_values":

        return flag_invalid_values(df, column)

    elif operation == "no_action":

        return df, {
            "operation": "no_action",
            "column": column,
            "rows_affected": 0
        }

    else:

        return df, {
            "operation": operation,
            "column": column,
            "rows_affected": 0,
            "status": "unsupported_operation"
        }


# ============================================================
# RUN CLEANING ENGINE
# ============================================================

def clean_dataset(df, plan):

    cleaning_results = []

    actions = plan.get("cleaning_plan", [])

    print("\n")
    print("=" * 70)
    print("                    CLEANING ENGINE")
    print("=" * 70)

    for i, action in enumerate(actions, start=1):

        operation = action.get("operation")
        column = action.get("column")

        print(f"\nExecuting Action {i}")
        print(f"  Operation : {operation}")
        print(f"  Column    : {column}")

        df, result = execute_operation(
            df,
            operation,
            column
        )

        cleaning_results.append(result)

        print(
            f"  Rows affected: "
            f"{result.get('rows_affected', 0)}"
        )

        if "status" in result:

            print(
                f"  Status: "
                f"{result['status']}"
            )

    return df, cleaning_results


# ============================================================
# SAVE CLEANING REPORT
# ============================================================

def save_cleaning_report(results):
    import os 
    os.makedirs("reports", exist_ok=True)

    with open(
        "reports/cleaning_report.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            default=str
        )

    print(
        "\nCleaning report saved to "
        "cleaning_report.json"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    input_file = "heart.csv"

    output_file = "cleaned_heart.csv"

    plan_file = "cleaning_plan.json"

    try:

        # ----------------------------------------------------
        # Load dataset
        # ----------------------------------------------------

        df = load_data(input_file)

        print(
            f"Original dataset: "
            f"{len(df)} rows, "
            f"{len(df.columns)} columns"
        )

        # ----------------------------------------------------
        # Load AI cleaning plan
        # ----------------------------------------------------

        plan = load_cleaning_plan(
            plan_file
        )

        print(
            "AI cleaning plan loaded successfully."
        )

        # ----------------------------------------------------
        # Execute cleaning
        # ----------------------------------------------------

        cleaned_df, results = clean_dataset(
            df,
            plan
        )

        # ----------------------------------------------------
        # Save cleaned dataset
        # ----------------------------------------------------

        cleaned_df.to_csv(
            output_file,
            index=False
        )

        print(
            f"\nCleaned dataset saved to "
            f"{output_file}"
        )

        print(
            f"Final dataset: "
            f"{len(cleaned_df)} rows, "
            f"{len(cleaned_df.columns)} columns"
        )

        # ----------------------------------------------------
        # Save report
        # ----------------------------------------------------

        save_cleaning_report(results)

        print("\nStage 5 completed successfully.")

    except Exception as e:

        print(
            f"\nError during cleaning: {e}"
        )