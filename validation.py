import pandas as pd
import json
import os


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(filename):

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Dataset not found: {filename}"
        )

    return pd.read_csv(filename)


# ============================================================
# COUNT MISSING VALUES
# ============================================================

def count_missing_values(df):

    total = int(df.isnull().sum().sum())

    return total


# ============================================================
# COUNT DUPLICATES
# ============================================================

def count_duplicates(df):

    return int(df.duplicated().sum())


# ============================================================
# COUNT OUTLIERS
# ============================================================

def count_outliers(df):

    total_outliers = 0

    numerical_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numerical_columns:

        # Ignore columns created by the cleaning engine
        if column.endswith("_outlier"):
            continue

        if column.endswith("_invalid"):
            continue

        values = df[column].dropna()

        if len(values) == 0:
            continue

        # Don't use IQR on binary columns
        if values.nunique() <= 2:
            continue

        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)

        IQR = Q3 - Q1

        # If IQR is zero, IQR-based detection is not useful
        if IQR == 0:
            continue

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = values[
            (values < lower_bound) |
            (values > upper_bound)
        ]

        total_outliers += len(outliers)

    return int(total_outliers)


# ============================================================
# COUNT INVALID HEART DATA VALUES
# ============================================================

def count_invalid_values(df):

    range_rules = {

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

    total_invalid = 0

    for column, (minimum, maximum) in range_rules.items():

        if column not in df.columns:
            continue

        invalid = df[
            (df[column] < minimum) |
            (df[column] > maximum)
        ]

        total_invalid += len(invalid)

    return int(total_invalid)


# ============================================================
# DATASET METRICS
# ============================================================

def calculate_metrics(df):

    metrics = {

        "rows": int(df.shape[0]),

        "columns": int(df.shape[1]),

        "missing_values": count_missing_values(df),

        "duplicate_rows": count_duplicates(df),

        "outliers": count_outliers(df),

        "invalid_values": count_invalid_values(df)
    }

    return metrics


# ============================================================
# CALCULATE QUALITY SCORE
# ============================================================

def calculate_quality_score(metrics):

    rows = metrics["rows"]

    if rows == 0:
        return 0

    # --------------------------------------------------------
    # Missing value score
    # --------------------------------------------------------

    missing_rate = (
        metrics["missing_values"] / rows
    )

    missing_score = max(
        0,
        100 - (missing_rate * 100)
    )

    # --------------------------------------------------------
    # Duplicate score
    # --------------------------------------------------------

    duplicate_rate = (
        metrics["duplicate_rows"] / rows
    )

    duplicate_score = max(
        0,
        100 - (duplicate_rate * 100)
    )

    # --------------------------------------------------------
    # Outlier score
    #
    # Outliers are NOT automatically treated as errors.
    # Therefore, they receive a smaller penalty.
    # --------------------------------------------------------

    outlier_rate = (
        metrics["outliers"] / rows
    )

    outlier_score = max(
        0,
        100 - (outlier_rate * 20)
    )

    # --------------------------------------------------------
    # Invalid value score
    # --------------------------------------------------------

    invalid_rate = (
        metrics["invalid_values"] / rows
    )

    invalid_score = max(
        0,
        100 - (invalid_rate * 100)
    )

    # --------------------------------------------------------
    # Overall score
    # --------------------------------------------------------

    score = (
        missing_score * 0.30
        + duplicate_score * 0.25
        + outlier_score * 0.15
        + invalid_score * 0.30
    )

    return round(score, 2)


# ============================================================
# COMPARE BEFORE AND AFTER
# ============================================================

def compare_metrics(before, after):

    comparison = {}

    for key in before:

        comparison[key] = {
            "before": before[key],
            "after": after[key],
            "change": after[key] - before[key]
        }

    return comparison


# ============================================================
# DETERMINE RESULT
# ============================================================

def determine_result(
    before_score,
    after_score,
    before,
    after
):

    # --------------------------------------------------------
    # Check how many rows were removed
    # --------------------------------------------------------

    rows_removed = (
        before["rows"] - after["rows"]
    )

    if before["rows"] > 0:

        removal_percentage = (
            rows_removed / before["rows"]
        ) * 100

    else:

        removal_percentage = 0


    # --------------------------------------------------------
    # Reject excessive data loss
    # --------------------------------------------------------

    if removal_percentage > 20:

        return (
            "REJECT",
            f"{removal_percentage:.2f}% of rows were removed."
        )


    # --------------------------------------------------------
    # Quality score improved
    # --------------------------------------------------------

    if after_score > before_score:

        return (
            "ACCEPT",
            "Dataset quality score improved."
        )


    # --------------------------------------------------------
    # No score improvement, but columns were added
    # --------------------------------------------------------

    if (
        after["columns"] > before["columns"]
        and after["rows"] == before["rows"]
    ):

        columns_added = (
            after["columns"] - before["columns"]
        )

        return (
            "ACCEPT",
            f"{columns_added} diagnostic columns were added "
            "without removing any data."
        )


    # --------------------------------------------------------
    # No changes
    # --------------------------------------------------------

    if (
        before["rows"] == after["rows"]
        and before["columns"] == after["columns"]
        and before["missing_values"] == after["missing_values"]
        and before["duplicate_rows"] == after["duplicate_rows"]
    ):

        return (
            "REVIEW",
            "No meaningful cleaning changes were made."
        )


    # --------------------------------------------------------
    # Score decreased
    # --------------------------------------------------------

    if after_score < before_score:

        return (
            "REJECT",
            "Dataset quality score decreased."
        )


    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    return (
        "REVIEW",
        "Dataset should be manually reviewed."
    )


# ============================================================
# PRINT VALIDATION REPORT
# ============================================================

def print_validation_report(
    before,
    after,
    before_score,
    after_score,
    comparison,
    result,
    reason
):

    print("\n")
    print("=" * 75)
    print("                  VALIDATION REPORT")
    print("=" * 75)

    print("\nBEFORE CLEANING")
    print("-" * 75)

    print(f"Rows             : {before['rows']}")
    print(f"Columns          : {before['columns']}")
    print(f"Missing values   : {before['missing_values']}")
    print(f"Duplicate rows   : {before['duplicate_rows']}")
    print(f"Outliers         : {before['outliers']}")
    print(f"Invalid values   : {before['invalid_values']}")

    print("\nAFTER CLEANING")
    print("-" * 75)

    print(f"Rows             : {after['rows']}")
    print(f"Columns          : {after['columns']}")
    print(f"Missing values   : {after['missing_values']}")
    print(f"Duplicate rows   : {after['duplicate_rows']}")
    print(f"Outliers         : {after['outliers']}")
    print(f"Invalid values   : {after['invalid_values']}")

    print("\nQUALITY SCORES")
    print("-" * 75)

    print(f"Before score     : {before_score}/100")
    print(f"After score      : {after_score}/100")

    difference = after_score - before_score

    print(
        f"Score change     : {difference:+.2f}"
    )

    print("\nCHANGES")
    print("-" * 75)

    for metric, values in comparison.items():

        print(
            f"{metric:18} "
            f"{values['before']} → "
            f"{values['after']} "
            f"({values['change']:+})"
        )

    print("\nVALIDATION RESULT")
    print("-" * 75)

    print(f"Decision : {result}")
    print(f"Reason   : {reason}")

    print("=" * 75)


# ============================================================
# SAVE VALIDATION REPORT
# ============================================================

def save_validation_report(
    before,
    after,
    before_score,
    after_score,
    comparison,
    result,
    reason
):

    report = {

        "before": before,

        "after": after,

        "quality_score": {

            "before": before_score,

            "after": after_score,

            "change": round(
                after_score - before_score,
                2
            )
        },

        "comparison": comparison,

        "decision": result,

        "reason": reason
    }

    import os
    os.makedirs("reports", exist_ok=True)
    with open(
        "reports/validation_report.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print(
        "\nValidation report saved to "
        "validation_report.json"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    before_file = "heart.csv"

    after_file = "cleaned_heart.csv"

    try:

        # ----------------------------------------------------
        # Load datasets
        # ----------------------------------------------------

        before_df = load_dataset(
            before_file
        )

        after_df = load_dataset(
            after_file
        )

        print(
            "Datasets loaded successfully."
        )

        # ----------------------------------------------------
        # Calculate metrics
        # ----------------------------------------------------

        before_metrics = calculate_metrics(
            before_df
        )

        after_metrics = calculate_metrics(
            after_df
        )

        # ----------------------------------------------------
        # Calculate quality scores
        # ----------------------------------------------------

        before_score = calculate_quality_score(
            before_metrics
        )

        after_score = calculate_quality_score(
            after_metrics
        )

        # ----------------------------------------------------
        # Compare
        # ----------------------------------------------------

        comparison = compare_metrics(
            before_metrics,
            after_metrics
        )

        # ----------------------------------------------------
        # Determine result
        # ----------------------------------------------------

        result, reason = determine_result(
            before_score,
            after_score,
            before_metrics,
            after_metrics
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print_validation_report(
            before_metrics,
            after_metrics,
            before_score,
            after_score,
            comparison,
            result,
            reason
        )

        # ----------------------------------------------------
        # Save report
        # ----------------------------------------------------

        save_validation_report(
            before_metrics,
            after_metrics,
            before_score,
            after_score,
            comparison,
            result,
            reason
        )

    except Exception as e:

        print(
            f"\nValidation error: {e}"
        )