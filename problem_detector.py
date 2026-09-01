import pandas as pd
from data_ingestion import load_data
from data_profiler import profile_data


# ============================================================
# 1. DETECT MISSING VALUES
# ============================================================

def detect_missing_values(df):
    problems = []

    for column in df.columns:

        missing_count = df[column].isnull().sum()

        if missing_count > 0:

            missing_percentage = (missing_count / len(df)) * 100

            problems.append({
                "type": "missing_values",
                "column": column,
                "count": int(missing_count),
                "percentage": round(missing_percentage, 2),
                "severity": (
                    "high" if missing_percentage > 30
                    else "medium" if missing_percentage > 5
                    else "low"
                )
            })

    return problems


# ============================================================
# 2. DETECT DUPLICATE ROWS
# ============================================================

def detect_duplicates(df):
    problems = []

    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:

        percentage = (duplicate_count / len(df)) * 100

        problems.append({
            "type": "duplicate_rows",
            "column": None,
            "count": int(duplicate_count),
            "percentage": round(percentage, 2),
            "severity": (
                "high" if percentage > 10
                else "medium" if percentage > 2
                else "low"
            )
        })

    return problems


# ============================================================
# 3. DETECT CONSTANT COLUMNS
# ============================================================

def detect_constant_columns(df):
    problems = []

    for column in df.columns:

        unique_count = df[column].nunique(dropna=False)

        if unique_count <= 1:

            problems.append({
                "type": "constant_column",
                "column": column,
                "count": unique_count,
                "severity": "medium"
            })

    return problems


# ============================================================
# 4. DETECT NUMERICAL OUTLIERS
# ============================================================

def detect_outliers(df):

    problems = []

    numerical_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numerical_columns:

        # Remove missing values
        values = df[column].dropna()

        if len(values) == 0:
            continue

        # Calculate Q1 and Q3
        Q1 = values.quantile(0.25)
        Q3 = values.quantile(0.75)

        # Interquartile range
        IQR = Q3 - Q1

        # Lower and upper limits
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Find outliers
        outliers = values[
            (values < lower_bound) |
            (values > upper_bound)
        ]

        if len(outliers) > 0:

            percentage = (len(outliers) / len(values)) * 100

            problems.append({
                "type": "outliers",
                "column": column,
                "count": int(len(outliers)),
                "percentage": round(percentage, 2),
                "lower_bound": round(lower_bound, 2),
                "upper_bound": round(upper_bound, 2),
                "severity": (
                    "high" if percentage > 10
                    else "medium" if percentage > 5
                    else "low"
                )
            })

    return problems


# ============================================================
# 5. DETECT INVALID NUMERICAL VALUES
# ============================================================

def detect_invalid_ranges(df):

    problems = []

    # Common range rules
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

        "oldpeak": (0, 10),

        "slope": (0, 2),

        "ca": (0, 4),

        "target": (0, 1)
    }

    for column, (minimum, maximum) in range_rules.items():

        if column not in df.columns:
            continue

        invalid = df[
            (df[column] < minimum) |
            (df[column] > maximum)
        ]

        if len(invalid) > 0:

            problems.append({
                "type": "invalid_range",
                "column": column,
                "count": int(len(invalid)),
                "expected_range": f"{minimum} - {maximum}",
                "severity": "high"
            })

    return problems


# ============================================================
# 6. DETECT HIGH-CARDINALITY CATEGORICAL COLUMNS
# ============================================================

def detect_high_cardinality(df):

    problems = []

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:

        unique_count = df[column].nunique()

        # If more than 50% of rows are unique
        if unique_count > len(df) * 0.5:

            problems.append({
                "type": "high_cardinality",
                "column": column,
                "unique_values": int(unique_count),
                "severity": "medium"
            })

    return problems


# ============================================================
# 7. DETECT EVERYTHING
# ============================================================

def detect_problems(df):

    problems = []

    problems.extend(
        detect_missing_values(df)
    )

    problems.extend(
        detect_duplicates(df)
    )

    problems.extend(
        detect_constant_columns(df)
    )

    problems.extend(
        detect_outliers(df)
    )

    problems.extend(
        detect_invalid_ranges(df)
    )

    problems.extend(
        detect_high_cardinality(df)
    )

    return problems


# ============================================================
# 8. PRINT PROBLEMS
# ============================================================

def print_problems(problems):

    print("\n")
    print("=" * 70)
    print("                    PROBLEM DETECTION")
    print("=" * 70)

    if not problems:

        print("\nNo data quality problems detected.")

        return

    print(
        f"\nTotal problems detected: {len(problems)}"
    )

    print("-" * 70)

    for i, problem in enumerate(problems, start=1):

        print(f"\nProblem {i}")
        print(f"  Type     : {problem['type']}")

        if problem.get("column") is not None:
            print(f"  Column   : {problem['column']}")

        if "count" in problem:
            print(f"  Count    : {problem['count']}")

        if "percentage" in problem:
            print(
                f"  Percentage: {problem['percentage']}%"
            )

        if "lower_bound" in problem:
            print(
                f"  Lower bound: {problem['lower_bound']}"
            )

        if "upper_bound" in problem:
            print(
                f"  Upper bound: {problem['upper_bound']}"
            )

        if "expected_range" in problem:
            print(
                f"  Expected range: "
                f"{problem['expected_range']}"
            )

        if "unique_values" in problem:
            print(
                f"  Unique values: "
                f"{problem['unique_values']}"
            )

        print(
            f"  Severity : {problem['severity']}"
        )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    file_path = "cleaned_heart.csv"

    try:

        # -------------------------------
        # Stage 1: Data Ingestion
        # -------------------------------

        df = load_data(file_path)

        print("Dataset loaded successfully.")

        # -------------------------------
        # Stage 2: Data Profiling
        # -------------------------------

        profile = profile_data(df)

        print(
            f"Dataset contains "
            f"{profile['rows']} rows and "
            f"{profile['columns']} columns."
        )

        # -------------------------------
        # Stage 3: Problem Detection
        # -------------------------------

        problems = detect_problems(df)

        print_problems(problems)

    except Exception as e:

        print(f"\nError: {e}")