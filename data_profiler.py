import pandas as pd
from data_ingestion import load_data


def profile_data(df):
    """
    Generate a profile of the dataset.

    The profiler checks:
    - Number of rows and columns
    - Column names
    - Data types
    - Missing values
    - Unique values
    - Basic statistics
    """

    profile = {}

    # Basic dataset information
    profile["rows"] = df.shape[0]
    profile["columns"] = df.shape[1]

    # Column information
    columns = {}

    for column in df.columns:

        col = df[column]

        column_info = {
            "data_type": str(col.dtype),
            "missing_values": int(col.isnull().sum()),
            "missing_percentage": round(
                (col.isnull().sum() / len(df)) * 100, 2
            ),
            "unique_values": int(col.nunique()),
        }

        # Numerical columns
        if pd.api.types.is_numeric_dtype(col):

            column_info["min"] = col.min()
            column_info["max"] = col.max()
            column_info["mean"] = round(col.mean(), 2)
            column_info["median"] = col.median()
            column_info["std"] = round(col.std(), 2)

        # Categorical / text columns
        else:

            column_info["most_frequent"] = (
                col.mode().iloc[0]
                if not col.mode().empty
                else None
            )

        columns[column] = column_info

    profile["columns_info"] = columns

    return profile


def print_profile(profile):
    """
    Display the dataset profile in a readable format.
    """

    print("\n" + "=" * 60)
    print("              DATASET PROFILE")
    print("=" * 60)

    print(f"\nNumber of rows    : {profile['rows']}")
    print(f"Number of columns : {profile['columns']}")

    print("\n" + "-" * 60)
    print("COLUMN INFORMATION")
    print("-" * 60)

    for column, info in profile["columns_info"].items():

        print(f"\nColumn: {column}")
        print(f"  Data type          : {info['data_type']}")
        print(f"  Missing values     : {info['missing_values']}")
        print(f"  Missing percentage : {info['missing_percentage']}%")
        print(f"  Unique values      : {info['unique_values']}")

        if "min" in info:
            print(f"  Minimum            : {info['min']}")
            print(f"  Maximum            : {info['max']}")
            print(f"  Mean               : {info['mean']}")
            print(f"  Median             : {info['median']}")
            print(f"  Standard deviation: {info['std']}")

        else:
            print(f"  Most frequent      : {info['most_frequent']}")


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

if __name__ == "__main__":

    file_path = "heart.csv"

    try:

        # Stage 1: Load dataset
        df = load_data(file_path)

        print("Dataset loaded successfully!")

        # Stage 2: Profile dataset
        profile = profile_data(df)

        # Display profile
        print_profile(profile)

    except Exception as e:

        print(f"Error: {e}")