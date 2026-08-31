import pandas as pd
import os


def load_data(file_path):
    """
    Load a dataset based on its file extension.

    Supported formats:
    - CSV
    - Excel (.xlsx, .xls)
    - JSON
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".csv":
        df = pd.read_csv(file_path)

    elif file_extension in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)

    elif file_extension == ".json":
        df = pd.read_json(file_path)

    else:
        raise ValueError(
            f"Unsupported file format: {file_extension}. "
            "Supported formats: CSV, Excel, JSON"
        )

    return df


# Test the ingestion system
if __name__ == "__main__":

    file_path = "heart.csv"

    try:
        df = load_data(file_path)

        print("Dataset loaded successfully!")
        print("-" * 40)

        print(f"Rows    : {df.shape[0]}")
        print(f"Columns : {df.shape[1]}")

        print("\nColumn names:")
        print(df.columns.tolist())

        print("\nFirst 5 rows:")
        print(df.head())

    except Exception as e:
        print(f"Error: {e}")