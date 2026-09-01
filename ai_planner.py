import os
import json

from google import genai

from data_ingestion import load_data
from data_profiler import profile_data
from problem_detector import detect_problems


# ============================================================
# GEMINI CLIENT
# ============================================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY environment variable is not set."
    )

client = genai.Client(api_key=api_key)


# ============================================================
# CREATE PROMPT
# ============================================================

def create_prompt(profile, problems):

    prompt = f"""
You are an expert data quality and data cleaning agent.

Your job is to analyze a dataset profile and the problems
detected in the dataset.

You must recommend appropriate cleaning operations.

IMPORTANT RULES:

1. Do not modify the dataset.
2. Do not generate Python code.
3. Only recommend cleaning actions.
4. Do not blindly delete outliers.
5. Consider the meaning of each column.
6. If you are uncertain about a problem, recommend flagging
   it instead of automatically changing the value.
7. Return ONLY valid JSON.
8. Use only the allowed operations listed below.

ALLOWED OPERATIONS:

- remove_duplicates
- fill_missing_mean
- fill_missing_median
- fill_missing_mode
- drop_missing_rows
- normalize_text
- convert_data_type
- flag_outliers
- remove_outliers
- flag_invalid_values
- remove_invalid_values
- standardize_categories
- no_action


DATASET PROFILE:

{json.dumps(profile, indent=2, default=str)}


DETECTED PROBLEMS:

{json.dumps(problems, indent=2, default=str)}


Return the result using EXACTLY this JSON structure:

{{
    "summary": "Short summary of the dataset quality",

    "overall_risk": "low/medium/high",

    "cleaning_plan": [
        {{
            "problem_type": "type of detected problem",

            "column": "column name or null",

            "operation": "one allowed operation",

            "reason": "why this operation is appropriate",

            "confidence": 0.0
        }}
    ]
}}
"""

    return prompt


# ============================================================
# SEND PROFILE TO GEMINI
# ============================================================

def generate_cleaning_plan(profile, problems):

    prompt = create_prompt(
        profile,
        problems
    )

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    return interaction.output_text


# ============================================================
# DISPLAY CLEANING PLAN
# ============================================================

def print_cleaning_plan(result):

    try:
        # Remove Markdown code fences if Gemini adds them
        result = result.strip()

        if result.startswith("```json"):
            result = result[7:]

        elif result.startswith("```"):
            result = result[3:]

        if result.endswith("```"):
            result = result[:-3]

        result = result.strip()

        plan = json.loads(result)

        print("\n")
        print("=" * 70)
        print("                     AI CLEANING PLAN")
        print("=" * 70)

        print("\nSummary:")
        print(plan.get("summary", "No summary"))

        print("\nOverall Risk:")
        print(plan.get("overall_risk", "Unknown"))

        print("\nRecommended Actions:")
        print("-" * 70)

        for i, action in enumerate(
            plan.get("cleaning_plan", []),
            start=1
        ):

            print(f"\nAction {i}")
            print(f"  Problem   : {action.get('problem_type')}")
            print(f"  Column    : {action.get('column')}")
            print(f"  Operation : {action.get('operation')}")
            print(f"  Reason    : {action.get('reason')}")
            print(f"  Confidence: {action.get('confidence')}")

    except json.JSONDecodeError:
        print("\nGemini returned invalid JSON:")
        print(result)

def save_cleaning_plan(result, filename="cleaning_plan.json"):

    result = result.strip()

    if result.startswith("```json"):
        result = result[7:]
    elif result.startswith("```"):
        result = result[3:]

    if result.endswith("```"):
        result = result[:-3]

    result = result.strip()

    plan = json.loads(result)

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(plan, file, indent=4)

    print(f"\nCleaning plan saved to {filename}")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    file_path = "heart.csv"

    try:

        # ----------------------------------------------------
        # Stage 1
        # ----------------------------------------------------

        df = load_data(file_path)

        print(
            "Stage 1 completed: Dataset loaded."
        )

        # ----------------------------------------------------
        # Stage 2
        # ----------------------------------------------------

        profile = profile_data(df)

        print(
            "Stage 2 completed: Dataset profiled."
        )

        # ----------------------------------------------------
        # Stage 3
        # ----------------------------------------------------

        problems = detect_problems(df)

        print(
            f"Stage 3 completed: "
            f"{len(problems)} problems detected."
        )

        # ----------------------------------------------------
        # Stage 4
        # ----------------------------------------------------

        print(
            "\nSending profile to Gemini..."
        )

        result = generate_cleaning_plan(
            profile,
            problems
        )

        print(
            "Gemini analysis completed."
        )

        # Display result
        print_cleaning_plan(result)
        
        save_cleaning_plan(result)
    except Exception as e:

        print(
            f"\nError: {e}"
        )