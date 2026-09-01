import sys
import os
import json
import time

from data_ingestion import load_data
from data_profiler import profile_data
from problem_detector import detect_problems

from ai_planner import generate_cleaning_plan

from cleaning_engine import (
    load_cleaning_plan,
    clean_dataset,
    save_cleaning_report
)

from validation import (
    calculate_metrics,
    calculate_quality_score,
    compare_metrics,
    determine_result,
    save_validation_report
)


# ============================================================
# DISPLAY HELPERS
# ============================================================

def print_header():

    print("\n")
    print("=" * 75)
    print("                     DATA CLEANING AGENT")
    print("=" * 75)


def print_stage(number, total, message):

    print(
        f"\n[{number}/{total}] {message}",
        end=" "
    )


def print_success():

    print("✓")


# ============================================================
# CLEAN GEMINI RESPONSE
# ============================================================

def clean_gemini_response(result):

    result = result.strip()

    if result.startswith("```json"):
        result = result[7:]

    elif result.startswith("```"):
        result = result[3:]

    if result.endswith("```"):
        result = result[:-3]

    return result.strip()


# ============================================================
# SAVE GEMINI PLAN
# ============================================================

def save_ai_plan(result):

    result = clean_gemini_response(result)

    plan = json.loads(result)

    os.makedirs("reports", exist_ok=True)

    with open(
        "reports/cleaning_plan.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            plan,
            file,
            indent=4
        )

    return plan


# ============================================================
# PRINT DETECTED PROBLEMS
# ============================================================

def print_problems(problems):

    print(
        f"\n\nProblems detected: {len(problems)}"
    )

    if len(problems) == 0:

        print("No problems detected.")

        return

    for i, problem in enumerate(
        problems,
        start=1
    ):

        print(
            f"  {i}. {problem}"
        )


# ============================================================
# PRINT AI PLAN
# ============================================================

def print_ai_plan(plan):

    actions = plan.get(
        "cleaning_plan",
        []
    )

    print("\n")
    print("=" * 75)
    print("                    AI CLEANING PLAN")
    print("=" * 75)

    print(
        f"\nSummary:\n{plan.get('summary', 'N/A')}"
    )

    print(
        f"\nOverall risk: "
        f"{plan.get('overall_risk', 'unknown')}"
    )

    print("\nRecommended actions:")

    for i, action in enumerate(
        actions,
        start=1
    ):

        print(
            f"\n  {i}. "
            f"{action.get('operation')}"
        )

        print(
            f"     Column: "
            f"{action.get('column')}"
        )

        print(
            f"     Reason: "
            f"{action.get('reason')}"
        )

        print(
            f"     Confidence: "
            f"{action.get('confidence')}"
        )


# ============================================================
# RUN AGENT
# ============================================================

def run_agent(input_file):

    total_stages = 6

    print_header()

    print(
        f"\nInput dataset: {input_file}"
    )

    start_time = time.time()

    # ========================================================
    # STAGE 1
    # ========================================================

    print_stage(
        1,
        total_stages,
        "Loading dataset..."
    )

    df = load_data(input_file)

    print_success()

    print(
        f"       Shape: "
        f"{df.shape[0]} rows × "
        f"{df.shape[1]} columns"
    )


    # ========================================================
    # STAGE 2
    # ========================================================

    print_stage(
        2,
        total_stages,
        "Profiling dataset..."
    )

    profile = profile_data(df)

    print_success()


    # ========================================================
    # STAGE 3
    # ========================================================

    print_stage(
        3,
        total_stages,
        "Detecting problems..."
    )

    problems = detect_problems(df)

    print_success()

    print_problems(problems)


    # ========================================================
    # STAGE 4
    # ========================================================

    print_stage(
        4,
        total_stages,
        "Consulting Gemini AI..."
    )

    if len(problems) == 0:

        print(
            "\n       No problems detected."
        )

        plan = {
            "summary": "No data quality problems detected.",
            "overall_risk": "low",
            "cleaning_plan": []
        }

        os.makedirs("reports", exist_ok=True)
        with open(
            "reports/cleaning_plan.json",
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                plan,
                file,
                indent=4
            )

    else:

        result = generate_cleaning_plan(
            profile,
            problems
        )

        plan = save_ai_plan(result)

    print_success()

    print_ai_plan(plan)


    # ========================================================
    # STAGE 5
    # ========================================================

    print_stage(
        5,
        total_stages,
        "Applying cleaning plan..."
    )

    cleaned_df, cleaning_results = clean_dataset(
        df.copy(),
        plan
    )

    cleaned_file = "cleaned_" + os.path.basename(
        input_file
    )

    cleaned_df.to_csv(
        cleaned_file,
        index=False
    )

    save_cleaning_report(
        cleaning_results
    )

    print_success()

    print(
        f"       Output: {cleaned_file}"
    )


    # ========================================================
    # STAGE 6
    # ========================================================

    print_stage(
        6,
        total_stages,
        "Validating results..."
    )

    before_metrics = calculate_metrics(
        df
    )

    after_metrics = calculate_metrics(
        cleaned_df
    )

    before_score = calculate_quality_score(
        before_metrics
    )

    after_score = calculate_quality_score(
        after_metrics
    )

    comparison = compare_metrics(
        before_metrics,
        after_metrics
    )

    result, reason = determine_result(
        before_score,
        after_score,
        before_metrics,
        after_metrics
    )

    save_validation_report(
        before_metrics,
        after_metrics,
        before_score,
        after_score,
        comparison,
        result,
        reason
    )

    print_success()


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    elapsed_time = time.time() - start_time

    print("\n")
    print("=" * 75)
    print("                         FINAL RESULT")
    print("=" * 75)

    print(
        f"\nDataset:"
        f" {input_file}"
    )

    print(
        f"Original size:"
        f" {df.shape[0]} rows × "
        f"{df.shape[1]} columns"
    )

    print(
        f"Final size:"
        f" {cleaned_df.shape[0]} rows × "
        f"{cleaned_df.shape[1]} columns"
    )

    print(
        f"\nProblems detected:"
        f" {len(problems)}"
    )

    print(
        f"Cleaning actions:"
        f" {len(plan.get('cleaning_plan', []))}"
    )

    print(
        f"\nQuality score:"
        f" {before_score} → {after_score}"
    )

    print(
        f"Score change:"
        f" {after_score - before_score:+.2f}"
    )

    print(
        f"\nDecision:"
        f" {result}"
    )

    print(
        f"Reason:"
        f" {reason}"
    )

    print(
        f"\nExecution time:"
        f" {elapsed_time:.2f} seconds"
    )

    print("\nGenerated files:")

    print(
        f"  ✓ {cleaned_file}"
    )

    print(
        "  ✓ cleaning_plan.json"
    )

    print(
        "  ✓ cleaning_report.json"
    )

    print(
        "  ✓ validation_report.json"
    )

    print("\n")
    print("=" * 75)
    print("                 AGENT EXECUTION COMPLETE")
    print("=" * 75)


# ============================================================
# COMMAND LINE ENTRY POINT
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            "\nUsage:"
        )

        print(
            "  python agent.py <dataset.csv>"
        )

        print(
            "\nExample:"
        )

        print(
            "  python agent.py heart.csv"
        )

        sys.exit(1)


    input_file = sys.argv[1]


    if not os.path.exists(input_file):

        print(
            f"\nError: Dataset not found: "
            f"{input_file}"
        )

        sys.exit(1)


    try:

        run_agent(input_file)

    except Exception as e:

        print("\n")
        print("=" * 75)
        print("                       AGENT ERROR")
        print("=" * 75)

        print(
            f"\n{type(e).__name__}: {e}"
        )

        print(
            "\nThe pipeline was stopped."
        )

        sys.exit(1)