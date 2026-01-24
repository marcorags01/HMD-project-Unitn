
"""
eval_survey.py — Minimal post-scenario questionnaire for Meal Kit Composer evaluation.

Usage:
  python eval_survey.py --pid P01 --scenario 1
  python eval_survey.py --pid P01 --scenario 2

Output:
  Appends one row per run to: eval/ratings.csv   (default outdir: "eval")
"""

import argparse
import csv
import os
from datetime import datetime


SCALE_TEXT = (
    "Scale: 1=Completely Disagree, 2=Disagree, 3=Neutral, 4=Agree, 5=Completely Agree"
)

QUESTIONS = [
    "Q1. The system followed my constraints (servings, time, calories, avoid items).",
    "Q2. The system asked for missing information or clarifications when something was unclear.",
    "Q3. The system's responses were clear and coherent (no contradictions or irrelevant details).",
    "Q4. It was easy to understand what to do next (e.g., select menu option, refine, confirm).",
    "Q5. Refinements worked as I expected (swap day / update avoid items / repair).",
    "Q6. I trust the final output enough to use it for planning meals/shopping.",
    "Q7. I would consider using the system again.",
]


def ask_likert(prompt: str) -> int:
    while True:
        raw = input(f"{prompt}\nYour score (1-5): ").strip()
        if raw in {"1", "2", "3", "4", "5"}:
            return int(raw)
        print("Invalid input. Please enter a number from 1 to 5.\n")


def ask_text(prompt: str) -> str:
    return input(f"{prompt}\nYour answer (optional): ").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-scenario questionnaire (appends to CSV).")
    ap.add_argument("--pid", required=True, help="Participant ID (e.g., P01)")
    ap.add_argument("--scenario", required=True, help="Scenario label/number (e.g., 1, 2, A, B)")
    ap.add_argument("--outdir", default="eval", help="Output directory (default: eval)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    outpath = os.path.join(args.outdir, "ratings.csv")

    print("\nPost-task questionnaire")
    print(SCALE_TEXT)
    print("-" * 72)

    scores = []
    for q in QUESTIONS:
        scores.append(ask_likert(q))
        print()  # spacer

    biggest_problem = ask_text("Open 1. What was the biggest problem you encountered?")
    print()
    one_improvement = ask_text("Open 2. What is one improvement you would most like?")
    print()

    # Use local time with timezone offset if available
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

    row = {
        "timestamp": timestamp,
        "participant_id": args.pid,
        "scenario": args.scenario,
    }
    for i, s in enumerate(scores, start=1):
        row[f"Q{i}"] = s
    row["biggest_problem"] = biggest_problem
    row["one_improvement"] = one_improvement

    fieldnames = (
        ["timestamp", "participant_id", "scenario"]
        + [f"Q{i}" for i in range(1, 8)]
        + ["biggest_problem", "one_improvement"]
    )

    write_header = not os.path.exists(outpath) or os.path.getsize(outpath) == 0
    with open(outpath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"Saved ratings to: {outpath}")
    print("Thank you.\n")


if __name__ == "__main__":
    main()
