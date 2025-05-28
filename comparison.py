
import pandas as pd
import argparse

def calculate_accuracy(manual_csv, automated_csv):
    # Load the CSV files
    manual_df = pd.read_csv(manual_csv)
    automated_df = pd.read_csv(automated_csv)

    # Merge on identifying columns
    merged_df = pd.merge(
        manual_df,
        automated_df,
        on=["unixTimestampInMs", "x", "y", "z"],
        suffixes=('_manual', '_automated')
    )

    # Check label matches
    merged_df['label_match'] = merged_df['label_manual'] == merged_df['label_automated']

    # Overall accuracy
    total = len(merged_df)
    correct = merged_df['label_match'].sum()
    accuracy = (correct / total) * 100 if total > 0 else 0

    print(f"Total comparisons: {total}")
    print(f"Correct matches: {correct}")
    print(f"Overall Accuracy: {accuracy:.2f}%")

    # Per-label accuracy breakdown
    print("\nPer-label accuracy breakdown:")
    label_groups = merged_df.groupby('label_manual')
    for label, group in label_groups:
        total_label = len(group)
        correct_label = (group['label_manual'] == group['label_automated']).sum()
        label_accuracy = (correct_label / total_label) * 100 if total_label > 0 else 0
        print(f"  {label}: {label_accuracy:.2f}% ({correct_label}/{total_label})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare automated vs manual annotation accuracy with per-label breakdown.")
    parser.add_argument("manual_csv", help="Path to the manual annotation CSV file.")
    parser.add_argument("automated_csv", help="Path to the automated annotation CSV file.")
    args = parser.parse_args()
    calculate_accuracy(args.manual_csv, args.automated_csv)
