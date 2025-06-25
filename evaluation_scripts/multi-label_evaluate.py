from sklearn.metrics import average_precision_score
import pandas as pd
import argparse

def calculate_map(pred_df: pd.DataFrame, label_df: pd.DataFrame) -> float:
    """
    Calculate mean Average Precision (mAP) for multi-label classification.

    Parameters:
        pred_df (pd.DataFrame): DataFrame of soft predictions (0–1), with same case_ids and label columns as label_df.
        label_df (pd.DataFrame): DataFrame of ground truth labels (0 or 1).

    Returns:
        float: mean Average Precision across all classes.
    """
    # Ensure matching order of rows by case_id
    merged = pd.merge(label_df, pred_df, on="case_id", suffixes=("_true", "_pred"))
    
    label_cols = [col for col in label_df.columns[1:]]
    # Store AP for each class
    ap_scores = []
    for col in label_cols:
        y_true = merged[f"{col}_true"]
        y_pred = merged[f"{col}_pred"]
        try:
            ap = average_precision_score(y_true, y_pred)
        except ValueError:
            ap = float('nan')  # Handle class with no positive samples
        ap_scores.append(ap)
    
    # Compute mean while ignoring NaNs
    map_score = pd.Series(ap_scores).mean(skipna=True)
    return map_score

def main():
    parser = argparse.ArgumentParser(description="Compute mAP for multi-label classification.")
    parser.add_argument("--pred_file", type=str, required=True, help="Path to prediction CSV file (soft scores).")
    parser.add_argument("--label_file", type=str, required=True, help="Path to ground-truth label CSV file.")
    args = parser.parse_args()

    pred_df = pd.read_csv(args.pred_file)
    label_df = pd.read_csv(args.label_file)

    map_score = calculate_map(pred_df, label_df)
    print(f"Mean Average Precision (mAP): {map_score:.4f}")

if __name__ == "__main__":
    main()