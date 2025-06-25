import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize

def calculate_multiclass_map(pred_file, gt_file):
    # Load files
    pred_df = pd.read_csv(pred_file)
    gt_df = pd.read_csv(gt_file)

    # Merge on "name"
    df = pd.merge(gt_df, pred_df, on="name")

    # Extract y_true and y_scores
    y_true = df["gt"].values
    class_cols = [col for col in pred_df.columns if col != "name"]
    y_scores = df[class_cols].values
    n_classes = len(class_cols)

    # Binarize ground truth labels for OVR mAP
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
    
    # Compute AP per class
    ap_list = []
    for i in range(n_classes):
        try:
            ap = average_precision_score(y_true_bin[:, i], y_scores[:, i])
        except ValueError:
            ap = float("nan")
        ap_list.append(ap)

    # Compute mean AP (ignoring NaNs)
    mean_ap = np.nanmean(ap_list)

    # Print results
    for i, ap in enumerate(ap_list):
        print(f"Class {i}: AP = {ap:.4f}")
    print(f"\nMean Average Precision (mAP): {mean_ap:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate mAP for multiclass classification predictions.")
    parser.add_argument("--pred_file", type=str, required=True, help="CSV with columns: name, class_0, class_1, ...")
    parser.add_argument("--gt_file", type=str, required=True, help="CSV with columns: name, label")
    args = parser.parse_args()

    calculate_multiclass_map(args.pred_file, args.gt_file)
