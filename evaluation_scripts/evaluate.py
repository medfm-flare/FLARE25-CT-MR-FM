#!/usr/bin/env python3
"""
Model Evaluation Script

Evaluates model predictions against ground truth for classification or regression tasks.

Usage:
    python evaluate.py --mode cls --pred predictions.csv --gt ground_truth.csv
    python evaluate.py --mode reg --pred predictions.csv --gt ground_truth.csv

CSV Format:
    Prediction file: columns 'name' and 'prediction'
    Ground truth file: columns 'name' and 'gt'
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, mean_absolute_error
import sys
import os


def load_and_validate_data(pred_file, gt_file):
    """Load and validate CSV files."""
    # Check if files exist
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
    
    # Load CSV files
    try:
        pred_df = pd.read_csv(pred_file)
        gt_df = pd.read_csv(gt_file)
    except Exception as e:
        raise ValueError(f"Error loading CSV files: {e}")
    
    # Validate required columns
    if 'name' not in pred_df.columns or 'prediction' not in pred_df.columns:
        raise ValueError("Prediction file must contain 'name' and 'prediction' columns")
    if 'name' not in gt_df.columns or 'gt' not in gt_df.columns:
        raise ValueError("Ground truth file must contain 'name' and 'gt' columns")
    
    # Merge on name column
    merged_df = pd.merge(pred_df, gt_df, on='name', how='inner')
    
    if len(merged_df) == 0:
        raise ValueError("No matching names found between prediction and ground truth files")
    
    # Check for missing values
    if merged_df['prediction'].isna().any() or merged_df['gt'].isna().any():
        print("Warning: Found missing values. Removing rows with NaN values.")
        merged_df = merged_df.dropna(subset=['prediction', 'gt'])
    
    if len(merged_df) == 0:
        raise ValueError("No valid data remaining after removing missing values")
    
    print(f"Successfully loaded {len(merged_df)} samples")
    
    return merged_df['prediction'].values, merged_df['gt'].values


def evaluate_classification(predictions, ground_truth):
    """Evaluate classification metrics."""
    try:
        # Convert to appropriate data types
        y_pred = np.array(predictions)
        y_true = np.array(ground_truth)
        
        # For balanced accuracy, we need discrete labels
        # If predictions are probabilities, convert to binary labels
        if np.all((y_pred >= 0) & (y_pred <= 1)) and not np.all(np.isin(y_pred, [0, 1])):
            # Predictions appear to be probabilities
            y_pred_labels = (y_pred > 0.5).astype(int)
            print("Detected probability predictions, converting to binary labels using threshold 0.5")
        else:
            y_pred_labels = y_pred.astype(int)
        
        y_true = y_true.astype(int)
        
        # Calculate Balanced Accuracy
        balanced_acc = balanced_accuracy_score(y_true, y_pred_labels)
        
        # Calculate AUROC
        # For AUROC, use original predictions if they're probabilities
        if np.all((predictions >= 0) & (predictions <= 1)) and not np.all(np.isin(predictions, [0, 1])):
            auroc = roc_auc_score(y_true, predictions)
        else:
            # If predictions are already discrete, use them directly
            unique_vals = len(np.unique(predictions))
            if unique_vals > 2:
                print("Warning: More than 2 unique prediction values detected for binary classification")
            auroc = roc_auc_score(y_true, predictions)
        
        return balanced_acc, auroc
        
    except Exception as e:
        raise ValueError(f"Error in classification evaluation: {e}")


def evaluate_regression(predictions, ground_truth):
    """Evaluate regression metrics."""
    try:
        y_pred = np.array(predictions, dtype=float)
        y_true = np.array(ground_truth, dtype=float)
        
        mae = mean_absolute_error(y_true, y_pred)
        return mae
        
    except Exception as e:
        raise ValueError(f"Error in regression evaluation: {e}")




def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model predictions against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --mode cls --pred predictions.csv --gt ground_truth.csv
  python evaluate.py --mode reg --pred predictions.csv --gt ground_truth.csv

CSV File Format:
  Prediction file: Must contain columns 'name' and 'prediction'
  Ground truth file: Must contain columns 'name' and 'gt'

        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['cls', 'reg'],
        required=True,
        help='Evaluation mode: cls for classification, reg for regression'
    )
    
    parser.add_argument(
        '--pred',
        required=True,
        help='Path to prediction CSV file (columns: name, prediction)'
    )
    
    parser.add_argument(
        '--gt',
        required=True, 
        help='Path to ground truth CSV file (columns: name, gt)'
    )
    
    args = parser.parse_args()
    
    try:
        # Load and validate data
        predictions, ground_truth = load_and_validate_data(args.pred, args.gt)
        
        print(f"Evaluation mode: {args.mode.upper()}")
        print(f"Prediction file: {args.pred}")
        print(f"Ground truth file: {args.gt}")
        print("-" * 50)
        
        # Generate output filename
        output_file = f"{args.mode}_result.csv"
        
        if args.mode == 'cls':
            balanced_acc, auroc = evaluate_classification(predictions, ground_truth)
            print(f"Balanced Accuracy: {balanced_acc:.4f}")
            print(f"AUROC: {auroc:.4f}")
            
            
        elif args.mode == 'reg':
            mae = evaluate_regression(predictions, ground_truth)
            print(f"MAE: {mae:.4f}")
            
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()