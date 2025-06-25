import pandas as pd
import numpy as np
from typing import Tuple


def calculate_metric(pred_df: pd.DataFrame, label_df: pd.DataFrame) -> float:
    """
    Optimized version using vectorized operations for better performance on large datasets.
    
    Args:
        pred_df (pd.DataFrame): DataFrame with columns ['name', 'prediction']
        label_df (pd.DataFrame): DataFrame with columns ['name', 'gt']
    
    Returns:
        float: C-index value between 0 and 1
    """
    # Merge dataframes on 'name' to align predictions with ground truth
    merged_df = pd.merge(pred_df, label_df, on='name', how='inner')
    
    if len(merged_df) < 2:
        raise ValueError("Need at least 2 samples to calculate C-index")
    
    predictions = merged_df['prediction'].values
    ground_truth = merged_df['gt'].values
    
    # Create matrices for pairwise comparisons
    gt_diff = ground_truth[:, np.newaxis] - ground_truth[np.newaxis, :]
    pred_diff = predictions[:, np.newaxis] - predictions[np.newaxis, :]
    
    # Count concordant pairs (same sign for gt_diff and pred_diff)
    concordant = (gt_diff * pred_diff > 0)
    
    # Count valid pairs (exclude tied ground truth values)
    valid_pairs = (gt_diff != 0)
    
    # Get upper triangular part to avoid double counting
    upper_tri = np.triu(np.ones_like(gt_diff, dtype=bool), k=1)
    
    concordant_count = np.sum(concordant & valid_pairs & upper_tri)
    total_count = np.sum(valid_pairs & upper_tri)
    
    if total_count == 0:
        return 0.5
    
    return concordant_count / total_count

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Calculate C-index for survival prediction')
    parser.add_argument('--pred_file', type=str, required=True,
                       help='Path to CSV file with predictions (columns: name, prediction)')
    parser.add_argument('--gt_file', type=str, required=True,
                       help='Path to CSV file with ground truth labels (columns: name, gt)')
    
    args = parser.parse_args()
    
    try:
        # Load data from CSV files
        pred_df = pd.read_csv(args.pred_file)
        label_df = pd.read_csv(args.gt_file)
        
        # Validate required columns
        if 'name' not in pred_df.columns or 'prediction' not in pred_df.columns:
            raise ValueError("pred_file must contain 'name' and 'prediction' columns")
        
        if 'name' not in label_df.columns or 'gt' not in label_df.columns:
            raise ValueError("label_file must contain 'name' and 'gt' columns")
        
        # Calculate C-index

        c_index = calculate_metric(pred_df, label_df)
        print(f"C-index: {c_index:.4f}")
            
        # Additional statistics
        merged_df = pd.merge(pred_df, label_df, on='name', how='inner')
        print(f"Number of samples: {len(merged_df)}")
        print(f"Samples in pred_file: {len(pred_df)}")
        print(f"Samples in label_file: {len(label_df)}")
        if len(merged_df) < len(pred_df) or len(merged_df) < len(label_df):
            print("Warning: Some samples were not matched between prediction and label files")
        
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)