# Model Evaluation Scripts

A collection of Python scripts for evaluating machine learning model predictions across different tasks: binary classification, multi-class classification, multi-label classification, regression, and survival analysis.

## Overview

This repository contains specialized evaluation scripts for different machine learning tasks:

- **`evaluate.py`** - Binary classification and regression evaluation
- **`multi-class_evaluation.py`** - Multi-class classification evaluation  
- **`multi-label_evaluate.py`** - Multi-label classification evaluation
- **`survival_evaluation.py`** - Survival analysis evaluation

## Requirements

```bash
pip install pandas numpy scikit-learn
```

## File Format Requirements

All scripts expect CSV files with specific column structures:

### Prediction Files
- **Binary/Regression**: `name`, `prediction`
- **Multi-class**: `name`, `class_0`, `class_1`, ..., `class_n`
- **Multi-label**: `case_id`, `label1`, `label2`, ..., `labeln` (soft scores 0-1)
- **Survival**: `name`, `prediction`

### Ground Truth Files
- **Binary/Regression**: `name`, `gt`
- **Multi-class**: `name`, `gt` (integer class labels)
- **Multi-label**: `case_id`, `label1`, `label2`, ..., `labeln` (binary 0/1)
- **Survival**: `name`, `gt`

## Usage Examples

### Binary Classification & Regression

```bash
# Binary classification evaluation
python evaluate.py --mode cls --pred predictions.csv --gt ground_truth.csv

# Regression evaluation  
python evaluate.py --mode reg --pred predictions.csv --gt ground_truth.csv
```

**Metrics:**
- **Classification**: Balanced Accuracy, AUROC
- **Regression**: Mean Absolute Error (MAE)


### Multi-Class Classification

```bash
python multi-class_evaluation.py --pred_file predictions.csv --gt_file ground_truth.csv
```

**Metrics:**
- Average Precision (AP) per class
- Mean Average Precision (mAP)

**Example prediction file:**
```csv
name,class_0,class_1,class_2
sample1,0.7,0.2,0.1
sample2,0.1,0.8,0.1
```

**Example ground truth file:**
```csv
name,gt
sample1,0
sample2,1
```

### Multi-Label Classification

```bash
python multi-label_evaluate.py --pred_file predictions.csv --label_file labels.csv
```

**Metrics:**
- Mean Average Precision (mAP) across all labels

**Example prediction file:**
```csv
case_id,disease_a,disease_b,disease_c
patient1,0.8,0.3,0.1
patient2,0.2,0.9,0.7
```

**Example label file:**
```csv
case_id,disease_a,disease_b,disease_c
patient1,1,0,0
patient2,0,1,1
```

### Survival Analysis

```bash
python survival_evaluation.py --pred_file predictions.csv --gt_file ground_truth.csv
```

**Metrics:**
- Concordance Index (C-index)




