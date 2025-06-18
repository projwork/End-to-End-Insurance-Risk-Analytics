# Data Version Control (DVC) Setup Documentation

## Overview

This document outlines the Data Version Control (DVC) implementation for the Insurance Risk Analytics project, ensuring reproducible and auditable data management for regulatory compliance.

## Project Structure

```
End-to-End-Insurance-Risk-Analytics/
├── .dvc/                           # DVC configuration directory
├── .dvcignore                      # DVC ignore patterns
├── data/
│   ├── MachineLearningRating_v3.txt.dvc        # Original dataset (505MB)
│   ├── MachineLearningRating_sample_v1.txt.dvc # Sample v1 (50k rows)
│   ├── MachineLearningRating_sample_v2.txt.dvc # Sample v2 (75k rows + features)
│   └── MachineLearningRating_sample_v3.txt.dvc # Sample v3 (100k rows + advanced features)
├── C:\Kifiya\Week3\dvc-storage/    # Local DVC remote storage
└── requirements.txt                # Updated with dvc>=3.60.0
```

## DVC Setup Summary

### 1. Installation & Initialization

- **DVC Version**: 3.60.1
- **Installation**: `pip install dvc` (added to requirements.txt)
- **Initialization**: `dvc init` (created .dvc/ directory and configuration)

### 2. Remote Storage Configuration

- **Remote Name**: `localstorage` (default)
- **Storage Path**: `C:\Kifiya\Week3\dvc-storage`
- **Command Used**: `dvc remote add -d localstorage C:\Kifiya\Week3\dvc-storage`

### 3. Data Versioning

#### Original Dataset

- **File**: `MachineLearningRating_v3.txt` (505MB)
- **DVC Hash**: `f6b7009b68ae21372b7deca9307fbb23`
- **Records**: ~1M insurance records

#### Sample Versions Created

1. **v1**: 50,000 rows (baseline sample)
2. **v2**: 75,000 rows + basic features (LossRatio, HighRisk)
3. **v3**: 100,000 rows + advanced features (PremiumBucket, HasMissingData)

### 4. Git Integration

- Updated `.gitignore` to track `.dvc` files but ignore actual data files
- Pattern: `/data/*` with exception `!/data/*.dvc`
- All DVC configuration files committed to Git

## Key Benefits for Insurance/Finance Industry

### Regulatory Compliance

- **Auditability**: Every data version is tracked with cryptographic hashes
- **Reproducibility**: Any analysis can be reproduced exactly using DVC checkout
- **Lineage**: Complete data lineage tracking for regulatory requirements

### Data Governance

- **Version Control**: Multiple data versions without storing duplicates
- **Storage Efficiency**: Content-addressable storage using MD5 hashes
- **Collaboration**: Team members can sync data versions seamlessly

## Common DVC Commands

### Basic Operations

```bash
# Check DVC status
dvc status

# List tracked files
dvc list . data

# Add new data file
dvc add data/new_dataset.csv

# Push data to remote
dvc push

# Pull data from remote
dvc pull

# Switch to specific data version
dvc checkout <commit-hash>
```

### Data Pipeline Commands

```bash
# Check data/pipeline status
dvc status

# Show data dependencies
dvc dag

# Get file info
dvc get . data/MachineLearningRating_v3.txt
```

### Remote Management

```bash
# List remotes
dvc remote list

# Modify remote
dvc remote modify localstorage url /new/path/to/storage

# Add cloud remote (example)
dvc remote add -d cloud s3://mybucket/dvcstore
```

## File Descriptions

### .dvc Files (tracked in Git)

- `MachineLearningRating_v3.txt.dvc`: Metadata for original 505MB dataset
- `MachineLearningRating_sample_v*.txt.dvc`: Metadata for sample versions
- Content: MD5 hash, file size, path information

### Data Files (tracked by DVC)

- Actual data files stored in DVC remote
- Referenced by .dvc metadata files
- Automatically synced using `dvc push/pull`

## Workflow for Data Scientists

### 1. Initial Setup

```bash
# Clone repository
git clone <repo-url>
cd End-to-End-Insurance-Risk-Analytics

# Setup virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Pull latest data
dvc pull
```

### 2. Working with Data

```bash
# Make changes to data/analysis
# Add new data versions
dvc add data/new_processed_data.csv

# Commit changes
git add data/new_processed_data.csv.dvc
git commit -m "feat: Add processed customer data v2.1"

# Push data to remote
dvc push
```

### 3. Reproducing Analysis

```bash
# Switch to specific version
git checkout <analysis-commit>
dvc checkout

# Run analysis with exact data state
python analysis_script.py
```

## Regulatory Compliance Features

### Audit Trail

- Every data modification tracked in Git history
- Cryptographic hashes ensure data integrity
- Complete lineage from raw data to analysis results

### Reproducibility

- Exact data state reconstruction using Git + DVC
- Environment reproducibility via requirements.txt
- Analysis reproducibility via modular scripts

### Security

- Data stored separately from code repository
- Access control via remote storage permissions
- No sensitive data in Git history

## Troubleshooting

### Common Issues

1. **DVC files ignored by Git**: Check .gitignore patterns
2. **Remote storage access**: Verify path and permissions
3. **Data sync issues**: Use `dvc status` and `dvc doctor`

### Recovery Commands

```bash
# Reset DVC state
dvc checkout --force

# Repair corrupted cache
dvc cache dir
dvc gc --force

# Check DVC configuration
dvc config --list
```

## Next Steps

1. **Pipeline Integration**: Set up DVC pipelines for automated data processing
2. **Cloud Storage**: Migrate to cloud-based remote (AWS S3, Azure Blob)
3. **CI/CD Integration**: Automate data validation and testing
4. **Metrics Tracking**: Use DVC metrics for model performance tracking

## Contact & Support

- DVC Documentation: https://dvc.org/doc
- Project Issue Tracker: [Repository Issues]
- Internal Support: [Team Contact Information]

---

_Last Updated: 2025-06-18_
_DVC Version: 3.60.1_
_Project: End-to-End Insurance Risk Analytics_
