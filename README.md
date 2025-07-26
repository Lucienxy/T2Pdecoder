# Converting Transcriptomics to Proteomics via T2Pdecoder
T2Pdecoder is an integrative multi-omics deep learning model designed to predict relative protein abundances across a wide range of proteins

# workflow
![image](https://github.com/Lucienxy/T2Pdecoder/blob/main/img/T2Pdecoder_workflow.png)

# Requirements

[![python >3.7.9](https://img.shields.io/badge/python-3.7.9+-blue)](https://www.python.org/) [![pytorch >2.0.1](https://img.shields.io/badge/pytorch-2.0.1+-red)](https://pytorch.org/) [![numpy-1.24.3](https://img.shields.io/badge/numpy-1.24.3-yellow)](https://numpy.org/) [![pandas-2.2.2](https://img.shields.io/badge/pandas-2.2.2-green)](https://pandas.pydata.org/) [![scikit-learn-1.3.0](https://img.shields.io/badge/scikit--learn-1.3.0-orange)](https://scikit-learn.org/) [![scipy-1.11.1](https://img.shields.io/badge/scipy-1.11.1-purple)](https://scipy.org/)

## Installation

### Using Conda (Recommended)

```bash
# Create a new conda environment
conda create -n t2pdecoder python=3.7.9
conda activate t2pdecoder

# Install PyTorch
conda install pytorch>=2.0.1 -c pytorch

# Install other dependencies
conda install numpy=1.24.3 pandas=2.2.2 scikit-learn=1.3.0 scipy=1.11.1
```

### Using pip

```bash
pip install -r requirements.txt
```

Or install them individually:

```bash
pip install torch>=2.0.1 numpy==1.24.3 pandas==2.2.2 scikit-learn==1.3.0 scipy==1.11.1
```

# Tutorial

## File Structure

```
T2Pdecoder/
├── data/                      # Data directory
│   ├── gene_list.csv         # List of 18,860 genes used in the model
│   ├── pro_list.csv          # List of 5,738 proteins used in the model
│   ├── Pan_rna_df.csv        # Pan-cancer RNA expression data
│   ├── Pan_pro_df.csv        # Pan-cancer protein abundance data
│   ├── test_gbm_rna_df.csv   # Test GBM RNA expression data
│   ├── test_gbm_pro_df.csv   # Test GBM protein abundance data
│   ├── cite_seq_gbm_pro.csv  # CITE-seq GBM protein data
│   ├── rna_emb_df_cite.csv   # CITE-seq RNA embeddings
│   ├── pro_emb_df_gbm.csv    # GBM protein embeddings
│   └── cite_ft_pro_idx.csv   # Indices of shared proteins between CITE-seq and T2Pdecoder
│
├── src/                      # Source code directory
│   ├── process.py            # Data preprocessing script
│   ├── CLIP_pre_train.py     # CLIP model pre-training script
│   ├── CLIP_fine_tune.py     # CLIP model fine-tuning script
│   ├── CLIP_ft_app.py        # CLIP model application script
│   ├── T2Pdecoder_VAE_train.py  # VAE model training script
│   ├── T2Pdecoder_VAE_ft_cite.py  # VAE model fine-tuning script for CITE-seq
│   ├── T2Pdecoder_generator.py  # Protein prediction script
│   └── model.py              # Model architecture definitions
│
├── saved_model/             # Saved model directory
│   ├── embedding/           # Pre-trained and fine-tuned CLIP models
│   │   ├── rna_encoder_best.pth  # Pre-trained RNA encoder
│   │   ├── pro_encoder_best.pth  # Pre-trained protein encoder
│   │   └── glioma/             # Glioma fine-tuned models
│   │       ├── rna_encoder_best.pth  # Fine-tuned RNA encoder for glioma
│   │       └── pro_encoder_best.pth  # Fine-tuned protein encoder for glioma
│   └── T2Pdecoder/         # Fine-tuned models for different cancer types
│       ├── glioma/         # Glioma-specific VAE models
│       │   └── BN_VAE_best.pth  # Best VAE model for glioma
│       └── cite_gbm/       # CITE-seq GBM-specific models
│           └── cite_ft_best.pth  # Fine-tuned model for CITE-seq GBM
│
├── results/                 # Results directory for all outputs
└── requirements.txt         # Python package dependencies

```

### Key Files Description:

#### Data Files:
- `data/gene_list.csv`: Contains the list of 18,860 genes used in the model
- `data/pro_list.csv`: Contains the list of 5,738 proteins used in the model
- `data/test_gbm_rna_df.csv`: Test GBM RNA expression data
- `data/test_gbm_pro_df.csv`: Test GBM protein abundance data
- `data/cite_seq_gbm_pro.csv`: CITE-seq GBM protein data for fine-tuning
- `data/rna_emb_df_cite.csv`: CITE-seq RNA embeddings
- `data/pro_emb_df_gbm.csv`: GBM protein embeddings
- `data/cite_ft_pro_idx.csv`: Contains indices of proteins shared between CITE-seq data and T2Pdecoder's 5,738 proteins

#### Source Code Files:
- `src/process.py`: Preprocesses RNA and protein data to match model requirements
- `src/CLIP_pre_train.py`: Implements CLIP model pre-training
- `src/CLIP_fine_tune.py`: Implements CLIP model fine-tuning
- `src/CLIP_ft_app.py`: Applies trained CLIP model to generate embeddings
- `src/T2Pdecoder_VAE_train.py`: Implements VAE model training
- `src/T2Pdecoder_VAE_ft_cite.py`: Implements VAE model fine-tuning for CITE-seq data
- `src/T2Pdecoder_generator.py`: Generates protein predictions from RNA data
- `src/model.py`: Contains model architecture definitions

#### Configuration Files:
- `requirements.txt`: Lists all Python package dependencies with their versions

#### Output Files:
- `saved_model/`: Contains saved model checkpoints
  - `embedding/`: 
    - Pre-trained CLIP models (rna_encoder_best.pth, pro_encoder_best.pth)
    - Fine-tuned models for glioma:
      - rna_encoder_best.pth (Fine-tuned RNA encoder)
      - pro_encoder_best.pth (Fine-tuned protein encoder)
  - `T2Pdecoder/`: 
    - Glioma-specific VAE models (BN_VAE_best.pth)
    - CITE-seq GBM-specific models (cite_ft_best.pth)

## 1. Data Preprocessing

### RNA Data Requirements:
- Data should be log2 transformed
- Each gene should be z-score normalized
- First column should be named "PID"

### Protein Data Requirements:
- Sample-level quantile normalization
- NA imputation
- Each protein should be z-score normalized
- First column should be named "PID"

### Run Preprocessing:
```bash
python src/process.py --pro_dir <protein_data_path> --rna_dir <rna_data_path> --out_dir <output_directory>
```

Parameters:
- `--pro_dir`: Path to protein data file (default: '')
- `--rna_dir`: Path to RNA data file (default: '')
- `--out_dir`: Directory to save processed data (default: ./results/matched/)

Output files:
- `matched_rna_df.csv`: Processed RNA data
- `matched_pro_df.csv`: Processed protein data

## 2. CLIP Model Pre-training

```bash
python src/CLIP_pre_train.py --ep <epochs> --dim <embedding_dim> --batch <batch_size> --lr <learning_rate> --pro_dir <protein_data_path> --rna_dir <rna_data_path> --out_dir <model_save_dir>
```

Parameters:
- `--ep`: Number of epochs (default: 100)
- `--dim`: Embedding dimension (default: 12)
- `--batch`: Batch size (default: 68)
- `--lr`: Learning rate (default: 0.0005)
- `--pro_dir`: Path to protein data file (default: ./data/test_gbm_pro_df.csv)
- `--rna_dir`: Path to RNA data file (default: ./data/test_gbm_rna_df.csv)
- `--out_dir`: Directory to save model (default: ./results/embedding/)

Output files:
- `rna_encoder_best.pth`: Best RNA encoder model
- `pro_encoder_best.pth`: Best protein encoder model
- `loss.csv`: Training loss records
- `train_pid.csv` and `test_pid.csv`: Sample IDs for training and test sets

## 3. CLIP Model Fine-tuning

```bash
python src/CLIP_fine_tune.py --ep <epochs> --dim <embedding_dim> --batch <batch_size> --lr <learning_rate> --pro_dir <protein_data_path> --rna_dir <rna_data_path> --pretrain_pth <pretrained_model_path> --out_dir <model_save_dir>
```

Parameters:
- `--ep`: Number of epochs (default: 20)
- `--dim`: Embedding dimension (default: 12)
- `--batch`: Batch size (default: 68)
- `--lr`: Learning rate (default: 0.0005)
- `--pro_dir`: Path to protein data file (default: ./data/test_gbm_pro_df.csv)
- `--rna_dir`: Path to RNA data file (default: ./data/test_gbm_rna_df.csv)
- `--pretrain_pth`: Path to pretrained model (default: ./saved_model/embedding/)
- `--out_dir`: Directory to save model (default: ./results/embedding/fine_tune/)

Output files:
- `rna_encoder_best.pth`: Fine-tuned RNA encoder model
- `pro_encoder_best.pth`: Fine-tuned protein encoder model
- `loss.csv`: Training loss records
- `train_pid.csv` and `test_pid.csv`: Sample IDs for training and test sets

## 4. VAE Model Training

```bash
python src/T2Pdecoder_VAE_train.py --num_epochs <epochs> --batch <batch_size> --lr <learning_rate> --dim <embedding_dim> --pro_dir <protein_data_path> --emb_dir <embedding_data_path> --out_dir <model_save_dir>
```

Parameters:
- `--num_epochs`: Number of epochs (default: 200)
- `--batch`: Batch size (default: 184)
- `--lr`: Learning rate (default: 0.0005)
- `--dim`: Embedding dimension (default: 12)
- `--pro_dir`: Path to protein data file (default: ./data/test_gbm_pro_df.csv)
- `--emb_dir`: Path to embedding data file (default: ./data/pro_emb_df_gbm.csv)
- `--out_dir`: Directory to save model (default: ./results/VAE/)

Output files:
- `BN_VAE_best.pth`: Best VAE model
- `train_loss.csv` and `test_loss.csv`: Training loss records
- `train_pid.csv` and `test_pid.csv`: Sample IDs for training and test sets

## 5. Using T2Pdecoder for Prediction

### Step 1: Generate RNA Embeddings
Use CLIP_ft_app.py to generate embeddings for RNA data:
```bash
python src/CLIP_ft_app.py --rna_dir <rna_data_path> --model_pth <model_path> --out_dir <embedding_save_dir>
```

Parameters:
- `--rna_dir`: Path to RNA data file (default: '')
- `--model_pth`: Path to trained model (default: ./saved_model/embedding/glioma/)
- `--out_dir`: Directory to save embeddings (default: ./results/embedding/fine_tune/)

Output files:
- `rna_emb_df_<save_name>.csv`: Generated RNA embeddings

### Step 2: Generate Protein Predictions
```bash
python src/T2Pdecoder_generator.py --model_pth <vae_model_path> --dim <embedding_dim> --emb_dir <rna_embedding_path> --out_dir <prediction_save_dir>
```

Parameters:
- `--model_pth`: Path to trained VAE model (default: ./saved_model/BN_VAE_best.pth)
- `--dim`: Embedding dimension (default: 12)
- `--emb_dir`: Path to RNA embedding data file (default: ./data/rna_emb_df_cohort.csv)
- `--out_dir`: Directory to save predictions (default: ./results/VAE)

Output files:
- `generated_data_<sample_count>.csv`: Predicted protein expression data

## 6. Fine-tuning on CITE-seq Data

### Step 1: Generate RNA Embeddings
Use the pre-trained CLIP RNA encoder to generate embeddings for CITE-seq RNA data:
```bash
python src/CLIP_ft_app.py --rna_dir <cite_seq_rna_data> --model_pth <pretrained_clip_model> --out_dir <embedding_save_dir>
```

### Step 2: Prepare Protein Data
1. Find the intersection between CITE-seq proteins and T2Pdecoder's 5,738 proteins
2. Generate indices of shared proteins in T2Pdecoder's protein list (0-based indexing)
3. Save the indices to `./data/cite_ft_pro_idx.csv`

### Step 3: Fine-tune Model
```bash
python src/T2Pdecoder_VAE_ft_cite.py --num_epochs <epochs> --batch <batch_size> --lr <learning_rate> --dim <embedding_dim> --pro_dir <protein_data_path> --emb_dir <embedding_data_path> --out_dir <model_save_dir>
```

Parameters:
- `--num_epochs`: Number of epochs (default: 300)
- `--batch`: Batch size (default: 1024)
- `--lr`: Learning rate (default: 0.0001)
- `--dim`: Embedding dimension (default: 12)
- `--pro_dir`: Path to protein data file (default: ./data/cite_seq_gbm_pro.csv)
- `--emb_dir`: Path to embedding data file (default: ./data/rna_emb_df_cite.csv)
- `--out_dir`: Directory to save model (default: ./results/VAE/cite_ft/)

Output files:
- `cite_ft_best.pth`: Fine-tuned model for CITE-seq data
- `sel_pro_generate_best_test.csv`: Generated protein predictions for test set
- `train_loss.csv` and `test_loss.csv`: Training loss records
- `train_pid.csv` and `test_pid.csv`: Sample IDs for training and test sets

## Notes
1. Ensure all input data formats are correct
2. Check sample ID correspondences
3. Ensure sufficient disk space for intermediate results and models
4. Protein order in predictions can be found in `./data/pro_list.csv`

