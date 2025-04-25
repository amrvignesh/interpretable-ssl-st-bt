# Interpretable Semi-Supervised Swin Transformer for Brain Tumor Segmentation

[![GitHub repo](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/amrvignesh/interpretable-ssl-st-bt)
## Overview

This project implements a deep learning framework for segmenting brain tumor subregions (enhancing tumor, necrotic core, edema) from multi-modal MRI scans using the BraTS 2023 dataset. It leverages the **Swin Transformer (SwinUNETR)** architecture within a **Semi-Supervised Learning (SSL)** setting using **Consistency Regularization** to improve data efficiency. Furthermore, it incorporates **Attention Rollout** for **interpretable visualizations**, aiming to enhance clinical trust and adoption.

The core idea is to address the limitations of traditional CNNs in capturing long-range dependencies and the challenge of limited annotated medical data by combining the power of transformers with SSL and explainability techniques.

## Features

* **Segmentation Model:** Uses the state-of-the-art SwinUNETR architecture from MONAI.
* **Dataset:** Designed for the [BraTS 2023 Adult Glioma dataset](https://www.synapse.org/#!Synapse:syn51514105/wiki/622358) (part of the BraTS Challenge).
* **Semi-Supervised Learning:** Implements Consistency Regularization to leverage both labeled and unlabeled data, improving performance under data scarcity.
* **Interpretability:** Includes (placeholder for) Attention Rollout to visualize the model's focus areas, providing explainability.
* **Framework:** Built using Python, PyTorch, and the [MONAI](https://monai.io/) framework for medical imaging.

## Dataset

* This project uses the **ASNR-MICCAI-BraTS2023 Glioma Challenge dataset**.
* **Access:** You need to register for the BraTS challenge and download the data from the official Synapse repository: [syn51514105](https://www.synapse.org/#!Synapse:syn51514105).
* **Prerequisites:** Requires the `synapseclient` Python package and a Synapse account configured with credentials.
* **Local Structure:** The code expects the downloaded and unzipped data to be organized locally. The `brats_dsc.csv` file (or a similar manifest) is used by the scripts to locate patient folders and MRI files (`t1c`, `t1n`, `t2f`, `t2w`, `seg`). Ensure the paths in your manifest file (`brats_dsc.csv`) are correct for your system.

## Requirements

* Python (3.8+)
* PyTorch (1.10+ recommended)
* MONAI (ensure compatibility with PyTorch version)
* Nibabel (for reading NIfTI files)
* Pandas (for reading CSV manifests)
* Matplotlib (for plotting)
* Jupyter Notebook / Lab (for running the demo notebook)
* `synapseclient` (for downloading data initially)


**Download Dataset:** Use synapseclient or manual download from Synapse into a designated data directory (e.g., ./data/brats2021challenge/). Unzip the ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip.

**Prepare Data Manifest:** Ensure you have a CSV file (like brats_dsc.csv) listing the absolute paths of the downloaded dataset structure. Place this file in the directory specified by --data_dir in the training script.

**Install Requirements:** Install the necessary Python packages (see Requirements section).

## Usage

### 1. Mini Training Verification (Optional but Recommended)
Run the brats_ssl_minitrain_notebook.ipynb notebook to verify data loading, transforms, and the basic training loop setup using a small subset of data. This helps catch configuration errors quickly.

### 2. Full Training
Use the main training script brats_ssl_train.py. Adjust parameters as needed via command-line arguments.
```
python brats_ssl_train.py \
    --data_dir /path/to/your/data_and_csv \
    --structure_csv your_structure_file.csv \
    --output_dir ./output_brats_ssl \
    --roi_size 128 128 128 \
    --batch_size 2 \
    --labeled_bs_ratio 0.5 \
    --max_epochs 150 \
    --val_every 10 \
    --lr 1e-4 \
    --consistency_weight 1.0 \
    --num_workers 8
    # Add other arguments as needed
```
Checkpoints (model_best.pt, model_latest.pt) and training history (training_history.json, training_history.png) will be saved in the --output_dir.3. Demonstration and VisualizationRun the brats_ssl_demo_notebook.ipynb notebook:It loads the best trained model (model_best.pt).Performs inference on a sample case.Visualizes the input MRI, ground truth segmentation, and model prediction.Includes a placeholder section for visualizing Attention Rollout maps (requires implementation).Summarizes quantitative results from the training history.

### File Structure (Key Files)interpretable-ssl-st-bt/
```
│
├── brats_ssl_train.py             # Main training script
├── brats_ssl_minitrain_notebook.ipynb # Notebook for quick verification
├── brats_ssl_demo_notebook.ipynb  # Notebook for results demo & visualization
│
├── data/                          # Recommended location for dataset & manifests
│   ├── brats2021challenge/        # Example data root folder
│   │   ├── ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/ # Unzipped data
│   │   │   └── BraTS-GLI-00000-000/
│   │   │       ├── BraTS-GLI-00000-000-t1c.nii.gz
│   │   │       ├── ... (other modalities and seg.nii.gz)
│   │   │   └── ... (other patient folders)
│   │   └── ... (Validation data zip, etc.)
│   └── brats_dsc.csv              # Example data structure manifest file
│
├── output_brats_ssl/              # Default output directory for training
│   ├── model_best.pt              # Best model checkpoint
│   ├── model_latest.pt            # Latest model checkpoint
│   ├── training_history.json      # Training metrics log
│   └── training_history.png       # Training plots
│
├── output_brats_ssl_notebook_viz/ # Default output for demo notebook visualizations
│   └── ... (saved segmentation/attention images)
│
├── fullpaper.pdf                  # Research paper draft (if included)
├── Poster_Presentation_VRAJA.pdf  # Poster presentation (if included)
├── README.md                      # This file
└── ... (Other project files: .gitignore, requirements.txt, etc.)
```

## Results
- Trained models and logs are saved in the specified output directory (default: ./output_brats_ssl/).
- Segmentation visualizations and quantitative summaries can be generated using the results_notebook.ipynb.
- Refer to the associated paper (paper-proposal.pdf) and poster (poster.pdf) for detailed methodology and findings.

## Citation / Key References

- BraTS Dataset: Baid, U., et al. "The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification." arXiv preprint arXiv:2107.02314 (2021). (Update with BraTS 2023 citation when available)

- Swin Transformer: Liu, Z., et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

- MONAI: Cardoso, M. J., et al. "MONAI: An open-source framework for deep learning in healthcare." Medical Image Analysis (MedIA), 2022. https://monai.io/

- Consistency Regularization (Example): Sohn, K., et al. "FixMatch: Simplifying semi-supervised learning with consistency and confidence." Advances in Neural Information Processing Systems (NeurIPS), 2020.

- Attention Rollout: Abnar, S., and Zuidema, W. "Quantifying attention flow in transformers." Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL), 2020.

## Authors

- Vignesh Azhagiyanambi Madaswamy Raja (vraja@gsu.edu)
- Sowmya Gonugunta (sgonuguntal@student.gsu.edu)
Department of Computer Science, Georgia State University

### License: 
This project is licensed under the MIT License.

