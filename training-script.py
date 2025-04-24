# -*- coding: utf-8 -*-
"""
Training a SwinUNETR model on BraTS 2023
using Semi-Supervised Learning (Consistency Regularization) and MONAI.

Data loading updated to use a structure CSV file (e.g., brats_dsc.csv).
"""

# --- Essential Imports ---
import os
import json
import shutil
import tempfile
import time
import argparse
import warnings
from functools import partial # Added for model_inferer

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import pandas as pd # For reading the structure CSV
import torch
import torch.nn.functional as F # For consistency loss if using KL divergence etc.
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset

# MONAI Imports
from monai.config import print_config
from monai.data import Dataset, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, DiceCELoss # DiceCE might be more stable
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai import transforms
from monai.transforms import (
    AsDiscrete,
    Activations,
    Compose,
    LoadImaged, # Added for explicitness
    EnsureChannelFirstd, # Added for explicitness
    EnsureTyped, # Added for explicitness
    ConvertToMultiChannelBasedOnBratsClassesd, # Added for explicitness
    NormalizeIntensityd, # Added for explicitness
    CropForegroundd, # Added for explicitness
    RandSpatialCropd, # Added for explicitness
    RandFlipd, # Added for explicitness
    RandRotate90d, # Added for explicitness
    RandScaleIntensityd, # Added for explicitness
    RandShiftIntensityd, # Added for explicitness
    RandGaussianNoised, # Added for strong aug
    RandGaussianSmoothd, # Added for strong aug
)
from monai.utils.enums import MetricReduction

# Suppress MONAI warnings if needed
# warnings.filterwarnings("ignore", category=UserWarning, module='monai')

print_config()

# --- Configuration & Constants ---
# Use argparse for better command-line control
parser = argparse.ArgumentParser(description="BraTS 2023 SSL SwinUNETR Training")
# Note: data_dir should contain the structure_csv file AND the unzipped data folder
parser.add_argument('--data_dir', type=str, default='/home/users/vraja/dl/data/brats2021challenge', help='Directory containing BraTS data folder and structure CSV')
parser.add_argument('--structure_csv', type=str, default='brats_dsc.csv', help='Filename of the CSV listing dataset structure (e.g., brats_dsc.csv)')
parser.add_argument('--output_dir', type=str, default='./output_brats_ssl', help='Directory to save models and logs')
parser.add_argument('--roi_size', type=int, nargs=3, default=[128, 128, 128], help='Input ROI size (x, y, z)')
parser.add_argument('--batch_size', type=int, default=1, help='Total batch size (labeled + unlabeled)')
parser.add_argument('--labeled_bs_ratio', type=float, default=0.5, help='Ratio of labeled samples in a batch')
parser.add_argument('--sw_batch_size', type=int, default=4, help='Sliding window batch size for validation')
parser.add_argument('--infer_overlap', type=float, default=0.5, help='Sliding window inference overlap')
parser.add_argument('--max_epochs', type=int, default=150, help='Maximum training epochs')
parser.add_argument('--val_every', type=int, default=10, help='Run validation every N epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Optimizer weight decay')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--num_workers', type=int, default=8, help='DataLoader workers')
parser.add_argument('--labeled_ratio', type=float, default=0.4, help='Ratio of data to use as labeled (e.g., 0.4 for 40%)')
parser.add_argument('--consistency_weight', type=float, default=1.0, help='Weight for the consistency loss term')
# Add more args as needed (e.g., for specific SSL params, model features)

# Check if running in notebook or script environment for arg parsing
if 'ipykernel' in sys.modules:
    # Use default args in notebook environment
    args = parser.parse_args([])
    print("Running in notebook environment, using default args.")
else:
    args = parser.parse_args()
    print("Running in script environment, parsing command-line args.")


# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# --- Helper Classes & Functions ---

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def get_brats2023_datalists(data_dir, structure_csv_filename, labeled_ratio=0.4):
    """
    Reads the BraTS 2023 structure CSV file and creates labeled/unlabeled data lists.

    Args:
        data_dir (str): Path to the base directory containing the structure CSV
                        and the 'ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData' folder.
        structure_csv_filename (str): Name of the CSV file listing the dataset structure.
        labeled_ratio (float): Fraction of data to be considered labeled.

    Returns:
        tuple: (labeled_files, unlabeled_files) - lists of dictionaries for MONAI datasets.
               Each dictionary should have keys like 'image' (list of paths) and 'label' (path).
               Paths are expected to be absolute based on the brats_dsc.csv example.
    """
    structure_csv_path = os.path.join(data_dir, structure_csv_filename)
    if not os.path.exists(structure_csv_path):
        raise FileNotFoundError(f"Structure CSV file not found: {structure_csv_path}")

    df = pd.read_csv(structure_csv_path)

    # Define the root directory based on the CSV structure and data_dir argument
    # Assuming the structure CSV is in data_dir and contains absolute paths starting from /home/users/vraja/dl/
    # We will primarily rely on the absolute paths in the CSV.
    training_data_root_identifier = "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    print(f"Using structure CSV: {structure_csv_path}")
    print(f"Identifying patient folders within paths containing: {training_data_root_identifier}")

    all_files = []
    # Filter for directories within the training data folder
    patient_dirs = df[(df['Is Directory'] == True) &
                      (df['Path'].str.contains(training_data_root_identifier)) &
                      (df['Path'].str.contains('BraTS-GLI-'))]['Path'].tolist()

    print(f"Found {len(patient_dirs)} potential patient directories.")

    # Iterate through identified patient directories
    for patient_path in patient_dirs:
        patient_folder_name = os.path.basename(patient_path) # e.g., 'BraTS-GLI-00000-000'

        # Construct expected file paths using the absolute patient path from CSV
        image_files = [
            os.path.join(patient_path, f"{patient_folder_name}-t1c.nii.gz"),
            os.path.join(patient_path, f"{patient_folder_name}-t1n.nii.gz"),
            os.path.join(patient_path, f"{patient_folder_name}-t2f.nii.gz"), # FLAIR
            os.path.join(patient_path, f"{patient_folder_name}-t2w.nii.gz"), # T2
        ]
        label_file = os.path.join(patient_path, f"{patient_folder_name}-seg.nii.gz")

        # Check if all required files exist
        if all(os.path.exists(f) for f in image_files) and os.path.exists(label_file):
            all_files.append({"image": image_files, "label": label_file})
        else:
            # Check which specific files are missing for better debugging
            missing = [f for f in image_files + [label_file] if not os.path.exists(f)]
            print(f"Warning: Missing files for patient {patient_folder_name}, skipping. Missing: {missing}")


    if not all_files:
        raise ValueError("No valid data files found. Check data_dir, structure_csv_filename, "
                         "CSV content (absolute paths), and file existence.")

    # Shuffle and split
    np.random.shuffle(all_files)
    num_labeled = int(len(all_files) * labeled_ratio)
    labeled_files = all_files[:num_labeled]
    # Use the remaining files as unlabeled. If you want to use *all* data for SSL (both labeled and unlabeled inputs),
    # you can set unlabeled_files = all_files. The training loop handles using labels only when available.
    unlabeled_files = all_files[num_labeled:]

    print(f"Total valid cases processed: {len(all_files)}")
    print(f"Using {len(labeled_files)} cases as labeled data.")
    print(f"Using {len(unlabeled_files)} cases as unlabeled data.")

    # It's often useful to have a separate validation set split *before* SSL splitting
    # For simplicity here, we'll use a portion of labeled data for validation later.
    # A more robust approach is a dedicated validation split.

    return labeled_files, unlabeled_files


def save_checkpoint(model, epoch, optimizer, filename="model.pt", best_acc=0.0, dir_add=args.output_dir):
    """Saves model checkpoint."""
    # Ensure the directory exists before saving
    os.makedirs(dir_add, exist_ok=True)
    state_dict = model.state_dict()
    save_dict = {
        "epoch": epoch,
        "best_acc": best_acc,
        "state_dict": state_dict,
    }
    # Save optimizer state only if it's provided and valid
    if optimizer is not None:
         save_dict["optimizer"] = optimizer.state_dict()

    filename = os.path.join(dir_add, filename)
    try:
        torch.save(save_dict, filename)
        print(f"Saving checkpoint {filename}")
    except Exception as e:
        print(f"Error saving checkpoint {filename}: {e}")


# --- Define Transforms ---
# Note: BraTS labels are: 1: NCR/NET, 2: ED, 4: ET
# ConvertToMultiChannelBasedOnBratsClassesd handles this mapping.
# Output channels will be: Channel 0: ET, Channel 1: TC (ET+NCR/NET), Channel 2: WT (ET+NCR/NET+ED)
# Check if this matches SwinUNETR output expectation (out_channels=3) and DiceLoss calculation.
# Or adjust `out_channels` and loss calculation accordingly. Let's assume 3 output channels for ET, TC, WT.

# Transforms for labeled data (Supervised Learning & Weak Aug for SSL)
train_transforms_weak = Compose(
    [
        LoadImaged(keys=["image", "label"], image_only=False, ensure_channel_first=True), # Combine loading and EnsureChannelFirst
        # Map BraTS labels (1, 2, 4) to multi-channel format (ET, TC, WT)
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        # Consider adding orientation standardization if needed
        # transforms.Orientationd(keys=["image", "label"], axcodes="RAS"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # Spacing standardization might be important if voxel sizes vary significantly
        # transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        CropForegroundd(
            keys=["image", "label"], source_key="image", k_divisible=args.roi_size # Ensure divisibility
        ),
        # Spatial Augmentations (Weak)
        RandSpatialCropd(keys=["image", "label"], roi_size=args.roi_size, random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.1, max_k=3),
        # Intensity Augmentations (Weak)
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
    ]
)

# Transforms for unlabeled data (Strong Augmentation for SSL)
# Starts similarly to weak, then adds stronger augmentations
train_transforms_strong = Compose(
    [
        LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True), # Load only image
        EnsureTyped(keys=["image"], dtype=torch.float32),
        # transforms.Orientationd(keys=["image"], axcodes="RAS"), # Apply if used in weak
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # transforms.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"), # Apply if used in weak
        CropForegroundd(keys=["image"], source_key="image", k_divisible=args.roi_size),
        # Weak spatial augs first
        RandSpatialCropd(keys=["image"], roi_size=args.roi_size, random_size=False),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image"], prob=0.1, max_k=3),
        # Weak intensity augs
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        # --- Add Stronger Augmentations Here ---
        RandGaussianNoised(keys="image", prob=0.3, mean=0.0, std=0.1),
        RandGaussianSmoothd(keys="image", prob=0.3, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
        # Add more aggressive intensity/geometric transforms if desired
        RandScaleIntensityd(keys="image", factors=0.2, prob=0.2), # More scaling
        RandShiftIntensityd(keys="image", offsets=0.2, prob=0.2), # More shifting
        # Consider RandAugment or CutMix/MixUp if applicable and adapted for 3D
        # --- End Strong Augmentations ---
    ]
)


# Validation transforms (minimal augmentation)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], image_only=False, ensure_channel_first=True),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # Add Spacingd or Orientationd if they were used during training
        # transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ]
)

# --- Custom Dataset for SSL ---
# This dataset returns both weakly and strongly augmented views of an image.
class SSLDataset(Dataset):
    def __init__(self, data, transform_weak, transform_strong):
        # Initialize with file paths, transforms will handle loading
        super().__init__(data=data, transform=None)
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

    def __getitem__(self, index):
        # Get file paths dictionary for the current index
        data_i = self.data[index]

        # Apply weak transform (handles loading image+label if available)
        # Need to handle potential missing 'label' key for unlabeled data
        data_weak = self.transform_weak(data_i.copy())
        img_weak = data_weak['image']
        # Include label if it exists (for supervised part)
        label_weak = data_weak.get('label', None) # Use .get for safety

        # Apply strong transform (handles loading image only)
        # Pass only the image path to the strong transform
        data_strong_input = {'image': data_i['image']}
        img_strong = self.transform_strong(data_strong_input)['image']

        # Return dictionary including both augmented images and the label (if present)
        output = {"image_weak": img_weak, "image_strong": img_strong}
        if label_weak is not None:
            output["label"] = label_weak # This label corresponds to the weakly augmented image
        return output


# --- Create Datasets and DataLoaders ---
print("Creating datasets and dataloaders...")
labeled_files, unlabeled_files = get_brats2023_datalists(
    args.data_dir, args.structure_csv, args.labeled_ratio
)

# Split labeled data for validation (e.g., 80% train, 20% val)
# Ensure there are enough labeled files for the split
if len(labeled_files) < 5: # Need at least a few samples for train/val
    raise ValueError(f"Only {len(labeled_files)} labeled files found. Need more for train/validation split.")
val_split_idx = max(1, int(len(labeled_files) * 0.8)) # Ensure at least one validation sample
train_labeled_files = labeled_files[:val_split_idx]
val_files = labeled_files[val_split_idx:]

print(f"Using {len(train_labeled_files)} cases for labeled training.")
print(f"Using {len(val_files)} cases for validation.")

# Create the SSL datasets
# Labeled data uses the SSLDataset structure but will have the 'label' key
train_ds_labeled = SSLDataset(data=train_labeled_files, transform_weak=train_transforms_weak, transform_strong=train_transforms_strong)
# Unlabeled data uses the same structure but won't have the 'label' key in the output dict
train_ds_unlabeled = SSLDataset(data=unlabeled_files, transform_weak=train_transforms_weak, transform_strong=train_transforms_strong)

# Validation dataset and loader (uses standard transforms)
val_ds = Dataset(data=val_files, transform=val_transforms)

# Calculate batch sizes
labeled_batch_size = int(args.batch_size * args.labeled_bs_ratio)
unlabeled_batch_size = args.batch_size - labeled_batch_size

# Ensure batch sizes are at least 1 if data exists
if not train_labeled_files:
    labeled_batch_size = 0
    warnings.warn("No labeled training files available.")
elif labeled_batch_size == 0 and unlabeled_batch_size > 0:
     labeled_batch_size = 1 # Ensure at least 1 if possible
     unlabeled_batch_size = max(0, args.batch_size - 1)
     warnings.warn("Labeled batch size was 0, adjusting to 1.")

if not train_ds_unlabeled:
     unlabeled_batch_size = 0
     warnings.warn("No unlabeled training files available.")
elif unlabeled_batch_size == 0 and labeled_batch_size > 0:
     unlabeled_batch_size = 1 # Ensure at least 1 if possible
     labeled_batch_size = max(0, args.batch_size - 1)
     warnings.warn("Unlabeled batch size was 0, adjusting to 1.")

if labeled_batch_size == 0 and unlabeled_batch_size == 0:
    raise ValueError("Both labeled and unlabeled batch sizes are zero. No data to train.")


print(f"Effective Batch size: {labeled_batch_size + unlabeled_batch_size} (Labeled: {labeled_batch_size}, Unlabeled: {unlabeled_batch_size})")

# Create DataLoaders
# Use separate loaders and iterate simultaneously in the training loop
train_loader_labeled = DataLoader(
    train_ds_labeled, batch_size=labeled_batch_size, shuffle=True,
    num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=list_data_collate,
    drop_last=True # Important if batch sizes vary / last batch is small
) if labeled_batch_size > 0 else None

train_loader_unlabeled = DataLoader(
    train_ds_unlabeled, batch_size=unlabeled_batch_size, shuffle=True,
    num_workers=args.num_workers, pin_memory=torch.cuda.is_available(), collate_fn=list_data_collate,
    drop_last=True # Important if batch sizes vary / last batch is small
) if unlabeled_batch_size > 0 else None

val_loader = DataLoader(
    val_ds, batch_size=1, shuffle=False, # Val batch size usually 1 for sliding window
    num_workers=args.num_workers, pin_memory=torch.cuda.is_available()
)

# --- Model, Loss, Optimizer, Scheduler ---
print("Initializing model, loss, optimizer...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SwinUNETR(
    img_size=args.roi_size,
    in_channels=4,  # t1c, t1n, t2f, t2w
    out_channels=3, # ET, TC, WT (Verify this matches ConvertToMultiChannelBasedOnBratsClassesd)
    feature_size=48, # Default, adjust as needed
    use_checkpoint=True, # Gradient checkpointing for memory saving
).to(device)

# Loss Functions
# Supervised loss (for labeled data)
supervised_loss = DiceCELoss(to_onehot_y=False, sigmoid=True, lambda_dice=0.5, lambda_ce=0.5)

# Consistency loss (for unlabeled data) - MSE between sigmoid outputs
consistency_loss = torch.nn.MSELoss()

# Post-processing for validation/inference
post_sigmoid = Activations(sigmoid=True) # Apply sigmoid to model output logits
post_pred = AsDiscrete(threshold=0.5) # Thresholding to get discrete segmentation map

# Metric for validation
dice_metric = DiceMetric(include_background=False, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)

# Sliding window inference for validation
model_inferer = partial(
    sliding_window_inference,
    roi_size=args.roi_size,
    sw_batch_size=args.sw_batch_size,
    predictor=model,
    overlap=args.infer_overlap,
    mode="gaussian", # Smoother blending
    progress=True # Show progress bar
)

# --- Training and Validation Functions ---

def train_epoch(model, labeled_loader, unlabeled_loader, optimizer, epoch, sup_loss_fn, cons_loss_fn, cons_weight):
    model.train()
    run_loss_sup = AverageMeter()
    run_loss_cons = AverageMeter()
    run_loss_total = AverageMeter()
    start_time = time.time()

    # Determine the number of batches based on the longer loader, or interleave
    # Simple approach: iterate up to the length of the longer loader
    len_l = len(labeled_loader) if labeled_loader else 0
    len_u = len(unlabeled_loader) if unlabeled_loader else 0
    steps = max(len_l, len_u)

    # Create iterators
    iter_l = iter(labeled_loader) if labeled_loader else None
    iter_u = iter(unlabeled_loader) if unlabeled_loader else None

    print(f"Starting Epoch {epoch+1}/{args.max_epochs}, Steps: {steps}")

    for i in range(steps):
        optimizer.zero_grad()
        total_loss_batch = 0.0
        current_labeled_bs = 0
        current_unlabeled_bs = 0

        # --- Supervised Loss (Labeled Data) ---
        if iter_l:
            try:
                batch_labeled = next(iter_l)
                # Data comes from SSLDataset: image_weak, image_strong, label
                images_l_weak, labels_l = batch_labeled["image_weak"].to(device), batch_labeled["label"].to(device)
                current_labeled_bs = images_l_weak.size(0)

                logits_l = model(images_l_weak) # Predict on weakly augmented labeled data
                loss_s = sup_loss_fn(logits_l, labels_l)
                run_loss_sup.update(loss_s.item(), n=current_labeled_bs)
                total_loss_batch += loss_s

            except StopIteration:
                # Reset iterator if it finishes early
                iter_l = iter(labeled_loader)
                batch_labeled = next(iter_l)
                images_l_weak, labels_l = batch_labeled["image_weak"].to(device), batch_labeled["label"].to(device)
                current_labeled_bs = images_l_weak.size(0)
                logits_l = model(images_l_weak)
                loss_s = sup_loss_fn(logits_l, labels_l)
                run_loss_sup.update(loss_s.item(), n=current_labeled_bs)
                total_loss_batch += loss_s
            except Exception as e:
                 print(f"Error processing labeled batch {i}: {e}")
                 continue # Skip batch on error


        # --- Consistency Loss (Unlabeled Data) ---
        if iter_u:
            try:
                batch_unlabeled = next(iter_u)
                # Data comes from SSLDataset: image_weak, image_strong (no label key)
                images_u_weak, images_u_strong = batch_unlabeled["image_weak"].to(device), batch_unlabeled["image_strong"].to(device)
                current_unlabeled_bs = images_u_weak.size(0)

                # Generate pseudo-labels from weakly augmented data
                with torch.no_grad():
                    logits_u_weak = model(images_u_weak)
                    pseudo_labels = torch.sigmoid(logits_u_weak.detach())
                    # Optional: Confidence thresholding could be applied here

                # Predict on strongly augmented data
                logits_u_strong = model(images_u_strong)
                preds_strong_sig = torch.sigmoid(logits_u_strong)

                # Calculate consistency loss
                loss_c = cons_loss_fn(preds_strong_sig, pseudo_labels)
                # Optional: Apply mask if using confidence thresholding
                # loss_c = (loss_c * mask).mean() / mask.mean().clamp(min=1e-6)

                run_loss_cons.update(loss_c.item(), n=current_unlabeled_bs)
                total_loss_batch += cons_weight * loss_c

            except StopIteration:
                # Reset iterator if it finishes early
                iter_u = iter(unlabeled_loader)
                batch_unlabeled = next(iter_u)
                images_u_weak, images_u_strong = batch_unlabeled["image_weak"].to(device), batch_unlabeled["image_strong"].to(device)
                current_unlabeled_bs = images_u_weak.size(0)
                with torch.no_grad():
                    logits_u_weak = model(images_u_weak)
                    pseudo_labels = torch.sigmoid(logits_u_weak.detach())
                logits_u_strong = model(images_u_strong)
                preds_strong_sig = torch.sigmoid(logits_u_strong)
                loss_c = cons_loss_fn(preds_strong_sig, pseudo_labels)
                run_loss_cons.update(loss_c.item(), n=current_unlabeled_bs)
                total_loss_batch += cons_weight * loss_c
            except Exception as e:
                 print(f"Error processing unlabeled batch {i}: {e}")
                 continue # Skip batch on error


        # Backpropagation (only if loss was computed)
        if isinstance(total_loss_batch, torch.Tensor) and total_loss_batch != 0:
            total_loss_batch.backward()
            optimizer.step()
            # Update total loss average meter (use total batch size for weighting)
            total_bs = current_labeled_bs + current_unlabeled_bs
            if total_bs > 0:
                run_loss_total.update(total_loss_batch.item(), n=total_bs)

            # Print progress less frequently (e.g., every 10 batches or %)
            if (i + 1) % 20 == 0 or (i + 1) == steps:
                 print(
                     f"Epoch {epoch+1}/{args.max_epochs} Batch {i+1}/{steps} | "
                     f"Loss Sup: {run_loss_sup.avg:.4f} | Loss Cons: {run_loss_cons.avg:.4f} | "
                     f"Loss Total: {run_loss_total.avg:.4f} | "
                     f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                     f"Time: {(time.time() - start_time):.2f}s"
                 )
                 start_time = time.time() # Reset timer

    print(f"--- Epoch {epoch+1} Training Finished ---")
    return run_loss_total.avg


def val_epoch(model, loader, epoch, acc_func, model_inferer, post_pred_val):
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()
    print(f"--- Starting Validation Epoch {epoch+1} ---")

    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            # Ensure keys exist before accessing
            if "image" not in batch_data or "label" not in batch_data:
                print(f"Warning: Skipping validation batch {i} due to missing keys.")
                continue

            val_images, val_labels = batch_data["image"].to(device), batch_data["label"].to(device)

            try:
                # Sliding window inference
                val_logits = model_inferer(val_images) # Shape: (1, num_classes, H, W, D)

                # Post-processing
                val_outputs = post_pred_val(val_logits) # Apply sigmoid + threshold

                # Detach and move to CPU for metric calculation
                val_outputs_list = decollate_batch(val_outputs)
                val_labels_list = decollate_batch(val_labels) # Ensure labels are also decollated

                # Calculate Dice score for this case
                acc_func.reset()
                acc_func(y_pred=val_outputs_list, y=val_labels_list)
                acc, not_nans = acc_func.aggregate()

                # Handle cases where Dice is NaN (e.g., empty ground truth and prediction)
                # Update accumulator only with valid scores
                valid_acc = acc[not_nans].cpu().numpy()
                valid_n = not_nans.cpu().numpy().astype(int) # Number of valid classes for this case
                if np.sum(valid_n) > 0: # Only update if there are valid classes
                     run_acc.update(valid_acc, n=valid_n)

                # Print stats per validation case (optional)
                dice_scores = acc.cpu().numpy() # Dice per class for the current case (can include NaNs)
                mean_dice_case = np.nanmean(dice_scores) # Use nanmean to ignore NaNs
                print(f"Val Case {i+1}/{len(loader)} | Dice Per Class: {dice_scores} | Avg Dice (valid classes): {mean_dice_case:.4f}")

            except Exception as e:
                print(f"Error during validation case {i+1}: {e}")
                continue # Skip to next validation case

    # Aggregate results for the epoch
    # Check if any valid updates were made
    if run_acc.count.sum() > 0:
         epoch_acc_avg_per_class = run_acc.avg # Average dice per class across all val cases (ignoring NaNs within cases)
         mean_dice_epoch = np.mean(epoch_acc_avg_per_class) # Mean of the per-class averages
    else:
         print("Warning: No valid Dice scores recorded during validation epoch.")
         epoch_acc_avg_per_class = np.array([0.0, 0.0, 0.0]) # Default to zeros
         mean_dice_epoch = 0.0

    print(f"--- Validation Epoch {epoch+1} Finished ---")
    print(f"Mean Dice per class (ET, TC, WT): {epoch_acc_avg_per_class}")
    print(f"Overall Mean Dice: {mean_dice_epoch:.4f}")
    print(f"Validation Time: {(time.time() - start_time):.2f}s")
    print("------------------------------------")

    # Ensure we return a fixed-size array for history tracking, handle potential NaNs if needed
    # If epoch_acc_avg_per_class might not have 3 elements due to errors, handle it:
    if len(epoch_acc_avg_per_class) < 3:
        padded_acc = np.zeros(3)
        padded_acc[:len(epoch_acc_avg_per_class)] = epoch_acc_avg_per_class
        epoch_acc_avg_per_class = padded_acc

    return mean_dice_epoch, epoch_acc_avg_per_class


def trainer(model, train_loader_l, train_loader_u, val_loader, optimizer, scheduler,
            loss_sup, loss_cons, cons_weight, acc_func, model_inferer, post_pred_val,
            start_epoch=0):
    val_acc_max = 0.0
    history = {"train_loss": [], "val_mean_dice": [], "val_dice_et": [], "val_dice_tc": [], "val_dice_wt": []}
    epochs_ran = [] # Track epochs actually run for plotting

    print(f"--- Starting Training ---")
    for epoch in range(start_epoch, args.max_epochs):
        epochs_ran.append(epoch + 1)
        print(f"Epoch {epoch+1}/{args.max_epochs}")

        train_loss = train_epoch(
            model, train_loader_l, train_loader_u, optimizer, epoch,
            loss_sup, loss_cons, cons_weight
        )
        history["train_loss"].append(train_loss)
        print(f"Epoch {epoch+1} Avg Training Loss: {train_loss:.4f}")

        # Validation
        if (epoch + 1) % args.val_every == 0 or epoch == args.max_epochs - 1:
            val_mean_dice, val_dice_per_class = val_epoch(
                model, val_loader, epoch, acc_func, model_inferer, post_pred_val
            )
            history["val_mean_dice"].append(val_mean_dice)
            # Assuming order is ET, TC, WT from ConvertToMultiChannelBasedOnBratsClassesd
            # Ensure val_dice_per_class has 3 elements before indexing
            history["val_dice_et"].append(val_dice_per_class[0] if len(val_dice_per_class) > 0 else 0.0)
            history["val_dice_tc"].append(val_dice_per_class[1] if len(val_dice_per_class) > 1 else 0.0)
            history["val_dice_wt"].append(val_dice_per_class[2] if len(val_dice_per_class) > 2 else 0.0)

            if val_mean_dice > val_acc_max:
                print(f"New best validation Dice: {val_mean_dice:.4f} (Previous max: {val_acc_max:.4f})")
                val_acc_max = val_mean_dice
                save_checkpoint(model, epoch + 1, optimizer, filename="model_best.pt", best_acc=val_acc_max)
            else:
                 print(f"Validation Dice: {val_mean_dice:.4f} (Best: {val_acc_max:.4f})")

            # Save latest model checkpoint periodically
            save_checkpoint(model, epoch + 1, optimizer, filename="model_latest.pt", best_acc=val_acc_max)

        # Step the scheduler after optimizer.step() and validation
        scheduler.step()

    print(f"--- Training Finished ---")
    print(f"Best Validation Mean Dice: {val_acc_max:.4f}")

    # Save history
    history_path = os.path.join(args.output_dir, "training_history.json")
    # Convert numpy arrays to lists for JSON serialization
    for key in history:
        if isinstance(history[key], np.ndarray):
            history[key] = history[key].tolist()
        elif isinstance(history[key], list) and len(history[key]) > 0 and isinstance(history[key][0], np.floating):
             history[key] = [float(x) for x in history[key]] # Convert numpy floats

    try:
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Training history saved to {history_path}")
    except Exception as e:
        print(f"Error saving training history: {e}")


    return val_acc_max, history, epochs_ran # Return epochs_ran for plotting


# --- Main Execution ---
if __name__ == "__main__":
    # Define post-processing for validation (sigmoid + threshold)
    post_pred_val = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

    # Start training
    best_acc, history, epochs_ran = trainer(
        model=model,
        train_loader_l=train_loader_labeled,
        train_loader_u=train_loader_unlabeled,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_sup=supervised_loss,
        loss_cons=consistency_loss,
        cons_weight=args.consistency_weight,
        acc_func=dice_metric,
        model_inferer=model_inferer,
        post_pred_val=post_pred_val,
        start_epoch=0 # Load checkpoint here if resuming
    )

    # --- Plotting (Optional) ---
    if history and epochs_ran: # Check if history and epochs_ran are available
        print("Plotting training history...")
        plt.figure("Training History", figsize=(12, 6))

        # Plot Loss
        plt.subplot(1, 2, 1)
        plt.title("Epoch Average Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        # Ensure train_loss has same length as epochs_ran
        if len(history["train_loss"]) == len(epochs_ran):
            plt.plot(epochs_ran, history["train_loss"], color="red", label="Training Loss")
        else:
            print(f"Warning: Mismatch between epochs run ({len(epochs_ran)}) and training loss entries ({len(history['train_loss'])}). Plotting might be incorrect.")
            # Attempt to plot anyway or handle differently
            plt.plot(history["train_loss"], color="red", label="Training Loss (Index based)")

        plt.grid(True)
        plt.legend()

        # Plot Validation Dice
        # Calculate epochs where validation was run
        val_epochs_plot = [e for e in epochs_ran if e % args.val_every == 0 or e == args.max_epochs]
        if not val_epochs_plot and args.max_epochs in epochs_ran: # Handle case if only last epoch validated
             val_epochs_plot = [args.max_epochs]

        # Ensure validation history lists are not empty and lengths match expected validation runs
        num_val_runs = len(val_epochs_plot)
        if len(history["val_mean_dice"]) == num_val_runs:
            plt.subplot(1, 2, 2)
            plt.title("Validation Mean Dice")
            plt.xlabel("Epoch")
            plt.ylabel("Dice Score")
            plt.plot(val_epochs_plot, history["val_mean_dice"], color="green", marker='o', label="Mean Dice")
            plt.plot(val_epochs_plot, history["val_dice_et"], color="blue", linestyle="--", marker='x', label="Dice ET")
            plt.plot(val_epochs_plot, history["val_dice_tc"], color="orange", linestyle="--", marker='s', label="Dice TC")
            plt.plot(val_epochs_plot, history["val_dice_wt"], color="purple", linestyle="--", marker='^', label="Dice WT")
            plt.grid(True)
            plt.legend()
        else:
             print(f"Warning: Mismatch between validation epochs ({len(val_epochs_plot)}) and validation metric entries ({len(history['val_mean_dice'])}). Skipping validation plot.")


        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, "training_history.png")
        try:
            plt.savefig(plot_path)
            print(f"Training plot saved to {plot_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        # plt.show() # Uncomment to display plot directly
    else:
        print("Skipping plotting due to missing history or epoch tracking data.")


    # --- Add Inference & Attention Rollout Section (Placeholder) ---
    # This would typically be done after training, possibly in the notebook
    print("\n--- Inference and Attention Rollout (Placeholder) ---")
    # 1. Load the best model checkpoint
    # best_model_path = os.path.join(args.output_dir, "model_best.pt")
    # if os.path.exists(best_model_path):
    #    model.load_state_dict(torch.load(best_model_path)["state_dict"])
    #    model.eval()
    #    print("Loaded best model for inference.")
    # else:
    #    print("Best model checkpoint not found.")
    # 2. Select a case (e.g., from validation set)
    # 3. Perform inference using model_inferer
    # 4. --- !!! IMPLEMENTATION REQUIRED !!! ---
    #    Call your Attention Rollout function:
    #    attention_maps = generate_attention_maps(model, sample_image_tensor)
    #    Visualize the maps overlaid on the input image.
    # --- End Placeholder ---

