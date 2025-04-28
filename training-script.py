# -*- coding: utf-8 -*-
"""
For training a SwinUNETR model on BraTS 2023
using Semi-Supervised Learning (Consistency Regularization) and MONAI.
"""

# --- Essential Imports ---
import os
import json
import shutil
import tempfile
import time
import argparse
import warnings
import sys
from functools import partial
import gc # For explicit garbage collection if needed
import logging # <<< Import logging module

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler, ConcatDataset
# <<< Import updated AMP modules >>>
from torch.amp import GradScaler, autocast # Use torch.amp directly

# MONAI Imports
from monai.config import print_config
from monai.data import Dataset, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai import transforms
from monai.transforms import (
    AsDiscrete,
    Activations,
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    ConvertToMultiChannelBasedOnBratsClassesd,
    NormalizeIntensityd,
    CropForegroundd,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
)
from monai.utils.enums import MetricReduction

# --- Logging Setup ---
# <<< Function to set up logging >>>
def setup_logging(log_file_path):
    """Configures logging to file and console."""
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()

    # Clear existing handlers
    root_logger.handlers.clear()

    # File Handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout) # Log to stdout
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    root_logger.setLevel(logging.INFO) # Set desired logging level (e.g., INFO, DEBUG)
    return root_logger

# --- Configuration & Constants ---
parser = argparse.ArgumentParser(description="BraTS 2023 SSL SwinUNETR Training with AMP")
parser.add_argument('--data_dir', type=str, default='/home/users/vraja/dl/data/brats2021challenge', help='Directory containing BraTS data folder and structure CSV')
parser.add_argument('--structure_csv', type=str, default='brats_dsc.csv', help='Filename of the CSV listing dataset structure (e.g., brats_dsc.csv)')
parser.add_argument('--output_dir', type=str, default='./output_brats_ssl_amp', help='Directory to save models and logs')
parser.add_argument('--roi_size', type=int, nargs=3, default=[128, 128, 128], help='Input ROI size (x, y, z)')
parser.add_argument('--batch_size', type=int, default=1, help='Total batch size (labeled + unlabeled) - Set to 1 if workers=0 causes OOM') # Back to 1 for safety w/o workers
parser.add_argument('--labeled_bs_ratio', type=float, default=0.5, help='Ratio of labeled samples in a batch')
parser.add_argument('--sw_batch_size', type=int, default=4, help='Sliding window batch size for validation')
parser.add_argument('--infer_overlap', type=float, default=0.5, help='Sliding window inference overlap')
parser.add_argument('--max_epochs', type=int, default=50, help='Maximum training epochs')
parser.add_argument('--val_every', type=int, default=10, help='Run validation every N epochs')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Optimizer weight decay')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
# <<< Set workers back to 0 >>>
parser.add_argument('--num_workers', type=int, default=0, help='DataLoader workers (0 for main process)')
parser.add_argument('--labeled_ratio', type=float, default=0.4, help='Ratio of data to use as labeled (e.g., 0.4 for 40%)')
parser.add_argument('--consistency_weight', type=float, default=1.0, help='Weight for the consistency loss term')
parser.add_argument('--log_interval', type=int, default=20, help='Log training progress every N batches') # Log more often w/o workers
parser.add_argument('--log_file', type=str, default='training.log', help='Name for the log file') # <<< Added log file argument

# Check if running in notebook or script environment for arg parsing
if 'ipykernel' in sys.modules:
    args = parser.parse_args([])
    print("Running in notebook environment, using default args.") # Keep print here for initial feedback
else:
    args = parser.parse_args()
    print("Running in script environment, parsing command-line args.") # Keep print here

# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# <<< Initialize Logging >>>
log_file = os.path.join(args.output_dir, args.log_file)
logger = setup_logging(log_file)
logger.info("--- Starting BraTS SSL Training Script ---")
logger.info(f"Command line args: {args}")

# Suppress MONAI warnings if needed (can log instead)
# warnings.filterwarnings("ignore", category=UserWarning, module='monai')
# logger.info("MONAI UserWarnings suppressed.")

# Log MONAI config (optional, can be verbose)
# logger.info("MONAI Configuration:")
# try:
#     print_config(file=sys.stderr) # Print to stderr to separate from logger
# except Exception as e:
#     logger.warning(f"Could not print MONAI config: {e}")


# Set random seed for reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
logger.info(f"Random seed set to {args.seed}")

# --- Helper Classes & Functions ---
# (AverageMeter remains the same)
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
    """
    structure_csv_path = os.path.join(data_dir, structure_csv_filename)
    if not os.path.exists(structure_csv_path):
        logger.error(f"Structure CSV file not found: {structure_csv_path}") # Use logger
        raise FileNotFoundError(f"Structure CSV file not found: {structure_csv_path}")

    df = pd.read_csv(structure_csv_path)
    training_data_root_identifier = "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    logger.info(f"Using structure CSV: {structure_csv_path}")
    logger.info(f"Identifying patient folders within paths containing: {training_data_root_identifier}")

    all_files = []
    patient_dirs = df[(df['Is Directory'] == True) &
                      (df['Path'].str.contains(training_data_root_identifier)) &
                      (df['Path'].str.contains('BraTS-GLI-'))]['Path'].tolist()

    logger.info(f"Found {len(patient_dirs)} potential patient directories.")

    missing_count = 0
    for patient_path in patient_dirs:
        patient_folder_name = os.path.basename(patient_path)
        image_files = [
            os.path.join(patient_path, f"{patient_folder_name}-t1c.nii.gz"),
            os.path.join(patient_path, f"{patient_folder_name}-t1n.nii.gz"),
            os.path.join(patient_path, f"{patient_folder_name}-t2f.nii.gz"), # FLAIR
            os.path.join(patient_path, f"{patient_folder_name}-t2w.nii.gz"), # T2
        ]
        label_file = os.path.join(patient_path, f"{patient_folder_name}-seg.nii.gz")

        if all(os.path.exists(f) for f in image_files) and os.path.exists(label_file):
            all_files.append({"image": image_files, "label": label_file})
        else:
            missing = [f for f in image_files + [label_file] if not os.path.exists(f)]
            missing_count += 1
            # Reduce warning verbosity - maybe log only first few or total count later
            if missing_count < 10:
                 logger.warning(f"Missing files for patient {patient_folder_name}, skipping. Missing: {missing}")
            elif missing_count == 10:
                 logger.warning("... further missing file warnings suppressed.")


    if not all_files:
        logger.error("No valid data files found after checking paths.")
        raise ValueError("No valid data files found.")
    logger.info(f"Found {len(all_files)} cases with all required files. Skipped {missing_count} cases due to missing files.")

    np.random.shuffle(all_files)
    num_labeled = int(len(all_files) * labeled_ratio)
    labeled_files = all_files[:num_labeled]
    unlabeled_files = all_files[num_labeled:]

    logger.info(f"Total valid cases processed: {len(all_files)}")
    logger.info(f"Using {len(labeled_files)} cases as labeled data.")
    logger.info(f"Using {len(unlabeled_files)} cases as unlabeled data.")

    return labeled_files, unlabeled_files


def save_checkpoint(model, epoch, optimizer, scaler, filename="model.pt", best_acc=0.0, dir_add=args.output_dir):
    """Saves model checkpoint, including optimizer and scaler state."""
    os.makedirs(dir_add, exist_ok=True)
    state_dict = model.state_dict()
    save_dict = {
        "epoch": epoch,
        "best_acc": best_acc,
        "state_dict": state_dict,
    }
    if optimizer is not None:
         save_dict["optimizer"] = optimizer.state_dict()
    if scaler is not None:
        save_dict["scaler"] = scaler.state_dict()

    filename = os.path.join(dir_add, filename)
    try:
        torch.save(save_dict, filename)
        logger.info(f"Saving checkpoint {filename}") # Use logger
    except Exception as e:
        logger.error(f"Error saving checkpoint {filename}: {e}") # Use logger


# --- Define Transforms ---
# (Transforms remain the same)
train_transforms_weak = Compose(
    [
        LoadImaged(keys=["image", "label"], image_only=False, ensure_channel_first=True),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image", k_divisible=args.roi_size, allow_smaller=True), # allow_smaller=True might be needed
        RandSpatialCropd(keys=["image", "label"], roi_size=args.roi_size, random_size=False),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.1, max_k=3),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
    ]
)
train_transforms_strong = Compose(
    [
        LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True),
        EnsureTyped(keys=["image"], dtype=torch.float32),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image"], source_key="image", k_divisible=args.roi_size, allow_smaller=True), # allow_smaller=True might be needed
        RandSpatialCropd(keys=["image"], roi_size=args.roi_size, random_size=False),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image"], prob=0.1, max_k=3),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
        RandGaussianNoised(keys="image", prob=0.3, mean=0.0, std=0.1),
        RandGaussianSmoothd(keys="image", prob=0.3, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
        RandScaleIntensityd(keys="image", factors=0.2, prob=0.2),
        RandShiftIntensityd(keys="image", offsets=0.2, prob=0.2),
    ]
)
val_transforms = Compose(
    [
        LoadImaged(keys=["image", "label"], image_only=False, ensure_channel_first=True),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)

# --- Custom Dataset for SSL ---
class SSLDataset(Dataset):
    def __init__(self, data, transform_weak, transform_strong):
        super().__init__(data=data, transform=None)
        self.transform_weak = transform_weak
        self.transform_strong = transform_strong

    def __getitem__(self, index):
        data_i = self.data[index]
        try: # Add error handling within __getitem__
            data_weak = self.transform_weak(data_i.copy())
            img_weak = data_weak['image']
            label_weak = data_weak.get('label', None)
            data_strong_input = {'image': data_i['image']}
            img_strong = self.transform_strong(data_strong_input)['image']
            output = {"image_weak": img_weak, "image_strong": img_strong}
            if label_weak is not None:
                output["label"] = label_weak
            return output
        except Exception as e:
            # Log error instead of printing directly
            logger.error(f"Error loading/transforming data at index {index}, path: {data_i.get('image', 'N/A')}. Error: {e}", exc_info=True) # Log traceback
            return None

# --- Custom Collate Function ---
def safe_collate(batch):
    """Collate function that filters out None values."""
    original_size = len(batch)
    batch = list(filter(lambda x: x is not None, batch))
    filtered_count = original_size - len(batch)
    if filtered_count > 0:
        logger.warning(f"Removed {filtered_count} None items from batch")
    if not batch:
        return None
    return list_data_collate(batch)


# --- Create Datasets and DataLoaders ---
logger.info("Creating datasets and dataloaders...")
labeled_files, unlabeled_files = get_brats2023_datalists(
    args.data_dir, args.structure_csv, args.labeled_ratio
)

if len(labeled_files) < 5:
    logger.error(f"Only {len(labeled_files)} labeled files found. Need more for train/validation split.")
    raise ValueError(f"Only {len(labeled_files)} labeled files found. Need more for train/validation split.")
val_split_idx = max(1, int(len(labeled_files) * 0.8))
train_labeled_files = labeled_files[:val_split_idx]
val_files = labeled_files[val_split_idx:]

logger.info(f"Using {len(train_labeled_files)} cases for labeled training.")
logger.info(f"Using {len(val_files)} cases for validation.")

train_ds_labeled = SSLDataset(data=train_labeled_files, transform_weak=train_transforms_weak, transform_strong=train_transforms_strong)
train_ds_unlabeled = SSLDataset(data=unlabeled_files, transform_weak=train_transforms_weak, transform_strong=train_transforms_strong)
val_ds = Dataset(data=val_files, transform=val_transforms)

labeled_batch_size = int(args.batch_size * args.labeled_bs_ratio)
unlabeled_batch_size = args.batch_size - labeled_batch_size

if not train_labeled_files: labeled_batch_size = 0; logger.warning("No labeled training files available.")
elif labeled_batch_size == 0 and unlabeled_batch_size > 0: labeled_batch_size = 1; unlabeled_batch_size = max(0, args.batch_size - 1); logger.warning("Labeled batch size was 0, adjusting to 1.")
if not train_ds_unlabeled: unlabeled_batch_size = 0; logger.warning("No unlabeled training files available.")
elif unlabeled_batch_size == 0 and labeled_batch_size > 0: unlabeled_batch_size = 1; labeled_batch_size = max(0, args.batch_size - 1); logger.warning("Unlabeled batch size was 0, adjusting to 1.")
if labeled_batch_size == 0 and unlabeled_batch_size == 0:
    logger.error("Both labeled and unlabeled batch sizes are zero. No data to train.")
    raise ValueError("Both labeled and unlabeled batch sizes are zero.")

logger.info(f"Effective Batch size: {labeled_batch_size + unlabeled_batch_size} (Labeled: {labeled_batch_size}, Unlabeled: {unlabeled_batch_size})")

pin_memory_flag = False
persistent_workers_flag = False

train_loader_labeled = DataLoader(
    train_ds_labeled, batch_size=labeled_batch_size, shuffle=True,
    num_workers=args.num_workers, pin_memory=pin_memory_flag, collate_fn=safe_collate,
    drop_last=True, persistent_workers=persistent_workers_flag
) if labeled_batch_size > 0 else None

train_loader_unlabeled = DataLoader(
    train_ds_unlabeled, batch_size=unlabeled_batch_size, shuffle=True,
    num_workers=args.num_workers, pin_memory=pin_memory_flag, collate_fn=safe_collate,
    drop_last=True, persistent_workers=persistent_workers_flag
) if unlabeled_batch_size > 0 else None

val_loader = DataLoader(
    val_ds, batch_size=1, shuffle=False,
    num_workers=args.num_workers, pin_memory=pin_memory_flag, persistent_workers=persistent_workers_flag
)

# --- Model, Loss, Optimizer, Scheduler ---
logger.info("Initializing model, loss, optimizer...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

model = SwinUNETR(
    img_size=args.roi_size,
    in_channels=4,
    out_channels=3,
    feature_size=48,
    use_checkpoint=False, # Disabled gradient checkpointing
).to(device)
logger.info("SwinUNETR model initialized.")

supervised_loss = DiceCELoss(to_onehot_y=False, sigmoid=True, lambda_dice=0.5, lambda_ce=0.5)
consistency_loss = torch.nn.MSELoss()
post_sigmoid = Activations(sigmoid=True)
post_pred = AsDiscrete(threshold=0.5)
dice_metric = DiceMetric(include_background=False, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)

scaler = GradScaler(device='cuda') if device.type == 'cuda' else None
if scaler is None:
    logger.warning("CUDA not available, AMP disabled.")
else:
    logger.info("AMP Enabled with GradScaler.")


model_inferer = partial(
    sliding_window_inference,
    roi_size=args.roi_size,
    sw_batch_size=args.sw_batch_size,
    predictor=model,
    overlap=args.infer_overlap,
    mode="gaussian",
    progress=True
)

# --- Training and Validation Functions ---

def train_epoch(model, labeled_loader, unlabeled_loader, optimizer, scaler, epoch, sup_loss_fn, cons_loss_fn, cons_weight):
    model.train()
    run_loss_sup = AverageMeter()
    run_loss_cons = AverageMeter()
    run_loss_total = AverageMeter()
    epoch_start_time = time.time()
    batch_start_time = time.time()

    len_l = len(labeled_loader) if labeled_loader else 0
    len_u = len(unlabeled_loader) if unlabeled_loader else 0
    steps = max(len_l, len_u)
    if steps == 0: logger.warning(f"No batches to process in Epoch {epoch+1}. Skipping."); return 0.0

    iter_l = iter(labeled_loader) if labeled_loader else None
    iter_u = iter(unlabeled_loader) if unlabeled_loader else None

    logger.info(f"Starting Epoch {epoch+1}/{args.max_epochs}, Steps: {steps}")

    for i in range(steps):
        optimizer.zero_grad()
        total_loss_batch = torch.tensor(0.0, device=device)
        loss_s_val = 0.0
        loss_c_val = 0.0
        current_labeled_bs = 0
        current_unlabeled_bs = 0
        use_amp = scaler is not None

        with autocast(device_type=device.type, enabled=use_amp):
            # --- Supervised Loss ---
            if iter_l:
                try:
                    batch_labeled = next(iter_l)
                    if batch_labeled is None: logger.warning(f"Skipping None labeled batch at step {i}"); continue
                    images_l_weak, labels_l = batch_labeled["image_weak"].to(device), batch_labeled["label"].to(device)
                    current_labeled_bs = images_l_weak.size(0)
                    logits_l = model(images_l_weak)
                    loss_s = sup_loss_fn(logits_l, labels_l)
                    loss_s_val = loss_s.item()
                    run_loss_sup.update(loss_s_val, n=current_labeled_bs)
                    total_loss_batch += loss_s
                except StopIteration:
                    iter_l = iter(labeled_loader); batch_labeled = next(iter_l)
                    if batch_labeled is None: logger.warning(f"Skipping None labeled batch at step {i} after reset"); continue
                    images_l_weak, labels_l = batch_labeled["image_weak"].to(device), batch_labeled["label"].to(device)
                    current_labeled_bs = images_l_weak.size(0)
                    logits_l = model(images_l_weak); loss_s = sup_loss_fn(logits_l, labels_l)
                    loss_s_val = loss_s.item(); run_loss_sup.update(loss_s_val, n=current_labeled_bs); total_loss_batch += loss_s
                except Exception as e: logger.exception(f"Error processing labeled batch {i}: {e}"); continue # Log full traceback

            # --- Consistency Loss ---
            if iter_u:
                try:
                    batch_unlabeled = next(iter_u)
                    if batch_unlabeled is None: logger.warning(f"Skipping None unlabeled batch at step {i}"); continue
                    images_u_weak, images_u_strong = batch_unlabeled["image_weak"].to(device), batch_unlabeled["image_strong"].to(device)
                    current_unlabeled_bs = images_u_weak.size(0)
                    with torch.no_grad():
                        with autocast(device_type=device.type, enabled=use_amp):
                             logits_u_weak = model(images_u_weak)
                        pseudo_labels = torch.sigmoid(logits_u_weak.detach())
                    logits_u_strong = model(images_u_strong)
                    preds_strong_sig = torch.sigmoid(logits_u_strong)
                    loss_c = cons_loss_fn(preds_strong_sig, pseudo_labels)
                    loss_c_val = loss_c.item()
                    run_loss_cons.update(loss_c_val, n=current_unlabeled_bs)
                    total_loss_batch += cons_weight * loss_c
                except StopIteration:
                    iter_u = iter(unlabeled_loader); batch_unlabeled = next(iter_u)
                    if batch_unlabeled is None: logger.warning(f"Skipping None unlabeled batch at step {i} after reset"); continue
                    images_u_weak, images_u_strong = batch_unlabeled["image_weak"].to(device), batch_unlabeled["image_strong"].to(device)
                    current_unlabeled_bs = images_u_weak.size(0)
                    with torch.no_grad():
                         with autocast(device_type=device.type, enabled=use_amp):
                              logits_u_weak = model(images_u_weak)
                         pseudo_labels = torch.sigmoid(logits_u_weak.detach())
                    logits_u_strong = model(images_u_strong); preds_strong_sig = torch.sigmoid(logits_u_strong)
                    loss_c = cons_loss_fn(preds_strong_sig, pseudo_labels)
                    loss_c_val = loss_c.item(); run_loss_cons.update(loss_c_val, n=current_unlabeled_bs); total_loss_batch += cons_weight * loss_c
                except Exception as e: logger.exception(f"Error processing unlabeled batch {i}: {e}"); continue # Log full traceback

        # --- Scale loss and step ---
        if total_loss_batch.item() > 0:
            try:
                if use_amp:
                    scaler.scale(total_loss_batch).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_loss_batch.backward()
                    optimizer.step()
                total_bs = current_labeled_bs + current_unlabeled_bs
                if total_bs > 0:
                    run_loss_total.update(total_loss_batch.item(), n=total_bs)
            except RuntimeError as e:
                logger.exception(f"RuntimeError during backward/step at batch {i}: {e}") # Log full traceback
                # gc.collect()
                # torch.cuda.empty_cache()
                continue

            # --- Logging ---
            if (i + 1) % args.log_interval == 0 or (i + 1) == steps:
                batch_time = time.time() - batch_start_time
                mem_alloc_gb = torch.cuda.memory_allocated(device) / (1024**3) if device.type == 'cuda' else 0
                mem_res_gb = torch.cuda.memory_reserved(device) / (1024**3) if device.type == 'cuda' else 0
                logger.info( # Use logger
                    f"Epoch {epoch+1}/{args.max_epochs} Batch {i+1}/{steps} | "
                    f"Time/batch: {batch_time/args.log_interval:.3f}s | "
                    f"Loss Total (Batch): {total_loss_batch.item():.4f} | "
                    f"Loss Sup (Batch): {loss_s_val:.4f} | Loss Cons (Batch): {loss_c_val:.4f} | "
                    f"Avg Loss Total: {run_loss_total.avg:.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                    f"Mem Alloc: {mem_alloc_gb:.2f}G | Mem Res: {mem_res_gb:.2f}G"
                )
                batch_start_time = time.time()

    logger.info(f"--- Epoch {epoch+1} Training Finished ---")
    logger.info(f"Epoch {epoch+1} Avg Training Loss: {run_loss_total.avg:.4f}")
    logger.info(f"Epoch {epoch+1} Time: {(time.time() - epoch_start_time):.2f}s")
    return run_loss_total.avg if run_loss_total.count > 0 else 0.0


def val_epoch(model, loader, epoch, acc_func, model_inferer, post_pred_val):
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()
    logger.info(f"--- Starting Validation Epoch {epoch+1} ---")
    use_amp = device.type == 'cuda'

    with torch.no_grad():
        for i, batch_data in enumerate(loader):
            if batch_data is None: logger.warning(f"Skipping None validation batch {i}"); continue
            if "image" not in batch_data or "label" not in batch_data:
                logger.warning(f"Skipping validation batch {i} due to missing keys.")
                continue
            val_images, val_labels = batch_data["image"].to(device), batch_data["label"].to(device)
            try:
                with autocast(device_type=device.type, enabled=use_amp):
                    val_logits = model_inferer(val_images)
                val_outputs = post_pred_val(val_logits)
                val_outputs_list = decollate_batch(val_outputs)
                val_labels_list = decollate_batch(val_labels)
                acc_func.reset()
                acc_func(y_pred=val_outputs_list, y=val_labels_list)
                acc, not_nans = acc_func.aggregate()
                valid_acc = acc[not_nans].cpu().numpy()
                valid_n = not_nans.cpu().numpy().astype(int) if not_nans.numel() > 0 else np.array([], dtype=int)
                if np.sum(valid_n) > 0:
                     if valid_acc.ndim == 0 and valid_n.ndim > 0 and np.sum(valid_n) == 1: run_acc.update(valid_acc.item(), n=1)
                     elif valid_acc.shape == valid_n.shape : run_acc.update(valid_acc, n=valid_n)
                     else: logger.warning(f"Shape mismatch in validation metrics. acc: {valid_acc.shape}, n: {valid_n.shape}")
                # Optional: Log dice per case if needed for debugging
                # dice_scores = acc.cpu().numpy()
                # mean_dice_case = np.nanmean(dice_scores)
                # logger.debug(f"Val Case {i+1}/{len(loader)} | Dice Per Class: {dice_scores} | Avg Dice: {mean_dice_case:.4f}")
            except Exception as e: logger.exception(f"Error during validation case {i+1}: {e}"); continue # Log full traceback

    if run_acc.count.sum() > 0:
         if isinstance(run_acc.avg, (np.ndarray, list)) and len(run_acc.avg) > 0 : epoch_acc_avg_per_class = np.nan_to_num(run_acc.avg); mean_dice_epoch = np.mean(epoch_acc_avg_per_class)
         else: epoch_acc_avg_per_class = np.array([run_acc.avg] * 3) if np.isscalar(run_acc.avg) else np.zeros(3); mean_dice_epoch = run_acc.avg if np.isscalar(run_acc.avg) else 0.0
    else: logger.warning("No valid Dice scores recorded during validation epoch."); epoch_acc_avg_per_class = np.array([0.0, 0.0, 0.0]); mean_dice_epoch = 0.0

    logger.info(f"--- Validation Epoch {epoch+1} Finished ---")
    logger.info(f"Mean Dice per class (ET, TC, WT): {epoch_acc_avg_per_class}")
    logger.info(f"Overall Mean Dice: {mean_dice_epoch:.4f}")
    logger.info(f"Validation Time: {(time.time() - start_time):.2f}s")
    logger.info("------------------------------------")

    if len(epoch_acc_avg_per_class) < 3:
        padded_acc = np.zeros(3); padded_acc[:len(epoch_acc_avg_per_class)] = epoch_acc_avg_per_class; epoch_acc_avg_per_class = padded_acc
    return mean_dice_epoch, epoch_acc_avg_per_class


def trainer(model, train_loader_l, train_loader_u, val_loader, optimizer, scaler, scheduler,
            loss_sup, loss_cons, cons_weight, acc_func, model_inferer, post_pred_val,
            start_epoch=0):
    val_acc_max = 0.0
    history = {"train_loss": [], "val_mean_dice": [], "val_dice_et": [], "val_dice_tc": [], "val_dice_wt": []}
    epochs_ran = []

    logger.info(f"--- Starting Training (AMP {'Enabled' if scaler else 'Disabled'}, Workers: {args.num_workers}) ---")
    for epoch in range(start_epoch, args.max_epochs):
        epochs_ran.append(epoch + 1)
        # logger.info(f"Epoch {epoch+1}/{args.max_epochs}") # Logged inside train_epoch

        train_loss = train_epoch(
            model, train_loader_l, train_loader_u, optimizer, scaler, epoch,
            loss_sup, loss_cons, cons_weight
        )
        history["train_loss"].append(train_loss)
        # Avg loss already logged in train_epoch end

        if (epoch + 1) % args.val_every == 0 or epoch == args.max_epochs - 1:
            if val_loader and len(val_loader) > 0:
                val_mean_dice, val_dice_per_class = val_epoch(
                    model, val_loader, epoch, acc_func, model_inferer, post_pred_val
                )
                history["val_mean_dice"].append(val_mean_dice)
                history["val_dice_et"].append(val_dice_per_class[0] if len(val_dice_per_class) > 0 else 0.0)
                history["val_dice_tc"].append(val_dice_per_class[1] if len(val_dice_per_class) > 1 else 0.0)
                history["val_dice_wt"].append(val_dice_per_class[2] if len(val_dice_per_class) > 2 else 0.0)

                if val_mean_dice > val_acc_max:
                    logger.info(f"New best validation Dice: {val_mean_dice:.4f} (Previous max: {val_acc_max:.4f})")
                    val_acc_max = val_mean_dice
                    save_checkpoint(model, epoch + 1, optimizer, scaler, filename="model_best.pt", best_acc=val_acc_max)
                else:
                     logger.info(f"Validation Dice: {val_mean_dice:.4f} (Best: {val_acc_max:.4f})")
                save_checkpoint(model, epoch + 1, optimizer, scaler, filename="model_latest.pt", best_acc=val_acc_max)
            else:
                logger.warning(f"Skipping validation for epoch {epoch+1} as validation loader is empty.")
                history["val_mean_dice"].append(0.0); history["val_dice_et"].append(0.0); history["val_dice_tc"].append(0.0); history["val_dice_wt"].append(0.0)

        scheduler.step()
        # Optional: Explicit garbage collection at end of epoch
        # gc.collect()
        # torch.cuda.empty_cache()

    logger.info(f"--- Training Finished ---")
    logger.info(f"Best Validation Mean Dice: {val_acc_max:.4f}")

    history_path = os.path.join(args.output_dir, "training_history.json")
    for key in history:
        if isinstance(history[key], np.ndarray): history[key] = history[key].tolist()
        elif isinstance(history[key], list) and len(history[key]) > 0 and isinstance(history[key][0], np.floating): history[key] = [float(x) for x in history[key]]
    try:
        with open(history_path, 'w') as f: json.dump(history, f, indent=4)
        logger.info(f"Training history saved to {history_path}")
    except Exception as e: logger.error(f"Error saving training history: {e}")

    return val_acc_max, history, epochs_ran


# --- Main Execution ---
if __name__ == "__main__":
    try: # <<< Add top-level try...except block >>>
        post_pred_val = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

        best_acc, history, epochs_ran = trainer(
            model=model,
            train_loader_l=train_loader_labeled,
            train_loader_u=train_loader_unlabeled,
            val_loader=val_loader,
            optimizer=optimizer,
            scaler=scaler, # Pass scaler
            scheduler=scheduler,
            loss_sup=supervised_loss,
            loss_cons=consistency_loss,
            cons_weight=args.consistency_weight,
            acc_func=dice_metric,
            model_inferer=model_inferer,
            post_pred_val=post_pred_val,
            start_epoch=0
        )

        # --- Plotting (Optional) ---
        if history and epochs_ran:
            logger.info("Plotting training history...")
            can_plot = True
            try:
                import matplotlib.pyplot as plt
                if not hasattr(sys, 'ps1'): import matplotlib; matplotlib.use('Agg')
                plt.figure("Training History", figsize=(12, 6))
            except ImportError: logger.warning("Matplotlib not found. Skipping plotting."); can_plot = False
            except Exception as e: logger.warning(f"Error initializing plot: {e}. Skipping plotting."); can_plot = False

            if can_plot:
                plt.subplot(1, 2, 1)
                plt.title("Epoch Average Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
                if len(history["train_loss"]) == len(epochs_ran): plt.plot(epochs_ran, history["train_loss"], color="red", label="Training Loss")
                else: logger.warning(f"Mismatch epochs run/train loss entries. Plotting might be incorrect."); plt.plot(history["train_loss"], color="red", label="Training Loss (Index based)")
                plt.grid(True); plt.legend()

                val_epochs_plot = [e for e in epochs_ran if e % args.val_every == 0 or e == args.max_epochs]
                if not val_epochs_plot and args.max_epochs in epochs_ran: val_epochs_plot = [args.max_epochs]
                num_val_runs = len(val_epochs_plot)
                if history.get("val_mean_dice") and len(history["val_mean_dice"]) == num_val_runs:
                    plt.subplot(1, 2, 2)
                    plt.title("Validation Mean Dice"); plt.xlabel("Epoch"); plt.ylabel("Dice Score")
                    plt.plot(val_epochs_plot, history["val_mean_dice"], color="green", marker='o', label="Mean Dice")
                    plt.plot(val_epochs_plot, history["val_dice_et"], color="blue", linestyle="--", marker='x', label="Dice ET")
                    plt.plot(val_epochs_plot, history["val_dice_tc"], color="orange", linestyle="--", marker='s', label="Dice TC")
                    plt.plot(val_epochs_plot, history["val_dice_wt"], color="purple", linestyle="--", marker='^', label="Dice WT")
                    plt.ylim(0, 1); plt.grid(True); plt.legend()
                else: logger.warning(f"Validation metric entries mismatch expected runs. Skipping validation plot.")

                plt.tight_layout()
                plot_path = os.path.join(args.output_dir, "training_history.png")
                try: plt.savefig(plot_path); logger.info(f"Training plot saved to {plot_path}")
                except Exception as e: logger.error(f"Error saving plot: {e}")
                # plt.show()
        else: logger.info("Skipping plotting due to missing history or epoch tracking data.")

        # --- Inference & Attention Rollout Placeholder ---
        logger.info("\n--- Inference and Attention Rollout (Placeholder) ---")
        # (Placeholder code remains the same)

    except Exception as main_exception: # <<< Catch any unexpected errors >>>
        logger.exception("An uncaught exception occurred during script execution.")
        sys.exit(1) # Exit with a non-zero code to indicate failure

    logger.info("--- Script Finished Successfully ---")

