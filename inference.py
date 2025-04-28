# -*- coding: utf-8 -*-
"""
Runs inference using a trained Swin UNETR model for BraTS 2023 segmentation.

Loads a checkpoint, processes specified input cases, performs sliding window inference,
converts the output to BraTS label format (1, 2, 4), and saves the segmentation maps.
"""

# --- Essential Imports ---
import os
import argparse
import warnings
import sys
from functools import partial
import gc
import logging
import time

import numpy as np
import nibabel as nib
import pandas as pd
import torch
from torch.utils.data import DataLoader

# MONAI Imports
from monai.config import print_config
from monai.data import Dataset, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai import transforms
from monai.transforms import (
    AsDiscrete,
    Activations,
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    NormalizeIntensityd,
    # Add other necessary inference transforms like Orientationd, Spacingd if used in training/validation
    # Orientationd,
    # Spacingd,
)
# For saving NIfTI files
from monai.data import NibabelSaver

# --- Logging Setup ---
def setup_logging(log_file_path):
    """Configures logging to file and console."""
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    # File Handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)
    return root_logger

# --- Configuration & Constants ---
parser = argparse.ArgumentParser(description="BraTS 2023 Swin UNETR Inference")
parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint (.pt file)')
parser.add_argument('--data_dir', type=str, required=True, help='Directory containing BraTS data folder and structure CSV')
parser.add_argument('--structure_csv', type=str, default='brats_dsc.csv', help='Filename of the CSV listing dataset structure')
parser.add_argument('--output_dir', type=str, default='./output_inference', help='Directory to save predicted segmentation maps')
parser.add_argument('--case_ids', type=str, nargs='*', default=None, help='List of specific Case IDs (e.g., BraTS-GLI-00000-000) to process. If None, processes all found cases.')
parser.add_argument('--roi_size', type=int, nargs=3, default=[128, 128, 128], help='Input ROI size used during training')
parser.add_argument('--sw_batch_size', type=int, default=4, help='Sliding window batch size for inference')
parser.add_argument('--infer_overlap', type=float, default=0.6, help='Sliding window inference overlap')
parser.add_argument('--log_file', type=str, default='inference.log', help='Name for the log file')
parser.add_argument('--save_probs', action='store_true', help='Save probability maps instead of discrete segmentations')

# Check if running in notebook or script environment for arg parsing
if 'ipykernel' in sys.modules:
    # Provide default args for notebook execution (modify as needed)
    default_args = [
        '--model_path', './output_brats_ssl_amp/model_best.pt', # Example path
        '--data_dir', '/home/users/vraja/dl/data/brats2021challenge',
        '--output_dir', './output_inference_test'
        # Add --case_ids 'ID1' 'ID2' here if needed for testing
    ]
    args = parser.parse_args(default_args)
    print("Running in notebook environment, using default args.")
else:
    args = parser.parse_args()
    print("Running in script environment, parsing command-line args.")

# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# Initialize Logging
log_file = os.path.join(args.output_dir, args.log_file)
logger = setup_logging(log_file)
logger.info("--- Starting BraTS Inference Script ---")
logger.info(f"Command line args: {args}")

# --- Data Finding Function ---
def find_brats_cases(data_dir, structure_csv_filename, specific_case_ids=None):
    """Finds file paths for specified or all available BraTS cases."""
    structure_csv_path = os.path.join(data_dir, structure_csv_filename)
    if not os.path.exists(structure_csv_path):
        logger.error(f"Structure CSV file not found: {structure_csv_path}")
        raise FileNotFoundError(f"Structure CSV file not found: {structure_csv_path}")

    df = pd.read_csv(structure_csv_path)
    training_data_root_identifier = "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData" # Assuming inference on training data structure
    logger.info(f"Reading structure CSV: {structure_csv_path}")

    all_patient_dirs_df = df[(df['Is Directory'] == True) &
                             (df['Path'].str.contains(training_data_root_identifier)) &
                             (df['Path'].str.contains('BraTS-GLI-'))]
    all_patient_ids = [os.path.basename(p) for p in all_patient_dirs_df['Path'].tolist()]
    logger.info(f"Found {len(all_patient_ids)} potential patient directories in CSV.")

    target_ids = specific_case_ids if specific_case_ids else all_patient_ids
    if specific_case_ids:
        logger.info(f"Processing specific Case IDs: {target_ids}")
    else:
        logger.info(f"Processing all {len(target_ids)} found cases.")

    case_file_list = []
    missing_count = 0
    found_count = 0
    for case_id in target_ids:
        # Find the full path matching the selected ID
        matching_paths = all_patient_dirs_df[all_patient_dirs_df['Path'].str.endswith(case_id)]['Path']
        if matching_paths.empty:
             logger.warning(f"Could not find path for case ID {case_id} in DataFrame, skipping.")
             missing_count += 1
             continue
        patient_path = matching_paths.iloc[0]

        image_files = [
            os.path.join(patient_path, f"{case_id}-t1c.nii.gz"),
            os.path.join(patient_path, f"{case_id}-t1n.nii.gz"),
            os.path.join(patient_path, f"{case_id}-t2f.nii.gz"), # FLAIR
            os.path.join(patient_path, f"{case_id}-t2w.nii.gz"), # T2
        ]
        # Ground truth label is optional for inference, but check if it exists
        label_file = os.path.join(patient_path, f"{case_id}-seg.nii.gz")

        # Check if all *image* files exist
        if all(os.path.exists(f) for f in image_files):
            data_dict = {"image": image_files, "id": case_id}
            # Include label path if it exists, useful for potential evaluation later
            if os.path.exists(label_file):
                data_dict["label"] = label_file
            case_file_list.append(data_dict)
            found_count += 1
        else:
            missing_imgs = [f for f in image_files if not os.path.exists(f)]
            logger.warning(f"Missing image files for case {case_id}, skipping. Missing: {missing_imgs}")
            missing_count += 1

    logger.info(f"Found {found_count} cases with complete image modalities to process.")
    if missing_count > 0:
        logger.warning(f"Skipped {missing_count} cases due to missing files or paths.")

    if not case_file_list:
        logger.error("No valid cases found to process.")
        raise ValueError("No valid cases found to process.")

    return case_file_list

# --- Define Inference Transforms ---
# Should match validation transforms used during training
# Add Orientationd/Spacingd here if they were used in training's val_transforms
inference_transforms = Compose(
    [
        LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True),
        EnsureTyped(keys=["image"], dtype=torch.float32),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # Example: Add orientation and spacing if needed
        # Orientationd(keys=["image"], axcodes="RAS"),
        # Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
    ]
)

# --- Load Model ---
logger.info(f"Loading model checkpoint from: {args.model_path}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

if not os.path.exists(args.model_path):
    logger.error(f"Model checkpoint not found at {args.model_path}")
    sys.exit(1)

# Initialize model architecture (must match the saved checkpoint)
model = SwinUNETR(
    img_size=args.roi_size, # Provide roi_size used during training
    in_channels=4,
    out_channels=3, # Assuming 3 output channels (ET, TC, WT)
    feature_size=48, # Must match trained model's feature size
    use_checkpoint=False, # No checkpointing needed for inference
).to(device)

# Load weights safely
try:
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    model_state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(model_state_dict)
    logger.info(f"Loaded model weights from epoch {checkpoint.get('epoch', 'N/A')} (weights_only=True)")
except Exception as e_safe:
    logger.warning(f"Could not load checkpoint with weights_only=True ({e_safe}). Trying weights_only=False.")
    try:
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        model_state_dict = checkpoint.get('state_dict', checkpoint)
        model.load_state_dict(model_state_dict)
        logger.info(f"Loaded model weights from epoch {checkpoint.get('epoch', 'N/A')} (weights_only=False)")
    except Exception as e_unsafe:
         logger.error(f"Failed to load checkpoint even with weights_only=False: {e_unsafe}")
         sys.exit(1)

model.eval() # Set model to evaluation mode
logger.info("Model loaded successfully.")

# --- Prepare Inferer and Post-processing ---
model_inferer = partial(
    sliding_window_inference,
    roi_size=args.roi_size,
    sw_batch_size=args.sw_batch_size,
    predictor=model,
    overlap=args.infer_overlap,
    mode="gaussian", # Smoother blending is often preferred
    progress=True
)

# Post-processing: Apply Sigmoid, then threshold for discrete map OR keep probabilities
if args.save_probs:
    logger.info("Saving probability maps (after sigmoid).")
    post_pred = Compose([Activations(sigmoid=True)]) # Only apply sigmoid
else:
    logger.info("Saving discrete segmentation maps (sigmoid + threshold).")
    post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

# Saver for writing NIfTI files - retains original affine/header
saver = NibabelSaver(output_dir=args.output_dir, output_postfix="_pred", output_ext=".nii.gz", separate_folder=False)

# --- Get Data Files ---
logger.info("Finding cases to process...")
case_files_to_process = find_brats_cases(args.data_dir, args.structure_csv, args.case_ids)

# --- Inference Loop ---
logger.info(f"--- Starting Inference on {len(case_files_to_process)} cases ---")
inference_start_time = time.time()

for i, case_data_dict in enumerate(case_files_to_process):
    case_id = case_data_dict['id']
    case_start_time = time.time()
    logger.info(f"Processing case {i+1}/{len(case_files_to_process)}: {case_id}")

    try:
        # Load and preprocess image data
        # Create a temporary Dataset/DataLoader for this single case
        # This ensures metadata (affine, etc.) is handled correctly by LoadImaged
        case_ds = Dataset(data=[case_data_dict], transform=inference_transforms)
        # Note: DataLoader not strictly necessary for batch size 1 and num_workers=0,
        # but using it maintains consistency with potential future batching or worker use.
        case_loader = DataLoader(case_ds, batch_size=1, shuffle=False, num_workers=0)
        processed_data = next(iter(case_loader)) # Get the single processed item

        input_tensor = processed_data["image"].to(device)
        # Keep metadata for saving
        metadata = processed_data['image_meta_dict']

        # Run inference
        with torch.no_grad():
            # Use autocast for potential speedup even during inference on V100
            with autocast(device_type=device.type, enabled=torch.cuda.is_available()):
                logits = model_inferer(input_tensor)

        # Apply post-processing
        if args.save_probs:
            # Output is probability map (C, H, W, D)
            output_processed = post_pred(logits)[0] # Get first item from batch, shape (3, H, W, D)
            # Need to decide how to save multi-channel probability maps
            # Option 1: Save each channel as a separate file (e.g., _pred_et.nii.gz, _pred_tc.nii.gz, ...)
            # Option 2: Combine into a single multi-channel NIfTI (less common for viewing)
            # Option 3: Convert back to single channel map based on argmax (but loses probabilities)
            # For now, let's log a warning and skip saving if --save_probs is used without further logic
            logger.warning(f"Saving multi-channel probability maps is not fully implemented. Skipping save for {case_id}.")
            # To implement saving: loop through channels, save each with modified metadata/filename
            # for chan_idx, chan_name in enumerate(["ET", "TC", "WT"]):
            #     chan_prob_map = output_processed[chan_idx:chan_idx+1, ...] # Keep channel dim
            #     # Modify metadata if needed (e.g., update description)
            #     # Adjust saver or save manually using nibabel
            #     # saver.save_batch(chan_prob_map, metadata) # This might overwrite, need custom save logic
            continue # Skip to next case for now if saving probs

        else:
            # Output is discrete segmentation (C, H, W, D) with values 0 or 1
            output_processed = post_pred(logits)[0] # Get first item from batch, shape (3, H, W, D)

            # Convert 3-channel output (ET, TC, WT) to single-channel BraTS format (1, 2, 4)
            # Output shape needs to be (1, H, W, D) for saver
            output_brats_format = torch.zeros_like(output_processed[0:1, ...], dtype=torch.int8) # Match shape of one channel, use int8
            # Order matters: WT includes TC includes ET. Start with WT=3, then TC=2, then ET=1
            # BraTS labels: 1=NCR/NET, 2=ED, 4=ET
            # Our channels: 0=ET, 1=TC, 2=WT
            # Mapping:
            # Where WT=1 (channel 2), set output to 2 (ED)
            # Where TC=1 (channel 1), set output to 1 (NCR/NET) - Overwrites ED where TC is present
            # Where ET=1 (channel 0), set output to 4 (ET) - Overwrites NCR/NET where ET is present
            output_brats_format[output_processed[2:3, ...] == 1] = 2 # Edema
            output_brats_format[output_processed[1:2, ...] == 1] = 1 # NCR/NET (part of TC)
            output_brats_format[output_processed[0:1, ...] == 1] = 4 # ET

            # Save the single-channel BraTS format segmentation map
            saver.save_batch(output_brats_format, metadata)
            logger.info(f"Saved prediction for {case_id} to {args.output_dir}")

    except Exception as e:
        logger.exception(f"Error processing case {case_id}: {e}") # Log full traceback

    case_time = time.time() - case_start_time
    logger.info(f"Finished case {case_id} in {case_time:.2f} seconds.")
    # Optional: clean up memory
    # del input_tensor, logits, output_processed, output_brats_format, processed_data
    # gc.collect()
    # torch.cuda.empty_cache()


total_time = time.time() - inference_start_time
logger.info(f"--- Inference Finished ---")
logger.info(f"Processed {len(case_files_to_process)} cases.")
logger.info(f"Total inference time: {total_time:.2f} seconds.")
logger.info(f"Average time per case: {total_time / len(case_files_to_process):.2f} seconds.")
logger.info(f"Predictions saved in: {args.output_dir}")

