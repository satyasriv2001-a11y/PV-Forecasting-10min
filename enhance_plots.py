#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Enhance Existing Plots - Make Text More Readable

This script enhances existing PNG plot images by:
1. Upscaling images (making them larger so text appears bigger)
2. Applying sharpening filters to improve text clarity

Note: Font sizes cannot be changed in existing PNG files, but upscaling makes
the text appear larger and more readable when viewing the images.

Usage:
    python enhance_plots.py --input-dir <path_to_plots_directory> [--output-dir <output_directory>]
    
Example:
    python enhance_plots.py --input-dir ./multi_resolution_predictions_LSTM_PV+NWP --output-dir ./enhanced_plots
"""



import os

import sys

import argparse

import glob

import numpy as np

import pandas as pd

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from matplotlib.dates import HourLocator, DateFormatter

from PIL import Image, ImageEnhance, ImageFilter

import warnings

warnings.filterwarnings('ignore')



# =============================================================================

# IMAGE ENHANCEMENT FUNCTIONS (for existing PNG files)

# =============================================================================

def enhance_png_image(input_path, output_path, scale_factor=2.0, sharpen=True):

    """

    Enhance an existing PNG image by upscaling and optionally sharpening.

    Note: This cannot change fonts, but can make text more readable by increasing resolution.

    

    Args:

        input_path: Path to input PNG file

        output_path: Path to save enhanced PNG file

        scale_factor: Factor to upscale the image (default: 2.0)

        sharpen: Whether to apply sharpening filter (default: True)

    """

    try:

        # Open the image

        img = Image.open(input_path)

        original_size = img.size

        

        # Upscale using LANCZOS resampling (high quality)

        new_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))

        img_upscaled = img.resize(new_size, Image.Resampling.LANCZOS)

        

        # Apply sharpening if requested

        if sharpen:

            # Convert to RGB if necessary

            if img_upscaled.mode != 'RGB':

                img_upscaled = img_upscaled.convert('RGB')

            

            # Apply unsharp mask for sharpening

            img_upscaled = img_upscaled.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

        

        # Save the enhanced image

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

        img_upscaled.save(output_path, 'PNG', dpi=(300, 300))

        

        print(f"  Enhanced: {os.path.basename(input_path)}")

        print(f"    Original size: {original_size[0]}x{original_size[1]}")

        print(f"    New size: {new_size[0]}x{new_size[1]} (scale: {scale_factor}x)")

        return True

    except Exception as e:

        print(f"  [ERROR] Failed to enhance {input_path}: {str(e)}")

        return False





def enhance_all_pngs_in_directory(input_dir, output_dir=None, scale_factor=2.0, sharpen=True):

    """

    Enhance all PNG files in a directory.

    

    Args:

        input_dir: Directory containing PNG files

        output_dir: Output directory (if None, creates 'enhanced' subdirectory)

        scale_factor: Factor to upscale images

        sharpen: Whether to apply sharpening

    """

    if output_dir is None:

        output_dir = os.path.join(input_dir, 'enhanced')

    

    os.makedirs(output_dir, exist_ok=True)

    

    # Find all PNG files

    png_files = glob.glob(os.path.join(input_dir, '*.png'))

    png_files.extend(glob.glob(os.path.join(input_dir, '**', '*.png'), recursive=True))

    

    if len(png_files) == 0:

        print(f"[WARNING] No PNG files found in {input_dir}")

        return []

    

    print(f"\nFound {len(png_files)} PNG file(s) to enhance")

    print(f"Output directory: {output_dir}")

    print("=" * 80)

    

    enhanced_files = []

    for png_file in png_files:

        # Create output path preserving directory structure

        rel_path = os.path.relpath(png_file, input_dir)

        output_path = os.path.join(output_dir, rel_path)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        

        if enhance_png_image(png_file, output_path, scale_factor, sharpen):

            enhanced_files.append(output_path)

    

    return enhanced_files





# =============================================================================

# PLOT REGENERATION FUNCTIONS (if data is available)

# =============================================================================

def find_prediction_data_files(input_dir):

    """

    Look for saved prediction data files (CSV, pickle, etc.)

    

    Returns:

        dict with keys: 'predictions', 'rmse_data', etc.

    """

    data_files = {}

    

    # Look for CSV files that might contain prediction data

    csv_files = glob.glob(os.path.join(input_dir, '*.csv'))

    csv_files.extend(glob.glob(os.path.join(input_dir, '**', '*.csv'), recursive=True))

    

    for csv_file in csv_files:

        filename = os.path.basename(csv_file).lower()

        if 'prediction' in filename or 'pred' in filename:

            data_files['predictions'] = csv_file

        elif 'rmse' in filename:

            data_files['rmse'] = csv_file

    

    # Look for pickle files

    pickle_files = glob.glob(os.path.join(input_dir, '*.pkl'))

    pickle_files.extend(glob.glob(os.path.join(input_dir, '**', '*.pkl'), recursive=True))

    if pickle_files:

        data_files['pickle'] = pickle_files[0]  # Use first one found

    

    return data_files





def regenerate_plots_from_data(input_dir, output_dir, font_scale=1.5):

    """

    Attempt to regenerate plots from saved data files.

    This is a placeholder - actual implementation depends on what data was saved.

    """

    data_files = find_prediction_data_files(input_dir)

    

    if not data_files:

        print("[INFO] No prediction data files found. Will enhance existing PNG images instead.")

        return False

    

    print(f"[INFO] Found data files: {list(data_files.keys())}")

    print("[INFO] Note: Full plot regeneration requires the original prediction data structure.")

    print("[INFO] Enhancing existing PNG images instead...")

    return False





# =============================================================================

# MAIN FUNCTION

# =============================================================================

def main():

    parser = argparse.ArgumentParser(

        description='Enhance existing plots with larger fonts or upscale existing PNG images',

        formatter_class=argparse.RawDescriptionHelpFormatter

    )

    

    parser.add_argument('--input-dir', type=str, required=True,

                       help='Directory containing plot PNG files')

    parser.add_argument('--output-dir', type=str, default=None,

                       help='Output directory for enhanced plots (default: <input_dir>/enhanced)')

    parser.add_argument('--scale-factor', type=float, default=2.0,

                       help='Upscaling factor for images (default: 2.0)')

    parser.add_argument('--no-sharpen', action='store_true',

                       help='Disable sharpening filter')

    parser.add_argument('--try-regenerate', action='store_true',

                       help='Try to regenerate plots from saved data (if available)')

    

    args = parser.parse_args()

    

    if not os.path.exists(args.input_dir):

        print(f"[ERROR] Input directory does not exist: {args.input_dir}")

        sys.exit(1)

    

    print("=" * 80)

    print("Plot Enhancement Script")

    print("=" * 80)

    print(f"Input directory: {args.input_dir}")

    print(f"Output directory: {args.output_dir or os.path.join(args.input_dir, 'enhanced')}")

    print(f"Scale factor: {args.scale_factor}x")

    print(f"Sharpening: {not args.no_sharpen}")

    print("=" * 80)

    

    # Try to regenerate plots if requested and data is available

    if args.try_regenerate:

        if regenerate_plots_from_data(args.input_dir, args.output_dir):

            print("\n[SUCCESS] Plots regenerated from saved data!")

            return

    

    # Enhance existing PNG images

    print("\nEnhancing existing PNG images...")

    print("Note: This upscales images and applies sharpening to make text more readable.")

    print("Font sizes cannot be changed in existing PNG files, but upscaling helps.")

    print("=" * 80)

    

    enhanced_files = enhance_all_pngs_in_directory(

        args.input_dir,

        args.output_dir,

        scale_factor=args.scale_factor,

        sharpen=not args.no_sharpen

    )

    

    if enhanced_files:

        print(f"\n{'='*80}")

        print(f"[SUCCESS] Enhanced {len(enhanced_files)} plot file(s)")

        print(f"Output directory: {args.output_dir or os.path.join(args.input_dir, 'enhanced')}")

        print(f"{'='*80}")

        print("\nNote: For best results with larger fonts, regenerate plots from the original data.")

        print("To do this, you would need to save prediction data when running the original script.")

    else:

        print(f"\n[WARNING] No files were enhanced. Check that PNG files exist in {args.input_dir}")





if __name__ == "__main__":

    main()

