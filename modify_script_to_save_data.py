#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Modify Prediction Script to Save Data for Plot Regeneration

This script modifies your prediction script to save prediction data
so plots can be regenerated with larger fonts later.

Usage:
    python modify_script_to_save_data.py --script-path <path_to_prediction_script.py>
"""



import os

import sys

import argparse

import re



def add_data_saving_to_script(script_path, output_path=None):

    """

    Modify a prediction script to save prediction data.

    

    Args:

        script_path: Path to the original script

        output_path: Path to save modified script (if None, creates backup and modifies in place)

    """

    if not os.path.exists(script_path):

        print(f"[ERROR] Script not found: {script_path}")

        return False

    

    # Read the original script

    with open(script_path, 'r', encoding='utf-8') as f:

        content = f.read()

    

    # Create backup

    backup_path = script_path + '.backup'

    with open(backup_path, 'w', encoding='utf-8') as f:

        f.write(content)

    print(f"Created backup: {backup_path}")

    

    # Check if already modified

    if 'SAVE_PREDICTIONS_FOR_REPLOTTING' in content:

        print("[INFO] Script appears to already have data saving enabled.")

        response = input("Continue anyway? (y/n): ")

        if response.lower() != 'y':

            return False

    

    # Find the function that creates predictions and add data saving

    # Pattern 1: Look for run_predictions_at_resolution or similar function

    # Pattern 2: Look for where all_predictions is created

    modifications = []

    

    # Add import for pickle at the top if not present

    if 'import pickle' not in content and 'from pickle' not in content:

        # Find the last import statement

        import_pattern = r'(import\s+\w+|from\s+\w+\s+import)'

        imports = list(re.finditer(import_pattern, content))

        if imports:

            last_import = imports[-1]

            insert_pos = last_import.end()

            # Find the end of that line

            line_end = content.find('\n', insert_pos)

            if line_end == -1:

                line_end = len(content)

            content = content[:line_end] + '\nimport pickle' + content[line_end:]

            modifications.append("Added 'import pickle'")

    

    # Add data saving code after predictions are created

    # Look for patterns like "all_predictions.append" or "return all_predictions"

    save_code = '''

    # =============================================================================

    # SAVE PREDICTIONS FOR REPLOTTING (added by modify_script_to_save_data.py)

    # =============================================================================

    SAVE_PREDICTIONS_FOR_REPLOTTING = True  # Set to False to disable

    if SAVE_PREDICTIONS_FOR_REPLOTTING:

        import pickle

        import os

        # Save predictions data

        predictions_data_path = os.path.join(output_dir, 'predictions_data.pkl')

        predictions_data = {

            'all_predictions': all_predictions,

            'hourly_rmse_data': hourly_rmse_data,

            'all_pred_data': all_pred_data if 'all_pred_data' in locals() else None,

            'model_name': config.get('experiment_name', 'Model'),

            'resolution_name': resolution_name if 'resolution_name' in locals() else 'Unknown'

        }

        os.makedirs(output_dir, exist_ok=True)

        with open(predictions_data_path, 'wb') as f:

            pickle.dump(predictions_data, f)

        print(f"\\n  [INFO] Saved prediction data for replotting: {predictions_data_path}")

    # =============================================================================

'''

    

    # Try to find where to insert the save code

    # Look for "return all_predictions" or similar return statement

    return_patterns = [

        r'return\s+all_predictions[,\s]*hourly_rmse_data',

        r'return\s+all_predictions[,\s]*hourly_rmse_data[,\s]*all_pred_data',

        r'return\s+all_predictions',

    ]

    

    inserted = False

    for pattern in return_patterns:

        matches = list(re.finditer(pattern, content))

        if matches:

            # Insert before the return statement

            match = matches[-1]  # Use the last match (likely the main return)

            insert_pos = match.start()

            # Find the start of the line

            line_start = content.rfind('\n', 0, insert_pos) + 1

            # Check if we're in the right function (look for function definition above)

            context = content[max(0, line_start-500):line_start]

            if 'def run_predictions_at_resolution' in context or 'def run' in context:

                content = content[:insert_pos] + save_code + '\n    ' + content[insert_pos:]

                modifications.append(f"Added data saving code before return statement")

                inserted = True

                break

    

    if not inserted:

        # Try to find the end of run_multi_resolution_predictions function

        # Look for the last part of the function before final return

        end_pattern = r'print\(f"\\n\{\'=\'\*80\}\)'

        matches = list(re.finditer(end_pattern, content))

        if matches and 'SUCCESS' in content[matches[-1].start():matches[-1].start()+200]:

            # Insert before the final success message

            match = matches[-1]

            insert_pos = match.start()

            # Find a good insertion point (before the final print statements)

            line_start = content.rfind('\n', 0, insert_pos) + 1

            content = content[:line_start] + save_code + content[line_start:]

            modifications.append("Added data saving code at end of function")

            inserted = True

    

    if not inserted:

        print("[WARNING] Could not automatically find insertion point.")

        print("You may need to manually add the data saving code.")

        print("\nAdd this code where predictions are created:")

        print(save_code)

        return False

    

    # Write modified script

    if output_path is None:

        output_path = script_path

    

    with open(output_path, 'w', encoding='utf-8') as f:

        f.write(content)

    

    print(f"\n[SUCCESS] Modified script saved to: {output_path}")

    print(f"Modifications made:")

    for mod in modifications:

        print(f"  - {mod}")

    print(f"\nBackup saved to: {backup_path}")

    print("\nThe script will now save prediction data to 'predictions_data.pkl' in the output directory.")

    print("You can use regenerate_plots_from_saved_data.py to regenerate plots with larger fonts.")

    

    return True





def main():

    parser = argparse.ArgumentParser(

        description='Modify prediction script to save data for plot regeneration',

        formatter_class=argparse.RawDescriptionHelpFormatter

    )

    

    parser.add_argument('--script-path', type=str, required=True,

                       help='Path to the prediction script to modify')

    parser.add_argument('--output-path', type=str, default=None,

                       help='Path to save modified script (default: modifies in place with backup)')

    

    args = parser.parse_args()

    

    print("=" * 80)

    print("Modify Script to Save Prediction Data")

    print("=" * 80)

    print(f"Script: {args.script_path}")

    if args.output_path:

        print(f"Output: {args.output_path}")

    else:

        print("Output: (will modify in place, backup created)")

    print("=" * 80)

    

    if add_data_saving_to_script(args.script_path, args.output_path):

        print("\n[SUCCESS] Script modification completed!")

    else:

        print("\n[ERROR] Script modification failed. See messages above.")

        sys.exit(1)





if __name__ == "__main__":

    main()

