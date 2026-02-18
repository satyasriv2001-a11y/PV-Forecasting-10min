#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Code Snippet to Add Data Saving to Your Prediction Script

Copy and paste this code into your prediction script where predictions are created.

Place it right before the return statement in run_predictions_at_resolution function.

"""



# =============================================================================
# ADD THIS CODE TO YOUR PREDICTION SCRIPT
# =============================================================================
# Place this code in run_predictions_at_resolution function, 
# right before: return all_predictions, hourly_rmse_data, all_pred_data
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
        'resolution_name': resolution_name if 'resolution_name' in locals() else f"{resolution_minutes}-minute"
    }
    
    os.makedirs(output_dir, exist_ok=True)
    with open(predictions_data_path, 'wb') as f:
        pickle.dump(predictions_data, f)
    
    print(f"\n  [INFO] Saved prediction data for replotting: {predictions_data_path}")

# =============================================================================
# END OF CODE TO ADD
# =============================================================================

