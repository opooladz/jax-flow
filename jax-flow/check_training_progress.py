#!/usr/bin/env python
"""Check training progress for COYO text-to-image model."""

import wandb
import sys
from datetime import datetime

# Initialize wandb API
api = wandb.Api()

# Get the latest run
project = "text_to_image_flow"
entity = "mama-mia"  # Your wandb username

try:
    # Get runs sorted by created time
    runs = api.runs(f"{entity}/{project}", order="-created_at")
    
    if not runs:
        print("No runs found in project")
        sys.exit(1)
    
    # Get the most recent run
    latest_run = runs[0]
    
    print(f"Latest run: {latest_run.name}")
    print(f"State: {latest_run.state}")
    print(f"Created: {latest_run.created_at}")
    print(f"URL: {latest_run.url}")
    print()
    
    # Get history
    history = latest_run.history()
    
    if len(history) == 0:
        print("No metrics logged yet - training might still be initializing")
        print("This is normal for COYO as it needs to download images from the web")
        print("First batch can take 10-30 minutes depending on network speed")
    else:
        print(f"Metrics logged: {len(history)} steps")
        
        # Get latest metrics
        if 'training/l2_loss' in history.columns:
            recent = history[['training/l2_loss', '_step']].dropna().tail(5)
            print("\nRecent loss values:")
            print(recent)
            
            # Check if loss is decreasing
            if len(recent) > 1:
                first_loss = recent.iloc[0]['training/l2_loss']
                last_loss = recent.iloc[-1]['training/l2_loss']
                if last_loss < first_loss:
                    print(f"\n✓ Loss is decreasing: {first_loss:.4f} → {last_loss:.4f}")
                else:
                    print(f"\n⚠ Loss not decreasing yet: {first_loss:.4f} → {last_loss:.4f}")
        
        # Get latest step
        last_step = history['_step'].max()
        print(f"\nCurrent step: {last_step}")
        
        # Estimate time remaining
        if last_step > 0:
            elapsed = (datetime.now() - datetime.fromisoformat(latest_run.created_at)).total_seconds() / 60
            steps_per_min = last_step / elapsed
            remaining_steps = 10000 - last_step  # max_steps from config
            eta_minutes = remaining_steps / steps_per_min if steps_per_min > 0 else 0
            print(f"Speed: {steps_per_min:.1f} steps/min")
            print(f"ETA: {eta_minutes:.0f} minutes ({eta_minutes/60:.1f} hours)")

except Exception as e:
    print(f"Error accessing wandb: {e}")
    print("\nTips:")
    print("1. Training might still be initializing (COYO downloads images on-the-fly)")
    print("2. First batch can take 10-30 minutes")
    print("3. Check https://wandb.ai/mama-mia/text_to_image_flow for manual inspection")