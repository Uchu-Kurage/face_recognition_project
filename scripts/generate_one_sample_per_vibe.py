import os
import sys
import json




# Adjust path to import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_bgm import generate_bgm, VIBE_PROMPTS_MAP

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def generate_samples():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.json')
    if os.path.exists(config_path):
        config = load_config(config_path)
        token = config.get("hf_token") or os.getenv("HF_TOKEN")
    else:
        token = os.getenv("HF_TOKEN")
    
    if not token:
        print("Error: HF_TOKEN not found in config.json or environment variables.")
        return

    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output/bgm_samples_current_v4")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting BGM generation... (1 sample per vibe, 10 seconds, CPU forced)")
    print(f"Output directory: {output_dir}")
    
    for vibe in VIBE_PROMPTS_MAP.keys():
        print(f"\nGenerating sample for Vibe: {vibe}")
        try:
            success, path = generate_bgm(
                vibe=vibe,
                duration_seconds=10,
                output_dir=output_dir,
                token=token,
                prompt_index=0, # Use the first prompt
                num_inference_steps=25 # Reduced steps for faster CPU generation
            )
            if success:
                print(f"  Success: {path}")
            else:
                print(f"  Failed to generate sample for {vibe}")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    generate_samples()
