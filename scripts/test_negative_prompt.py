import os
import sys

# Adjust path to import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_bgm import generate_bgm

def test_negative_prompt():
    token = os.getenv("HF_TOKEN")
    
    output_dir = "output/bgm_negative_test"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Testing BGM generation with negative prompt...")
    
    # Generate with default negative prompt
    # We expect "Drums, Percussion, Jazz, Dissonance, Atonal, Complex, Muddy" to be used
    success, path = generate_bgm(
        vibe="穏やか",
        duration_seconds=10,
        output_dir=output_dir,
        token=token,
        num_inference_steps=25
        # negative_prompt argument omitted to use default
    )
    
    if success:
        print(f"Success! Output: {path}")
    else:
        print("Failed.")

if __name__ == "__main__":
    test_negative_prompt()
