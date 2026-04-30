import numpy as np
import argparse
import sys

def truncate_npz(input_path, output_path, target_frames, original_frames=None):
    try:
        data = np.load(input_path)
        new_data = {}
        
        print(f"Processing {input_path}...")
        
        for key in data.files:
            arr = data[key]
            shape = list(arr.shape)
            print(f"  Key: {key}, Shape: {shape}")
            
            # Determine which dimension to truncate
            # If original_frames is specified, look for that dimension
            # Otherwise, look for a dimension that looks like time (e.g. 966)
            
            trunc_dim = -1
            if original_frames is not None:
                if original_frames in shape:
                    trunc_dim = shape.index(original_frames)
            else:
                # Heuristic: Find dimension 966 if implied context, or just ask user? 
                # The user mentioned 966 in the prompt.
                if 966 in shape:
                    trunc_dim = shape.index(966)
            
            if trunc_dim != -1:
                # Slicing logic
                if target_frames > shape[trunc_dim]:
                    print(f"    Warning: Target frames {target_frames} > current frames {shape[trunc_dim]}. Skipping truncation for this array.")
                    new_data[key] = arr
                else:
                    print(f"    Truncating dimension {trunc_dim} from {shape[trunc_dim]} to {target_frames}")
                    
                    # Create slice object
                    # slice(None) is equivalent to ':'
                    slices = [slice(None)] * arr.ndim
                    slices[trunc_dim] = slice(0, target_frames)
                    
                    new_data[key] = arr[tuple(slices)]
            else:
                 # If we haven't found 966, maybe the user wants to truncate the largest dimension?
                 # OR, we verify if there is a dimension that matches the common time dim.
                 # For now, let's look for the largest dimension if it's significantly large?
                 # No, better to be safe. If 966 isn't found, check if there's a dimension close to it?
                 # Or just report no truncation.
                 print(f"    No dimension matching 966 found. Copying as is.")
                 new_data[key] = arr

        # Save to new file
        np.savez_compressed(output_path, **new_data)
        print(f"Saved truncated file to {output_path}")

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Truncate frames in an NPZ file.")
    parser.add_argument("input_path", help="Path to input NPZ file")
    parser.add_argument("output_path", help="Path to output NPZ file")
    parser.add_argument("frames", type=int, help="Number of frames to keep")
    parser.add_argument("--original_frames", type=int, default=966, help="Expected original frame count to identify time dimension (default: 966)")

    args = parser.parse_args()
    
    truncate_npz(args.input_path, args.output_path, args.frames, args.original_frames)
