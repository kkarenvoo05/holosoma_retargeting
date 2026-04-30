import numpy as np
import sys

def inspect_npz(file_path):
    try:
        data = np.load(file_path)
        print(f"Inspecting: {file_path}")
        print("Keys, Shapes, and Stats:")
        for key in data.files:
            if 'pos_w' not in key:
                continue
            val = data[key]
            if np.issubdtype(val.dtype, np.number):
                print(f"  {key}: shape={val.shape}, min={val.min():.4f}, max={val.max():.4f}, mean={val.mean():.4f}")
                if val.ndim > 1 and val.shape[-1] == 3:
                    # Print per-axis stats for X, Y, Z
                    dims = val.reshape(-1, 3)
                    print(f"    X: min={dims[:, 0].min():.4f}, max={dims[:, 0].max():.4f}, mean={dims[:, 0].mean():.4f}")
                    print(f"    Y: min={dims[:, 1].min():.4f}, max={dims[:, 1].max():.4f}, mean={dims[:, 1].mean():.4f}")
                    print(f"    Z: min={dims[:, 2].min():.4f}, max={dims[:, 2].max():.4f}, mean={dims[:, 2].mean():.4f}")
            else:
                print(f"  {key}: shape={val.shape}, type={val.dtype}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_npz.py <path_to_npz>")
    else:
        inspect_npz(sys.argv[1])
