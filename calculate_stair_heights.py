import csv

csv_path = 'src/holosoma_retargeting/holosoma_retargeting/demo_data/TML_data/m00/m00_h07_d13.csv'

def extract_heights():
    with open(csv_path, 'r') as f:
        lines = f.readlines()

    # Fixed indices based on 0-indexing
    # Line 4 (Name) -> index 3
    # Line 7 (Component) -> index 6
    # Line 8 (Axis) -> index 7
    # Line 9 (Data) -> index 8
    
    name_row = lines[3].strip().split(',')
    comp_row = lines[6].strip().split(',')
    axis_row = lines[7].strip().split(',')
    
    markers = ['Stairs:Marker 001', 'Stairs:Marker 002', 'Stairs:Marker 003', 'KAREN_SKELETON:Hip']
    marker_data = {m: {'X': [], 'Y': [], 'Z': []} for m in markers}
    
    marker_indices = {}
    for m in markers:
        indices = [i for i, x in enumerate(name_row) if x == m]
        # Store indices for X, Y, Z
        xyz_indices = {'X': None, 'Y': None, 'Z': None}
        for idx in indices:
            if idx < len(comp_row) and 'Position' in comp_row[idx]:
                if idx < len(axis_row):
                    axis = axis_row[idx]
                    if axis in xyz_indices:
                        xyz_indices[axis] = idx
        marker_indices[m] = xyz_indices

    print("Marker Indices:", marker_indices)
    
    data_start_line = 8
    
    frames = 0
    for i in range(data_start_line, len(lines)):
        row = lines[i].strip().split(',')
        if not row or len(row) < 10: continue
        frames += 1
        
        for m in markers:
            for axis in ['X', 'Y', 'Z']:
                idx = marker_indices[m][axis]
                if idx is not None:
                    try:
                        val = float(row[idx])
                        marker_data[m][axis].append(val)
                    except ValueError:
                        pass

    print(f"Processed {frames} frames.")

    print("\n--- Stair Locations (Avg) ---")
    stair_avgs = {}
    for m in ['Stairs:Marker 001', 'Stairs:Marker 002', 'Stairs:Marker 003']:
        print(f"{m}:")
        avgs = []
        for axis in ['X', 'Y', 'Z']:
            vals = marker_data[m][axis]
            if vals:
                avg = sum(vals) / len(vals)
                print(f"  {axis}: {avg:.4f}")
                avgs.append(avg)
            else:
                print(f"  {axis}: N/A")
                avgs.append(0.0)
        stair_avgs[m] = avgs

    print("\n--- Hip Motion ---")
    hip = 'KAREN_SKELETON:Hip'
    for axis in ['X', 'Y', 'Z']:
        vals = marker_data[hip][axis]
        if vals:
            print(f"{axis}: Start={vals[0]:.4f}, End={vals[-1]:.4f}, Min={min(vals):.4f}, Max={max(vals):.4f}, Delta={vals[-1]-vals[0]:.4f}")

if __name__ == '__main__':
    extract_heights()
