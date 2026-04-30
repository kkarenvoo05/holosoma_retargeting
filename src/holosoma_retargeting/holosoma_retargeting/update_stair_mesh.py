# Move staircase mesh vertices by an offset
# X = lateral, Y = forward/back, Z = up/down
offset_x = -0.3
offset_y = 0.0  # positive = forward (toward robot)

OBJ_PATH = "/Users/karenvo/TML_research/holosoma/src/holosoma_retargeting/holosoma_retargeting/demo_data/climb/staircase/multi_boxes.obj"

with open(OBJ_PATH) as f:
    lines = f.readlines()
with open(OBJ_PATH, 'w') as f:
    for line in lines:
        if line.startswith('v '):
            parts = line.split()
            x = float(parts[1]) + offset_x
            y = float(parts[2]) + offset_y
            z = float(parts[3])
            f.write(f'v {x:.8f} {y:.8f} {z:.8f}\n')
        else:
            f.write(line)