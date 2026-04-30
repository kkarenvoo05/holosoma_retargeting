
import os

obj_path = 'src/holosoma_retargeting/holosoma_retargeting/demo_data/climb/staircase/multi_boxes.obj'

# New heights in meters (m00 data)
# New heights in meters (m00 data)
# Removed inverse scaling to allow URDF to scale down the object
h1 = 0.19809 
h2 = 0.37630
h3 = 0.55733

def update_mesh():
    with open(obj_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    vertex_count = 0
    
    # Vertex ranges (1-based indices in OBJ, so we count as we go)
    # Box 1: 1-8. Top vertices: 5,6,7,8
    # Box 2: 9-16. Top vertices: 13,14,15,16
    # Box 3: 17-24. Top vertices: 21,22,23,24
    
    for line in lines:
        if line.startswith('v '):
            vertex_count += 1
            parts = line.strip().split()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            
            new_z = z
            if vertex_count in [5, 6, 7, 8]:
                new_z = h1
            elif vertex_count in [13, 14, 15, 16]:
                new_z = h2
            elif vertex_count in [21, 22, 23, 24]:
                new_z = h3
            
            new_lines.append(f"v {x:.8f} {y:.8f} {new_z:.8f}\n")
        else:
            new_lines.append(line)
            
    with open(obj_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"Updated {obj_path} with new heights: {h1}, {h2}, {h3}")

if __name__ == '__main__':
    update_mesh()
