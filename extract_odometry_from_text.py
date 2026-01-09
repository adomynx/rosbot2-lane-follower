input_file = "topicdata_ex.txt"
output_csv = "topic_data.csv"
lines_per_frame = 102
frame_total = 1691

with open(input_file, 'r') as f:
    all_lines = f.readlines()

with open(output_csv, 'w') as out:
    for i in range(frame_total):
        start = i * lines_per_frame
        block = all_lines[start:start + lines_per_frame]
        linear_x = None
        angular_z = None
        inside_linear = False
        inside_angular = False

        for line in block:
            stripped = line.strip()
            if stripped.startswith("linear:"):
                inside_linear = True
                inside_angular = False
            elif stripped.startswith("angular:"):
                inside_angular = True
                inside_linear = False
            elif inside_linear and stripped.startswith("x:"):
                linear_x = float(stripped.split(":")[1].strip())
            elif inside_angular and stripped.startswith("z:"):
                angular_z = float(stripped.split(":")[1].strip())

        if linear_x is not None and angular_z is not None:
            out.write(f"frame_{i:05d}.png,{linear_x:.6f},{angular_z:.6f}\n")
        else:
            print(f"[⚠️] Skipped frame {i:05d} (incomplete data)")
