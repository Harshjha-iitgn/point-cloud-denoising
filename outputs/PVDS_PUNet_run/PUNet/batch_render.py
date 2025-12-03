import os
import numpy as np
import matplotlib.pyplot as plt

# Input folder containing all .xyz files
input_dir = "outputs/PVDS_PUNet_run/output/PUNet/P2PBridge_steps_20/pcl"

# Output folder for all rendered PNGs
output_dir = "rendered_pcl"
os.makedirs(output_dir, exist_ok=True)

def render_pointcloud(xyz_path, out_path):
    pts = np.loadtxt(xyz_path)
    x, y, z = pts[:,0], pts[:,1], pts[:,2]

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=1)

    # Equal aspect ratio
    max_range = (
        np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    )
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5

    ax.set_xlim(mid_x-max_range, mid_x+max_range)
    ax.set_ylim(mid_y-max_range, mid_y+max_range)
    ax.set_zlim(mid_z-max_range, mid_z+max_range)

    ax.view_init(elev=20, azim=40)
    ax.set_axis_off()

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved", out_path)


# Iterate over every .xyz and render it
for file in sorted(os.listdir(input_dir)):
    if file.endswith(".xyz"):
        base = os.path.splitext(file)[0]
        xyz_path = os.path.join(input_dir, file)
        out_path = os.path.join(output_dir, base + ".png")
        render_pointcloud(xyz_path, out_path)
