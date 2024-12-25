import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Load CSV
csv_file_path = 'Test1217.csv'

df = pd.read_csv(csv_file_path, dtype=str, header=None, skiprows=1)

# Selected joints information
selected_joints = df.iloc[1, 9:370].values.tolist()
selected_joints_num = [
    0, 1, 2, 3, 4, 5, 6, 7, 23, 24, 25, 26, 42, 43, 44, 45, 46, 47, 48, 49
]

# Data extraction
selected_data = df.iloc[9::3, 9:370].values.tolist()
data = []
joint_names = []

for frame_data in selected_data:
    data_line = []
    joints_line = [selected_joints[j * 7] for j in selected_joints_num]  # Joint names
    print(joints_line)
    for j in selected_joints_num:
        #print(j)
        # Extract only xyz (position), skipping xyzw (quaternion)
        position_start = (j * 7) + 4  # Start after quaternion components
        position_end = position_start + 3  # Next 3 components are position
        xyz_position = frame_data[position_start:position_end]
        #print(xyz_position)
        # Ensure positions are parsed as floats
        data_line.extend([float(value) for value in xyz_position])
    #print("11111111")
    data.append(data_line)
    joint_names.append(joints_line)

# Convert to numpy array
data = np.array(data)
# Prepare data for visualization
num_frames = len(data)
num_joints = len(selected_joints_num)
joint_positions = data.reshape(num_frames, num_joints, 3)

# Set up 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initialize scatter plot
sc = ax.scatter([], [], [], c='red', s=50)

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set axis limits dynamically based on data
x_min, x_max = np.min(joint_positions[:, :, 0]), np.max(joint_positions[:, :, 0])
y_min, y_max = np.min(joint_positions[:, :, 1]), np.max(joint_positions[:, :, 1])
z_min, z_max = np.min(joint_positions[:, :, 2]), np.max(joint_positions[:, :, 2])

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

# Update function for animation
def update(frame_idx):
    ax.set_title(f"Frame {frame_idx}")
    sc._offsets3d = (
        joint_positions[frame_idx, :, 0],
        joint_positions[frame_idx, :, 1],
        joint_positions[frame_idx, :, 2]
    )
    return sc,

# Animate using FuncAnimation
ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

# Show animation
plt.show()
