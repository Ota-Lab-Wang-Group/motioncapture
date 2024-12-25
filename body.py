import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Load CSV
csv_file_path = 'Test1217.csv'
df = pd.read_csv(csv_file_path, dtype=str, header=None, skiprows=1)

# Define Full-Body Joints
full_body_joints = [
    'Skeleton:Ab', 'Skeleton:Chest', 'Skeleton:Neck', 'Skeleton:Head', 'Skeleton:LShoulder', 'Skeleton:LUArm', 'Skeleton:LFArm', 'Skeleton:LHand', 'Skeleton:RShoulder', 'Skeleton:RUArm', 'Skeleton:RFArm', 'Skeleton:RHand', 'Skeleton:LThigh', 'Skeleton:LShin', 'Skeleton:LFoot', 'Skeleton:LToe', 'Skeleton:RThigh', 'Skeleton:RShin', 'Skeleton:RFoot', 'Skeleton:RToe'
]

# Locate column indices corresponding to Full Body joints' position (X, Y, Z)
joint_columns = []
for col_idx, col_name in enumerate(df.iloc[1, 2:]):
    if any(joint == str(col_name).strip() for joint in full_body_joints) and "Position" in str(df.iloc[3, col_idx + 2]) and "Bone" == str(df.iloc[0, col_idx + 2]).strip()  :
        joint_columns.append(col_idx + 2)

# Extract position data
selected_data = df.iloc[8::3, joint_columns].values.astype(float)

# Prepare for visualization
num_frames = len(selected_data)
num_joints = len(joint_columns) // 3  # Each joint has X, Y, Z
joint_positions = selected_data.reshape(num_frames, num_joints, 3)
print(num_joints)
# Define connections for the skeleton
connections = [
    # Spine and Head
    (0, 1), (1, 2), (2, 3),  # Ab → Chest → Neck → Head

    # Left Arm
    (1, 4), (4, 5), (5, 6), (6, 7),  # Chest → LShoulder → LUArm → LFArm → LHand

    # Right Arm
    (1, 8), (8, 9), (9, 10), (10, 11),  # Chest → RShoulder → RUArm → RFArm → RHand

    # Left Leg
    (0, 12), (12, 13), (13, 14), (14, 15),  # Ab → LThigh → LShin → LFoot → LToe

    # Right Leg
    (0, 16), (16, 17), (17, 18), (18, 19)  # Ab → RThigh → RShin → RFoot → RToe
]


# Set up 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=135, azim=-90)

# Initialize scatter plot and lines
sc = ax.scatter([], [], [], c='blue', s=50)
lines = [ax.plot([], [], [], c='black')[0] for _ in connections]

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set axis limits
x_min, x_max = np.min(joint_positions[:, :, 0]), np.max(joint_positions[:, :, 0])
y_min, y_max = np.min(joint_positions[:, :, 1]), np.max(joint_positions[:, :, 1])
z_min, z_max = np.min(joint_positions[:, :, 2]), np.max(joint_positions[:, :, 2])

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

# Update function for animation
def update(frame_idx):
    ax.set_title(f"Frame {frame_idx}")
    
    # Update scatter points
    sc._offsets3d = (
        joint_positions[frame_idx, :, 0],
        joint_positions[frame_idx, :, 1],
        joint_positions[frame_idx, :, 2]
    )
    
    # Update connections
    for line, (start, end) in zip(lines, connections):
        x_coords = np.array([joint_positions[frame_idx, start, 0], joint_positions[frame_idx, end, 0]])
        y_coords = np.array([joint_positions[frame_idx, start, 1], joint_positions[frame_idx, end, 1]])
        z_coords = np.array([joint_positions[frame_idx, start, 2], joint_positions[frame_idx, end, 2]])
        line.set_data(x_coords, y_coords)
        line.set_3d_properties(z_coords)
    
    return [sc] + lines

# Animate using FuncAnimation
ani = FuncAnimation(fig, update, frames=num_frames, interval=50, blit=False)

# Show animation
plt.show()
