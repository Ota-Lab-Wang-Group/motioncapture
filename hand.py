import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Load CSV
csv_file_path = 'Test1217.csv'
df = pd.read_csv(csv_file_path, dtype=str, header=None, skiprows=1)

# Define Right Hand Joints
right_hand_joints = [
    "Skeleton:RHand",
    "Skeleton:RThumb1", "Skeleton:RThumb2", "Skeleton:RThumb3",
    "Skeleton:RIndex1", "Skeleton:RIndex2", "Skeleton:RIndex3",
    "Skeleton:RMiddle1", "Skeleton:RMiddle2", "Skeleton:RMiddle3",
    "Skeleton:RRing1", "Skeleton:RRing2", "Skeleton:RRing3",
    "Skeleton:RPinky1", "Skeleton:RPinky2", "Skeleton:RPinky3"
]

# Locate column indices corresponding to Right Hand joints' position (X, Y, Z)
joint_columns = []
for col_idx, col_name in enumerate(df.iloc[1, 2:]):
    if any(joint in col_name for joint in right_hand_joints) \
            and "Position" in str(df.iloc[3, col_idx + 2]) \
            and "RHandIn" not in col_name \
            and "RHandOut" not in col_name:
        joint_columns.append(col_idx + 2)

# Extract position data
selected_data = df.iloc[8::3, joint_columns].values.astype(float)

# Prepare for visualization
num_frames = len(selected_data)
num_joints = len(joint_columns) // 3  # Each joint has X, Y, Z
joint_positions = selected_data.reshape(num_frames, num_joints, 3)

# Define connections between joints (pairs of indices)
connections = [
    (0, 1), (1, 2), (2, 3),       # Thumb
    (0, 4), (4, 5), (5, 6),       # Index finger
    (0, 7), (7, 8), (8, 9),       # Middle finger
    (0, 10), (10, 11), (11, 12),  # Ring finger
    (0, 13), (13, 14), (14, 15)   # Pinky finger
]

# Set up 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=135, azim=-90)
# Initialize scatter plot and line objects
sc = ax.scatter([], [], [], c='blue', s=50)
lines = [ax.plot([], [], [], c='black')[0] for _ in connections]

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
    
    # Update scatter plot
    sc._offsets3d = (
        joint_positions[frame_idx, :, 0],
        joint_positions[frame_idx, :, 1],
        joint_positions[frame_idx, :, 2]
    )
    
    # Update lines for connections
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
