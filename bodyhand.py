import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Load CSV
csv_file_path = 'Test1217.csv'
df = pd.read_csv(csv_file_path, dtype=str, header=None, skiprows=1)

# Define Full-Body Joints with Hands and Fingers
full_body_joints = [
    # Spine and Head
    'Skeleton:Ab', 'Skeleton:Chest', 'Skeleton:Neck', 'Skeleton:Head',
    # Left Arm
    'Skeleton:LShoulder', 'Skeleton:LUArm', 'Skeleton:LFArm', 'Skeleton:LHand',
    'Skeleton:LThumb1', 'Skeleton:LThumb2', 'Skeleton:LThumb3',
    'Skeleton:LIndex1', 'Skeleton:LIndex2', 'Skeleton:LIndex3',
    'Skeleton:LMiddle1', 'Skeleton:LMiddle2', 'Skeleton:LMiddle3',
    'Skeleton:LRing1', 'Skeleton:LRing2', 'Skeleton:LRing3',
    'Skeleton:LPinky1', 'Skeleton:LPinky2', 'Skeleton:LPinky3',
    # Right Arm
    'Skeleton:RShoulder', 'Skeleton:RUArm', 'Skeleton:RFArm', 'Skeleton:RHand',
    'Skeleton:RThumb1', 'Skeleton:RThumb2', 'Skeleton:RThumb3',
    'Skeleton:RIndex1', 'Skeleton:RIndex2', 'Skeleton:RIndex3',
    'Skeleton:RMiddle1', 'Skeleton:RMiddle2', 'Skeleton:RMiddle3',
    'Skeleton:RRing1', 'Skeleton:RRing2', 'Skeleton:RRing3',
    'Skeleton:RPinky1', 'Skeleton:RPinky2', 'Skeleton:RPinky3',
    # Legs
    'Skeleton:LThigh', 'Skeleton:LShin', 'Skeleton:LFoot', 'Skeleton:LToe',
    'Skeleton:RThigh', 'Skeleton:RShin', 'Skeleton:RFoot', 'Skeleton:RToe'
]

# Locate column indices corresponding to Full Body joints' position (X, Y, Z)
joint_columns = []
for col_idx, col_name in enumerate(df.iloc[1, 2:]):
    if any(joint == str(col_name).strip() for joint in full_body_joints) \
            and "Position" in str(df.iloc[3, col_idx + 2]) \
            and "Bone" == str(df.iloc[0, col_idx + 2]).strip():
        joint_columns.append(col_idx + 2)

# Extract position data
selected_data = df.iloc[8::3, joint_columns].values.astype(float)

# Prepare for visualization
num_frames = len(selected_data)
num_joints = len(joint_columns) // 3  # Each joint has X, Y, Z
joint_positions = selected_data.reshape(num_frames, num_joints, 3)

# Define connections for the skeleton, including hands
connections = [
    # Spine and Head
    (0, 1), (1, 2), (2, 3),  # Ab → Chest → Neck → Head

    # Left Arm
    (1, 4), (4, 5), (5, 6), (6, 7),  # Chest → LShoulder → LUArm → LFArm → LHand
    (7, 8), (8, 9), (9, 10),  # Left Thumb
    (7, 11), (11, 12), (12, 13),  # Left Index
    (7, 14), (14, 15), (15, 16),  # Left Middle
    (7, 17), (17, 18), (18, 19),  # Left Ring
    (7, 20), (20, 21), (21, 22),  # Left Pinky

    # Right Arm
    (1, 23), (23, 24), (24, 25), (25, 26),  # Chest → RShoulder → RUArm → RFArm → RHand
    (26, 27), (27, 28), (28, 29),  # Right Thumb
    (26, 30), (30, 31), (31, 32),  # Right Index
    (26, 33), (33, 34), (34, 35),  # Right Middle
    (26, 36), (36, 37), (37, 38),  # Right Ring
    (26, 39), (39, 40), (40, 41),  # Right Pinky

    # Left Leg
    (0, 42), (42, 43), (43, 44), (44, 45),  # Ab → LThigh → LShin → LFoot → LToe

    # Right Leg
    (0, 46), (46, 47), (47, 48), (48, 49)  # Ab → RThigh → RShin → RFoot → RToe
]

# Set up 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=135, azim=-90)

# Define finger joints explicitly
finger_joints = [
    'Skeleton:LThumb1', 'Skeleton:LThumb2', 'Skeleton:LThumb3',
    'Skeleton:LIndex1', 'Skeleton:LIndex2', 'Skeleton:LIndex3',
    'Skeleton:LMiddle1', 'Skeleton:LMiddle2', 'Skeleton:LMiddle3',
    'Skeleton:LRing1', 'Skeleton:LRing2', 'Skeleton:LRing3',
    'Skeleton:LPinky1', 'Skeleton:LPinky2', 'Skeleton:LPinky3',
    'Skeleton:RThumb1', 'Skeleton:RThumb2', 'Skeleton:RThumb3',
    'Skeleton:RIndex1', 'Skeleton:RIndex2', 'Skeleton:RIndex3',
    'Skeleton:RMiddle1', 'Skeleton:RMiddle2', 'Skeleton:RMiddle3',
    'Skeleton:RRing1', 'Skeleton:RRing2', 'Skeleton:RRing3',
    'Skeleton:RPinky1', 'Skeleton:RPinky2', 'Skeleton:RPinky3'
]

# Create a size array based on joint type
joint_sizes = []
for joint_name in full_body_joints:
    if joint_name in finger_joints:
        joint_sizes.append(20)  # Smaller size for finger joints
    else:
        joint_sizes.append(50)  # Larger size for major joints

# Initialize scatter plot and lines
sc = ax.scatter([], [], [], c='blue', s=20)
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

ani.save('skeleton_animation.gif', writer='pillow', fps=30)

# Show animation
plt.show()
