# Vision-Based Dynamic Obstacle Avoidance for UR5e using ArUco, YOLO, MuJoCo and Kinematic NMPC

IMPORTANT:
Read and understand the entire uploaded repository before writing any code.
Generate a repository analysis report first.
Do not redesign components that already exist.
Reuse existing YOLO, ArUco, MuJoCo, and UR5e code whenever possible.

You are an expert Robotics Engineer specializing in:

- MuJoCo
- UR5e
- Computer Vision
- ArUco Localization
- CasADi
- Nonlinear Model Predictive Control (NMPC)
- Motion Planning
- Robot Collision Avoidance

I am providing an existing project that already contains:

1. YOLO-based bottle detection. (in cvforRobot)
2. ArUco marker localization.
3. UR5e MuJoCo Menagerie model. (in cvforRobot/mujoco_menagerie main universal_robots_ur5e)
4. Existing simulation environment.

Your task is NOT to rebuild these components.

Instead, analyze the provided codebase and integrate them into a complete Vision-Based Dynamic Obstacle Avoidance Framework using Kinematic NMPC.

---

# Existing Components

The repository already includes:

- Bottle detection using computer vision.
- ArUco marker detection.
- MuJoCo simulation.
- UR5e robot model.
- Existing camera pipeline.

Reuse as much code as possible.

Do not replace existing implementations unless necessary.

---

# Main Objective

Build a system where:

Webcam

↓

YOLO Bottle Detection

↓

Bottle Pixel Center

↓

ArUco Localization

↓

Bottle World Position

↓

Obstacle Representation

↓

Kinematic NMPC

↓

UR5e Motion Generation

↓

MuJoCo Simulation

The UR5e must move toward a target while avoiding bottles detected by the vision system.

---

# Repository Analysis

First:

1. Analyze the complete project structure.
2. Identify:
    - YOLO detection files
    - ArUco localization files
    - MuJoCo simulation files
    - UR5e model files
    - Existing control code
3. Explain how data currently flows through the project.
4. Propose the minimum set of modifications needed.

Generate a dependency graph.

---

# Vision System

Assume the vision system already outputs:

Bottle detections.

Determine whether the project currently outputs:

- pixel coordinates
- camera coordinates
- world coordinates

If world coordinates are not available:

use ArUco localization to estimate them.

Output:

obstacle_position = [x,y,z]

for each detected bottle.

---

# ArUco Localization

Use the existing ArUco implementation.

Requirements:

1. Establish a world frame.
2. Estimate camera pose.
3. Convert bottle pixel coordinates into world coordinates.
4. Express bottle positions relative to the ArUco board.

Assume:

Origin:

center of ArUco board

X:

left-right

Y:

forward-backward

Z:

upward

The final output must be:

obstacle_position = [x,y,z]

in meters.

---

# Obstacle Representation

Represent each bottle as:

Option 1:

Sphere

Option 2:

Cylinder

Configurable parameters:

Bottle radius

Bottle height

Safety margin

Example:

radius = 0.04 m

height = 0.22 m

safety_margin = 0.10 m

---

# UR5e Model

Use the existing UR5e MuJoCo Menagerie model.

Do not create a new robot model.

Identify:

- joints
- links
- end-effector
- joint limits

Automatically extract this information from the existing model.

---

# Kinematic NMPC

IMPORTANT:

Use a kinematic NMPC formulation based on:

- Forward Kinematics
- Jacobians
- Differential Kinematics

Avoid full rigid-body dynamics unless explicitly required.

Do NOT use inverse dynamics.

Do NOT formulate a dynamic MPC with robot inertia matrices.

The first implementation must prioritize:

- Simplicity
- Stability
- Real-time performance

State:

q

Control:

q_dot

Prediction Model:

q(k+1) = q(k) + q_dot(k) * dt

Use Jacobians to compute end-effector motion.

---

# Forward Kinematics

Use the existing MuJoCo model to compute:

- link positions
- link transforms
- end-effector pose

Create reusable functions.

---

# Collision Model

Do NOT only check the end-effector.

Model all robot links.

Represent each UR5e link as a capsule.

Links:

- shoulder
- upper_arm
- forearm
- wrist_1
- wrist_2
- wrist_3

Represent bottles as spheres or cylinders.

Compute:

distance(link_capsule, obstacle)

for every link.

---

# Obstacle Avoidance Constraints

For every link:

distance(link_i, obstacle)

must satisfy:

distance > safety_margin

Implement:

1. Hard constraints
2. Soft constraints

Explain advantages and disadvantages.

---

# Cost Function

Minimize:

J =

tracking_cost

+

control_effort

+

smoothness_cost

+

obstacle_cost

Tracking:

move toward target

Control effort:

minimize joint velocity magnitude

Smoothness:

minimize changes in velocity

Obstacle:

strongly penalize approaching obstacles

Provide mathematical equations.

---

# CasADi

Implement using:

CasADi

IPOPT

Create:

- state variables
- control variables
- constraints
- cost function
- solver

Provide modular code.

---

# MuJoCo Integration

Create:

get_robot_state()

compute_forward_kinematics()

compute_link_capsules()

update_obstacle_positions()

build_nmpc()

solve_nmpc()

apply_control()

simulation_step()

main_loop()

---

# Dynamic Obstacles

Assume bottle locations can change every frame.

The controller must:

1. Read new bottle locations.
2. Update constraints.
3. Replan online.
4. Avoid collisions continuously.

---

# Visualization

Inside MuJoCo visualize:

- UR5e
- Bottles
- ArUco frame
- Predicted NMPC trajectory
- Current trajectory
- Safety margin regions

---

# Evaluation

Compute:

- Tracking error
- Minimum obstacle distance
- Solve time
- Success rate
- Constraint violations
- End-effector error

---

# Baseline Comparisons

Provide a framework to compare:

1. RRTConnect
2. Artificial Potential Field
3. Kinematic NMPC (Proposed)

using the same obstacle positions generated from YOLO + ArUco.

Metrics:

- Path length
- Clearance
- Solve time
- Success rate
- Tracking accuracy

---

# Deliverables

1. Repository analysis.
2. Architecture diagram.
3. Required modifications.
4. NMPC mathematical formulation.
5. CasADi implementation.
6. MuJoCo integration.
7. Collision avoidance implementation.
8. Dynamic obstacle avoidance.
9. Evaluation framework.
10. Complete runnable code integrated into the existing repository.

The final solution should be research-grade and suitable for a Master's thesis on Vision-Based Obstacle Avoidance for UR5e Manipulators.