# drone_hover
Camera-based hover control for micro drones.

## Branches

**`drone_img_processing/demo-version`**
Real hardware vision pipeline. Uses two webcams to detect the drone and publish its 2D pixel coordinates via background subtraction (primary method), with YOLOv8 and LED blob detection also available.

**`sim_drone_control`**
Simulation pipeline built on Gazebo. Uses semantic segmentation + depth cameras to compute the drone's 3D position in the camera frame and publishes it for downstream use.

## Status
Sensor fusion (EKF) and the hover controller are currently being worked on and will be added here once stable.
