# Root Analysis and Robotic Control Pipeline

This document provides a comprehensive overview of a Python-based pipeline for analyzing plant root images, processing the data, and controlling a robotic arm to perform tasks based on the analysis. The pipeline includes advanced image processing, root measurement, robot coordinate conversion, and control via a reinforcement learning (RL) model.

---

## Table of Contents
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Pipeline Overview](#pipeline-overview)
- [Key Functions and Methods](#key-functions-and-methods)
  - [Image Preprocessing](#image-preprocessing)
  - [Prediction and Reconstruction](#prediction-and-reconstruction)
  - [Root Analysis](#root-analysis)
  - [Visualization and Saving](#visualization-and-saving)
  - [Robot Control](#robot-control)
- [Execution Workflow](#execution-workflow)
- [Robot Control with Reinforcement Learning](#robot-control-with-reinforcement-learning)
- [Execution Script](#execution-script)

---

## Introduction

The pipeline is designed to analyze images of plant roots captured in a laboratory setting. It processes these images to extract key features like root length, orientation, and endpoint positions. The results are then used to control a robotic arm to interact with the plant roots, such as by inoculating them with substances. The system integrates image analysis, machine learning, and robotics seamlessly.

---

## Dependencies

The following libraries and modules are required:

- General-purpose modules: `os`, `re`, `time`, `pandas`
- Image processing: `cv2` (OpenCV), `numpy`, `PIL.Image`, `skimage.morphology`
- Machine learning: `tensorflow.keras`, `tensorflow.keras.models`, `tensorflow.keras.backend`
- Spatial analysis: `scipy.spatial.distance`, `skimage.graph`
- Reinforcement learning: `stable_baselines3`
- Custom modules: `ot2_gym_wrapper`, `PID_Controller_232230`

---

## Pipeline Overview

The pipeline consists of the following stages:

1. **Image Preprocessing:** Prepare root images for analysis using cropping, padding, and noise reduction techniques.
2. **Prediction and Reconstruction:** Utilize a pre-trained neural network to segment roots and reconstruct the processed image.
3. **Root Analysis:** Skeletonize root structures, identify endpoints, and measure root lengths and orientations.
4. **Robot Coordinate Conversion:** Translate pixel-based root data into robot workspace coordinates.
5. **Robot Control:** Employ a reinforcement learning model to move a robotic arm to specific positions for root inoculation.
6. **Visualization:** Generate overlays and visual outputs to verify results and document the analysis process.

---

## Key Functions and Methods

### Image Preprocessing

#### `f1_score(y_true, y_pred)`
Computes the F1 score to evaluate model predictions based on precision and recall.

#### `padder(image, divisor, padding_value=(0, 0, 0))`
Pads an image to ensure its dimensions are divisible by the specified divisor. Useful for ensuring compatibility with model input requirements.

#### `reduce_noise(image)`
Applies Gaussian blur and morphological operations to reduce noise and enhance the clarity of root structures.

#### `morphological_petri_dish_crop(image)`
Crops the input image to focus on the region containing the petri dish, which houses the plant roots.

#### `padder_with_overlap(image, divisor, padding_value=(0, 0, 0))`
Applies padding with overlap to maintain continuity between patches during processing.

#### `patch_image(image, patch_size=256, stride=128)`
Divides the image into smaller patches of specified size and stride for batch processing by the model.

### Prediction and Reconstruction

#### `predict_patches(patches, model, batch_size=32)`
Uses a pre-trained neural network to generate predictions for each image patch.

#### `unpatch_image(patches, positions, image_shape, patch_size=256)`
Reconstructs the full image from its processed patches and their original positions.

#### `reverse_padding_and_cropping(reconstructed, original_shape, bbox)`
Restores the reconstructed image to its original dimensions by reversing padding and cropping operations.

### Root Analysis

#### `process_root_mask(mask, kernel_size=1, iterations=1400, min_area=150)`
Cleans and enhances the binary root mask by applying morphological operations and filtering small components.

#### `skeletonize_mask_skimage(processed_mask, min_size)`
Skeletonizes the binary root mask to create a simplified representation of root structures.

#### `create_overlay(image, mask, alpha=0.5)`
Creates a visualization by overlaying the root mask on the original image with adjustable transparency.

#### `find_endpoints(skeleton)`
Identifies endpoints in the skeletonized root structure to determine root tip positions.

#### `measure_root_from_component(component_image, label_id)`
Calculates root length and identifies endpoints for a specified root component.

#### `is_moderately_vertical(skeleton_coords, max_horizontal_to_vertical_ratio=0.5)`
Evaluates whether a skeleton is predominantly vertical based on its coordinate distribution.

#### `isolate_and_measure_roots_by_plant(...)`
Separates and measures root structures for each plant, organizing results by plant.

### Visualization and Saving

#### `display_and_save_roots_by_plant(results_by_plant, output_directory, image_basename)`
Saves skeleton overlays for individual roots to the specified output directory.

#### `plot_tips_on_original_image(original_image_path, final_results, output_path)`
Plots the root tip positions on the original image for verification.

#### `extract_root_coordinates(final_results)`
Generates a structured dictionary of root start and tip positions with associated lengths.

### Robot Control

#### `is_within_bounds(robot_coords, env)`
Verifies that a given robot coordinate lies within the workspace bounds.

#### `convert_to_robot_coords(...)`
Converts pixel coordinates from the image to robot workspace coordinates.

#### `normalize_within_envelope(goal_coords, envelope_bounds)`
Ensures that robot goals remain within the operational envelope by clamping coordinates.

#### `integrate_endpoints_into_robot_space(...)`
Maps and normalizes root endpoints into the robot's coordinate system for execution.

---

## Execution Workflow

### `main_pipeline_example(env)`
Steps:
1. **Model Loading:** Loads the pre-trained neural network for segmentation.
2. **Image Processing:** Captures the grayscale image and crops the region containing the petri dish.
3. **Segmentation:** Processes the image to identify and segment root structures.
4. **Skeletonization:** Skeletonizes the root mask to extract key features.
5. **Visualization:** Generates and saves overlays, skeleton images, and annotated outputs.
6. **Data Conversion:** Extracts root measurements and prepares the data for robotic control.

### `run_entire_process()`
Comprehensive execution:
1. Initializes the OT2 environment.
2. Executes the image analysis pipeline.
3. Filters root data to retain primary roots.
4. Transforms pixel coordinates into robot coordinates.
5. Utilizes reinforcement learning to guide the robot to root tips and perform inoculations.
6. Gracefully handles errors and resets to complete all goals.

---

## Robot Control with Reinforcement Learning

The pipeline uses a pre-trained RL model (PPO) to control the robotic arm. The robot dynamically adjusts its movements based on feedback from the environment.

### RL Execution Details:
1. **Goal Assignment:** The RL model receives a goal coordinate in the robot workspace.
2. **Action Prediction:** The model predicts the optimal actions to move closer to the goal.
3. **Environment Feedback:** The robot’s current state is updated based on the action and environment response.
4. **Error Reduction:** Movement continues until the positional error is below a defined threshold.
5. **Task Completion:** The robot performs the required action (e.g., inoculation) upon reaching the goal.

### Example:
```python
obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=False)
    obs, rewards, terminated, truncated, info = env.step(action)

    distance_vec = root_pos_robot - obs[:3]  # Calculate error
    error = np.linalg.norm(distance_vec)

    if error < 0.001:
        action = np.array([0, 0, 0, 1])  # Drop inoculum
        obs, rewards, terminated, truncated, info = env.step(action)
        break

    if terminated:
        obs, info = env.reset()
```

---

## Execution Script

The execution script ties all components together:

1. **Environment Initialization:** Sets up the OT2 environment and loads the RL model.
2. **Pipeline Execution:** Runs the image processing and segmentation pipeline to extract root data.
3. **Goal Conversion:** Filters and converts pixel-based root endpoints into robot workspace goals.
4. **Robotic Execution:** Uses RL to guide the robot to each goal and perform tasks.
5. **Error Handling:** Resets and reinitializes as needed to ensure task completion.

---

