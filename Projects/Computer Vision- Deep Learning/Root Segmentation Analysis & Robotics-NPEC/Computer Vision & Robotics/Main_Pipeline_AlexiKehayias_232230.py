import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize, remove_small_objects
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model
from ot2_gym_wrapper import OT2Env  # Assuming this is where the OT2Env is defined
from scipy.spatial.distance import euclidean
from skimage.graph import route_through_array
import pandas as pd
import re
import time
from stable_baselines3 import PPO

# -------------------------------------------------------------------
# ----------------------- Old + New CV Pipeline -------------------------
# -------------------------------------------------------------------

def f1_score(y_true, y_pred):
    def recall_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = TP / (Positives + K.epsilon())
        return recall
    
    def precision_m(y_true, y_pred):
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives + K.epsilon())
        return precision
    
    precision_val, recall_val = precision_m(y_true, y_pred), recall_m(y_true, y_pred)
    return 2 * ((precision_val * recall_val) / (precision_val + recall_val + K.epsilon()))


def padder(image, divisor, padding_value=(0, 0, 0)):
    original_height, original_width = image.shape[:2]
    pad_height = (divisor - (original_height % divisor)) % divisor
    pad_width = (divisor - (original_width % divisor)) % divisor

    top_pad = pad_height // 2
    bottom_pad = pad_height - top_pad
    left_pad = pad_width // 2
    right_pad = pad_width - left_pad

    padded_image = cv2.copyMakeBorder(
        image,
        top_pad, bottom_pad,
        left_pad, right_pad,
        cv2.BORDER_CONSTANT,
        value=padding_value
    )
    return padded_image


def reduce_noise(image):
    blurred_image = cv2.GaussianBlur(image, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened_image = cv2.morphologyEx(blurred_image, cv2.MORPH_OPEN, kernel)
    closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel)
    return closed_image


def morphological_petri_dish_crop(image):
    thresh = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    
    x, y, w, h = cv2.boundingRect(largest_contour)
    size = int(max(w, h) * 0.98)
    cx, cy = x + w // 2, y + h // 2
    
    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(image.shape[1], cx + size // 2)
    y2 = min(image.shape[0], cy + size // 2)
    
    cropped_img = image[y1:y2, x1:x2].astype(np.float32) / 255.0
    cropped_img = cv2.resize(cropped_img, (size, size), interpolation=cv2.INTER_AREA)
    
    bbox = (x1, y1, x2, y2)
    return cropped_img, bbox


def padder_with_overlap(image, divisor, padding_value=(0, 0, 0)):
    original_height, original_width = image.shape[:2]
    new_height = original_height - (original_height % divisor)
    new_width = original_width - (original_width % divisor)

    top_overlap = (original_height - new_height) // 2
    bottom_overlap = original_height - new_height - top_overlap
    left_overlap = (original_width - new_width) // 2
    right_overlap = (original_width - new_width) - left_overlap

    cropped_image = image[top_overlap:original_height-bottom_overlap, 
                          left_overlap:original_width-right_overlap]

    padded_image = cv2.copyMakeBorder(
        cropped_image,
        top_overlap, bottom_overlap,
        left_overlap, right_overlap,
        cv2.BORDER_CONSTANT,
        value=padding_value
    )
    return padded_image, (new_height, new_width)


def patch_image(image, patch_size=256, stride=128):
    patches = []
    positions = []
    for i in range(0, image.shape[0] - patch_size + 1, stride):
        for j in range(0, image.shape[1] - patch_size + 1, stride):
            patch = image[i:i + patch_size, j:j + patch_size]
            patches.append(patch)
            positions.append((i, j))
    return np.array(patches), positions


def predict_patches(patches, model, batch_size=32):
    patches = np.array(patches)
    if patches.ndim == 2:
        patches = patches[..., np.newaxis]
    predictions = model.predict(patches, batch_size=batch_size)
    return predictions


def unpatch_image(patches, positions, image_shape, patch_size=256):
    reconstructed = np.zeros((*image_shape, 1), dtype=np.float32)
    patch_count = np.zeros((*image_shape, 1), dtype=np.float32)

    for patch, (i, j) in zip(patches, positions):
        if patch.ndim == 2:
            patch = patch[..., np.newaxis]
        reconstructed[i:i + patch_size, j:j + patch_size, :] += patch
        patch_count[i:i + patch_size, j:j + patch_size, :] += 1

    reconstructed /= np.maximum(patch_count, 1)
    return np.squeeze(reconstructed)


def reverse_padding_and_cropping(reconstructed, original_shape, bbox):
    final_mask = np.zeros(original_shape, dtype=reconstructed.dtype)
    x1, y1, x2, y2 = bbox
    final_mask[y1:y2, x1:x2] = reconstructed[:y2 - y1, :x2 - x1]
    return final_mask


def process_root_mask(mask, kernel_size=1, iterations=1400, min_area=150):
    mask_normalized = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
    mask_8bit = np.uint8(mask_normalized)

    _, binary_mask = cv2.threshold(mask_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    opened_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    dilated_mask = cv2.dilate(opened_mask, kernel, iterations=iterations)
    closed_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed_mask)
    filtered_mask = np.zeros_like(closed_mask)
    for i in range(1, num_labels):  # skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered_mask[labels == i] = 255

    return filtered_mask


def skeletonize_mask_skimage(processed_mask, min_size):
    binary_mask = np.array(processed_mask > 0, dtype=bool)
    cleaned_mask = remove_small_objects(binary_mask, min_size=min_size)
    skeleton = skeletonize(cleaned_mask)
    skeleton_uint8 = np.uint8(skeleton) * 255
    return skeleton_uint8


def create_overlay(image, mask, alpha=0.5):
    image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    image_rgb = np.stack([image]*3, axis=-1)
    mask_rgb = np.zeros_like(image_rgb)
    mask_rgb[..., 0] = mask
    
    overlay_image = cv2.addWeighted(image_rgb, 1 - alpha, mask_rgb, alpha, 0)
    return overlay_image


def find_endpoints(skeleton):
    skeleton_coords = np.column_stack(np.where(skeleton > 0))
    endpoints = []
    for coord in skeleton_coords:
        x, y = coord
        neighborhood = skeleton[max(0, x - 1):x + 2, max(0, y - 1):y + 2]
        if np.sum(neighborhood) == 2:
            endpoints.append((x, y))
    return endpoints


def measure_root_from_component(component_image, label_id):
    root_mask = (component_image == label_id).astype(np.uint8)
    skeleton = skeletonize(root_mask > 0)

    endpoints = find_endpoints(skeleton)
    if len(endpoints) < 2:
        raise ValueError(f"Not enough endpoints detected for label {label_id}.")

    # topmost endpoint => start_point
    start_point = min(endpoints, key=lambda p: p[0])
    # farthest from start => tip
    tip = max(endpoints, key=lambda p: euclidean(start_point, p))

    skeleton_coords = np.column_stack(np.where(skeleton > 0))
    distances = [euclidean(start_point, coord) for coord in skeleton_coords]
    length = max(distances)

    return length, start_point, tip, skeleton


def is_moderately_vertical(skeleton_coords, max_horizontal_to_vertical_ratio=0.5):
    if len(skeleton_coords) < 2:
        return False
    y_coords = skeleton_coords[:, 0]
    x_coords = skeleton_coords[:, 1]
    total_vertical_change = np.ptp(y_coords)
    total_horizontal_change = np.ptp(x_coords)
    ratio = total_horizontal_change / (total_vertical_change + 1e-6)
    return ratio <= max_horizontal_to_vertical_ratio


def isolate_and_measure_roots_by_plant(
    image_path, 
    min_area=80, 
    max_horizontal_to_vertical_ratio=0.5, 
    min_length=10, 
    dish_bbox=None
):
    if dish_bbox is None or len(dish_bbox) != 4:
        raise ValueError("A valid dish_bbox (dish_x1, dish_y1, dish_x2, dish_y2) must be provided.")

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read the image: {image_path}")

    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(gray)

    results_by_plant = []
    for plant_label in range(1, retval):  # skip background
        plant_area = stats[plant_label, cv2.CC_STAT_AREA]
        if plant_area >= min_area:
            x, y, w, h = (
                stats[plant_label, cv2.CC_STAT_LEFT],
                stats[plant_label, cv2.CC_STAT_TOP],
                stats[plant_label, cv2.CC_STAT_WIDTH],
                stats[plant_label, cv2.CC_STAT_HEIGHT]
            )

            plant_mask = (labels == plant_label).astype(np.uint8)
            plant_image = plant_mask[y:y+h, x:x+w]

            retval_roots, root_labels, root_stats, _ = cv2.connectedComponentsWithStats(plant_image)

            plant_roots = []
            for root_label in range(1, retval_roots):  # skip background
                root_area = root_stats[root_label, cv2.CC_STAT_AREA]
                if root_area >= min_area:
                    try:
                        length, start, tip, skeleton = measure_root_from_component(root_labels, root_label)
                        skeleton_coords = np.column_stack(np.where(skeleton > 0))
                        if length >= min_length and is_moderately_vertical(skeleton_coords, max_horizontal_to_vertical_ratio):
                            root_x_rel = root_stats[root_label, cv2.CC_STAT_LEFT]
                            root_y_rel = root_stats[root_label, cv2.CC_STAT_TOP]
                            root_w = root_stats[root_label, cv2.CC_STAT_WIDTH]
                            root_h = root_stats[root_label, cv2.CC_STAT_HEIGHT]

                            root_x = x + root_x_rel
                            root_y = y + root_y_rel

                            global_start = (start[0] + y, start[1] + x)
                            global_tip   = (tip[0] + y,   tip[1] + x)

                            plant_roots.append({
                                "root_label": root_label,
                                "length": length,
                                "start": global_start,
                                "tip": global_tip,
                                "skeleton": skeleton,
                                "bounding_box": (root_x, root_y, root_w, root_h)
                            })
                    except ValueError:
                        pass  # Not enough endpoints

            results_by_plant.append({
                "plant_label": plant_label,
                "plant_area": plant_area,
                "roots": plant_roots,
                "bounding_box": (x, y, w, h)
            })

    # Flatten all roots
    all_roots = [root for plant in results_by_plant for root in plant["roots"]]

    dish_x1, dish_y1, dish_x2, dish_y2 = dish_bbox
    dish_width = dish_x2 - dish_x1
    segment_width = dish_width / 5.0
    plant_bins = []
    for i in range(5):
        left_bound = dish_x1 + int(round(i * segment_width))
        right_bound = dish_x1 + int(round((i + 1) * segment_width))
        plant_bins.append((left_bound, right_bound))

    # Sort by x-coord of start => col = 1
    all_roots = sorted(all_roots, key=lambda root: root["start"][1])

    final_results = [{"plant_id": i+1, "length": 0.0, "roots": []} for i in range(5)]
    for root in all_roots:
        start_x = root["start"][1]
        for i, (left_bound, right_bound) in enumerate(plant_bins):
            if left_bound <= start_x < right_bound:
                final_results[i]["roots"].append(root)
                if root["length"] > final_results[i]["length"]:
                    final_results[i]["length"] = root["length"]
                break

    for plant_result in final_results:
        sorted_roots = sorted(plant_result["roots"], key=lambda r: r["length"], reverse=True)
        for idx, root in enumerate(sorted_roots):
            root["root_id"] = f"Root {plant_result['plant_id']}-{idx+1}"

    for i, plant_result in enumerate(final_results):
        if not plant_result["roots"]:
            plant_result["length"] = 0.0

    return final_results


def display_and_save_roots_by_plant(results_by_plant, output_directory, image_basename):
    os.makedirs(output_directory, exist_ok=True)
    for plant in results_by_plant:
        for root in plant["roots"]:
            root_id = root.get("root_id", "unknown")
            root_filename = f"{image_basename}_{root_id}.png"
            x, y, w, h = root["bounding_box"]
            skeleton_mask = root['skeleton']

            blank_canvas = np.zeros((h, w, 3), dtype=np.uint8)
            skeleton_overlay = (skeleton_mask * 255).astype(np.uint8)
            blank_canvas[:skeleton_mask.shape[0], :skeleton_mask.shape[1], 2] = skeleton_overlay

            output_path = os.path.join(output_directory, root_filename)
            cv2.imwrite(output_path, blank_canvas)


def extract_root_coordinates(final_results):
    """
    Returns a dict of the form:
    {
       'plant_1': [
          { 'root_id': 'Root 1-1', 'start':(row,col), 'tip':(row,col), 'length':...},
          { 'root_id': 'Root 1-2', ... },
          ...
       ],
       'plant_2': [...],
       ...
    }
    """
    coords_dict = {}
    for plant_data in final_results:
        plant_id = plant_data["plant_id"]
        plant_key = f"plant_{plant_id}"
        coords_dict[plant_key] = []
        
        for root in plant_data["roots"]:
            root_id = root.get("root_id", "unknown_root")
            start_pt = root["start"]
            tip_pt   = root["tip"]
            length   = root["length"]

            coords_dict[plant_key].append({
                "root_id": root_id,
                "start": start_pt,
                "tip": tip_pt,
                "length": length
            })
    return coords_dict


def plot_tips_on_original_image(original_image_path, final_results, output_path):
    original_img = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
    if original_img is None:
        raise FileNotFoundError(f"Could not load original image: {original_image_path}")

    for plant_data in final_results:
        for root in plant_data["roots"]:
            tip = root["tip"]  # (row, col)
            tip_x = tip[1]
            tip_y = tip[0]
            cv2.circle(original_img, (tip_x, tip_y), radius=6, color=(0, 0, 255), thickness=-1)
            root_id = root.get("root_id", "UnknownRoot")
            cv2.putText(original_img, root_id, (tip_x+8, tip_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(output_path, original_img)


def main_pipeline_example(env):
    # 1) Load model
    model_path = "final_modified_model.h5"
    model = load_model(model_path, custom_objects={"f1_score": f1_score})

    # 2) Initialize the OT2 environment (already created as `env`)
    image_path = env.get_plate_image()
    if not image_path:
        raise ValueError("Failed to retrieve plate image path from the environment.")

    # 3) Load the grayscale image
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 4) Morphological crop
    cropped_img, petri_dish_bbox = morphological_petri_dish_crop(original_image)

    # 5) Pad
    patching_size = 256
    padded_image, _ = padder_with_overlap(cropped_img, patching_size)

    # 6) Patch
    stride = 256
    patched, positions = patch_image(padded_image, patch_size=patching_size, stride=stride)

    # 7) Predict
    predicted = predict_patches(patched, model)

    # 8) Unpatch
    unpatched = unpatch_image(predicted, positions, padded_image.shape, patch_size=patching_size)

    # 9) Reverse crop/padding
    final_mask = reverse_padding_and_cropping(unpatched, original_image.shape, petri_dish_bbox)

    # 10) Post-process
    processed_mask = process_root_mask(final_mask)
    skeletonized_mask = skeletonize_mask_skimage(processed_mask, min_size=150)

    # 11) Create overlay
    overlay = create_overlay(original_image, skeletonized_mask, alpha=0.5)

    # 12) Save final mask + overlay
    output_dir = "Final_Masks_Iteration_10"
    os.makedirs(output_dir, exist_ok=True)
    base_name = "plate_image"

    final_mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    overlay_path    = os.path.join(output_dir, f"{base_name}_overlay.png")

    cv2.imwrite(final_mask_path, (final_mask * 255).astype(np.uint8))
    overlay_bgr = cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(overlay_path, overlay_bgr)

    print(f"Saved final mask to {final_mask_path}")
    print(f"Saved overlay to {overlay_path}")

    # 13) Isolate + measure
    results_by_plant = isolate_and_measure_roots_by_plant(
        final_mask_path,
        min_area=500,
        max_horizontal_to_vertical_ratio=0.5,
        min_length=10,
        dish_bbox=petri_dish_bbox
    )

    # 14) Save skeleton overlays
    skeleton_output_dir = "Skeleton_Iteration_10"
    display_and_save_roots_by_plant(results_by_plant, skeleton_output_dir, base_name)

    # 15) Plot tips on original
    final_tips_plot_path = os.path.join(skeleton_output_dir, f"{base_name}_tips_plotted.png")
    plot_tips_on_original_image(image_path, results_by_plant, final_tips_plot_path)
    print(f"Plotted root endpoints on original image: {final_tips_plot_path}")

    # 16) Convert to dictionary
    coords_dict = extract_root_coordinates(results_by_plant)
    return coords_dict, petri_dish_bbox, image_path


# -------------------------------------------------------------------
# ----------------------- Bigger Pipeline ---------------------------
# -------------------------------------------------------------------

def is_within_bounds(robot_coords, env):
    x, y, z = robot_coords
    x_bounds, y_bounds, z_bounds = env.envelope["position_bounds"]
    return x_bounds[0] <= x <= x_bounds[1] and y_bounds[0] <= y <= y_bounds[1] and z_bounds[0] <= z <= z_bounds[1]


def convert_to_robot_coords(pixel_coords, plate_size_mm, plate_size_pixels, plate_position_robot, bbox_origin_pixels=(0, 0)):
    # plate_size_pixels might be int or tuple
    if isinstance(plate_size_pixels, tuple):
        assert plate_size_pixels[0] == plate_size_pixels[1], "Non-square plate image detected."
        plate_size_pixels = plate_size_pixels[0]

    # mm per pixel
    conversion_factor = plate_size_mm / plate_size_pixels

    # Shift by bounding box origin
    adjusted_pixel_coords = np.array(pixel_coords) - np.array(bbox_origin_pixels)

    # Convert to mm
    root_tip_mm = adjusted_pixel_coords * conversion_factor

    # Add plate position offset
    root_tip_robot_xy = root_tip_mm + np.array(plate_position_robot[:2])

    # Include z
    root_tip_robot = np.append(root_tip_robot_xy, plate_position_robot[2])

    return tuple(root_tip_robot)


def normalize_within_envelope(goal_coords, envelope_bounds):
    x = np.clip(goal_coords[0], envelope_bounds["x"][0], envelope_bounds["x"][1])
    y = np.clip(goal_coords[1], envelope_bounds["y"][0], envelope_bounds["y"][1])
    z = np.clip(goal_coords[2], envelope_bounds["z"][0], envelope_bounds["z"][1])
    return np.array([x, y, z])


def integrate_endpoints_into_robot_space(
    all_endpoints, 
    plate_size_mm, 
    plate_size_pixels, 
    plate_position_robot, 
    bbox_origin_pixels=(0, 0), 
    envelope_bounds=None
):
    """
    all_endpoints = coords_dict from main pipeline (already filtered so only -1 roots)
    {
       'plant_1': [
          { 'root_id': 'Root 1-1', 'start':(row,col), 'tip':(row,col), 'length':...},
          ...
       ],
       'plant_2': [...],
       ...
    }
    """
    goal_positions = []

    # Ensure we process plant_1, plant_2, plant_3, etc. in ascending order
    sorted_plant_keys = sorted(all_endpoints.keys(), key=lambda k: int(k.split("_")[-1]))

    for plant_id in sorted_plant_keys:
        for root_data in all_endpoints[plant_id]:
            # We already filtered to only keep those that end with "-1"
            root_label = root_data.get("root_id", "NoLabel")

            start_pixel = root_data["start"]  # (row, col)
            tip_pixel   = root_data["tip"]    # (row, col)

            start_robot_coords = convert_to_robot_coords(
                start_pixel, 
                plate_size_mm, 
                plate_size_pixels, 
                plate_position_robot, 
                bbox_origin_pixels
            )
            tip_robot_coords   = convert_to_robot_coords(
                tip_pixel, 
                plate_size_mm, 
                plate_size_pixels, 
                plate_position_robot, 
                bbox_origin_pixels
            )

            if envelope_bounds:
                start_robot_coords = normalize_within_envelope(start_robot_coords, envelope_bounds)
                tip_robot_coords   = normalize_within_envelope(tip_robot_coords,   envelope_bounds)

            goal_positions.append({
                "plant_id": plant_id,    # e.g. 'plant_1'
                "root_label": root_label,
                "start": start_robot_coords,
                "tip": tip_robot_coords
            })

    return goal_positions


def run_entire_process():
    # 1) Create the environment once with render=True
    env = OT2Env(render=True)
    
    # 2) Run the old+new pipeline -> coords_dict
    coords_dict, petri_dish_bbox, original_image_path = main_pipeline_example(env)

    print("\n--- Received coords_dict from pipeline ---")
    for p, items in coords_dict.items():
        print(p, items)

    # -----------------------------------------------------------
    # *** FILTER ROOTS TO KEEP ONLY THOSE WHOSE 'root_id' ENDS WITH "-1" ***
    # -----------------------------------------------------------
    filtered_coords_dict = {}
    for plant_key, roots in coords_dict.items():
        filtered_roots = [r for r in roots if r["root_id"].endswith("-1")]
        filtered_coords_dict[plant_key] = filtered_roots

    print("\n--- Filtered coords_dict (only '-1' roots) ---")
    for p, items in filtered_coords_dict.items():
        print(p, items)

    # 3) BBox origin
    bbox_origin_pixels = (petri_dish_bbox[0], petri_dish_bbox[1])
    plate_size_pixels  = petri_dish_bbox[2] - petri_dish_bbox[0]  # width in pixels

    # 4) Plate size: 0.15 (i.e. 150 mm).
    plate_size_mm = 0.15

    # NOTE: Subtract 0.026 from the Y coordinate
    #       Originally, we had: (0.10775, 0.088, 0.057)
    #       Now => Y = 0.088 - 0.026 = 0.062
    plate_position_robot = (0.10775, 0.088 - 0.026, 0.057)

    # 5) Envelope bounds in meters
    envelope_bounds = {
        "x": [-0.1872, 0.2531],
        "y": [-0.1711, 0.2201],
        "z": [0.1691,  0.2896]
    }

    # 6) Convert the filtered endpoints (only -1 roots) to robot coordinates
    goal_positions = integrate_endpoints_into_robot_space(
        filtered_coords_dict,
        plate_size_mm,
        plate_size_pixels,
        plate_position_robot,
        bbox_origin_pixels,
        envelope_bounds
    )

    print("\n--- Goal positions in robot space (only '-1' roots, left to right) ---")
    for g in goal_positions:
        print(g)

    # 7) Assign these goals to the environment
    env.goals = goal_positions

    # 8) Load your RL model
    model_path = (
        "C:/Users/User/Desktop/2024-25b-fai2-adsai-AlexiKehayias232230/"
        "datalab_tasks/task11/Local Runs/Iteration 5- Gamma-1.0/models/3k99r6z0/best_model.zip"
    )
    model = PPO.load(model_path)

    # 9) Move the robot to each goal in order
    for goal_pos in goal_positions:
        # We'll go to the tip
        root_pos_robot = np.array(goal_pos["tip"], dtype=float)  # (x, y, z) in meters

        print(f"\n--- Attempting to move to goal: {root_pos_robot}")
        if is_within_bounds(root_pos_robot, env):
            env.goal_position = root_pos_robot
            print(f"Moving to {root_pos_robot} ...")
            
            obs, info = env.reset()
            while True:
                action, _states = model.predict(obs, deterministic=False)
                obs, rewards, terminated, truncated, info = env.step(action)

                # Distance to goal
                distance_vec = root_pos_robot - obs[:3]  # assume obs[:3] is robot end-effector
                error = np.linalg.norm(distance_vec)

                print(f"Current Pos: {obs[:3]}, Error: {error:.4f}, Action: {action}")

                if error < 0.001:  # ~1 mm threshold
                    # Drop inoculum (example action)
                    action = np.array([0, 0, 0, 1])  # "drop" 
                    obs, rewards, terminated, truncated, info = env.step(action)
                    print(f"Inoculum dropped at: {root_pos_robot}")
                  
                    break
                time.sleep(0.1)
                if terminated:
                    obs, info = env.reset()
                    print("Terminated -> Reset environment, continuing...")

        else:
            print(f"Goal {root_pos_robot} out of bounds, skipping.")


if __name__ == "__main__":
    run_entire_process()
