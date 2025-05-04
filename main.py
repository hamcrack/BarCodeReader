import cv2
import os
import time
from pyzbar.pyzbar import decode
import numpy as np
import math

def get_line_stats(lines, image):
    angle_threshold = 15
    length_threshold = 15
    min_length = 30

    num_lines = len(lines)
    line_stats = np.empty((num_lines, 3), dtype=np.float32)
    angle_bins = {}
    length_bins = {}

    for index, line in enumerate(lines):
        num_lines = len(lines)
    line_stats = np.empty((num_lines, 3), dtype=np.float32)
    angle_length_id_map = {}

    for index, line in enumerate(lines):
        x1, y1, x2, y2 = line[0]
        length = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))
        if length < min_length:
            continue
        opposite = (x1 - x2)
        adjacent = (y1 - y2)
        if adjacent == 0:
            adjacent = 0.00001
        angle_rad = math.atan(opposite / adjacent)
        line_stats[index] = [length, angle_rad, index]

        angle_deg = math.degrees(angle_rad)
        normalized_angle = angle_deg % 180

        # Group by angle
        found_angle_bin = False
        for bin_angle in angle_length_id_map:
            if abs(normalized_angle - bin_angle) <= angle_threshold:
                found_angle_bin = True
                length_map = angle_length_id_map[bin_angle]
                found_length_bin = False
                for bin_length in length_map:
                    if abs(length - bin_length) <= length_threshold:
                        length_map[bin_length].append(index)
                        found_length_bin = True
                        break
                if not found_length_bin:
                    length_map[length] = [index]
                break
        if not found_angle_bin:
            angle_length_id_map[normalized_angle] = {length: [index]}

    lines_in_angle_bins = {}
    print("Angle bins:")
    for angle, length_bins in angle_length_id_map.items():
        lines_in_angle_bins[angle] = 0
        print(f"Angle: {angle:.2f}°")
        for length, indices in length_bins.items():
            lines_in_angle_bins[angle] += len(indices)
            print(f"Length: {length:.2f} - Indices length: {len(indices)}")
        print("      lines_in_angle_bins[angle]:", lines_in_angle_bins[angle])

    return line_stats, angle_length_id_map, lines_in_angle_bins

def get_barcodes(image):
    lsd = cv2.createLineSegmentDetector(0)
    cv_lines = lsd.detect(image)[0]
    line_stats, angle_length_bins, lines_in_angle_bins = get_line_stats(cv_lines, image)
    return cv_lines, line_stats, angle_length_bins, lines_in_angle_bins


def detect_barcodes(image):
    """
    Detects barcodes in an image using OpenCV and returns the pixel coordinates of the corners.

    Args:
        image (numpy.ndarray): The input image (OpenCV format).

    Returns:
        list: A list of tuples, where each tuple contains the four corner points
              of a detected barcode in pixel coordinates.
              Returns an empty list if no barcodes are found.
              Example: [((x1, y1), (x2, y2), (x3, y3), (x4, y4)), ...]
    """
    try:
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use OpenCV's barcode detector
        barcode_detector = cv2.barcode_BarcodeDetector()

        # Detect barcodes using pyzbar
        barcodes = decode(gray)
        print(f"Detected {len(barcodes)} barcodes using pyzbar:", barcodes)

        # My barcode detection
        cv_lines, line_stats, angle_length_bins, lines_in_angle_bins = get_barcodes(gray)

        barcode_lines = []

        most_seen_angs = max(lines_in_angle_bins, key=lambda key: lines_in_angle_bins[key])
        no_angle_lines = lines_in_angle_bins[most_seen_angs]
        print("Most seen angle:", most_seen_angs, "lines_in_angle_bins[most_seen_angs]:", no_angle_lines)
        most_seen_lens = max(angle_length_bins[most_seen_angs], key=lambda key: len(angle_length_bins[most_seen_angs][key]))
        no_len_lines = len(angle_length_bins[most_seen_angs][most_seen_lens])
        print("Most seen length:", most_seen_lens, "lines_in_angle_bins[most_seen_angs][most_seen_lens]:", no_len_lines)
        for index in angle_length_bins[most_seen_angs][most_seen_lens]:
            barcode_lines.append(index)
            # x1, y1, x2, y2 = cv_lines[index][0]
            # cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        for angle, length_bins in angle_length_bins.items():
            new_angle = False
            if lines_in_angle_bins[angle] > lines_in_angle_bins[most_seen_angs] / 2 and angle != most_seen_angs:
                print("Also check angle ", angle)
            for length, indices in length_bins.items():
                if len(indices) > no_len_lines / 3 and length != most_seen_lens:
                    print("Also check len ", length)
                    for index in indices:
                        barcode_lines.append(index)
                        # x1, y1, x2, y2 = cv_lines[index][0]
                        # cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        for index in barcode_lines:
            x1, y1, x2, y2 = cv_lines[index][0]
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        
        # Detect the barcodes
        retv, barcode_corners = barcode_detector.detect(gray)
        if not retv:
            print("No barcodes detected.")
            return []  # Return an empty list if no barcodes are found
        else:
            print(f"Detected {len(barcode_corners)} barcodes using cv:", barcode_corners)

        return barcode_corners

    except Exception as e:
        print(f"An error occurred during barcode detection: {e}")
        return []  # Return an empty list in case of an error

def display_and_resize_images(folder_path, output_folder="out"):
    """
    Displays and resizes each image in a folder to fit the screen, and saves them to a new folder.

    Args:
        folder_path (str): The path to the folder containing the images.
        output_folder (str, optional): The path to the folder where resized images will be saved. Defaults to "out".
    """
    try:
        # Get screen dimensions
        screen_width, screen_height = 1920, 1080 #  set to a default.  You can use  `get_monitors()` from the screeninfo library if needed

        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created output folder: {output_folder}")

        # Get a list of all files in the folder
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))]

        if not image_files:
            print(f"No images found in the folder: {folder_path}")
            return

        # Loop through each image file
        for image_file in image_files:
            # Construct the full path to the image
            image_path = os.path.join(folder_path, image_file)
            output_path = os.path.join(output_folder, image_file) #save with original name

            # Read the image using OpenCV
            img = cv2.imread(image_path)

            if img is None:
                print(f"Error reading image: {image_path}")
                continue  # Skip to the next image

            # Detect barcodes
            barcode_corners_list = detect_barcodes(img)

            # if len(barcode_corners_list) == 0:
            #     for corners in barcode_corners_list:
            #         for corner in corners:

            # M, status = cv2.findHomography(matched_corners[i][0], matched_corners[i][1])
            # top = possible_plate_corners[i][0][1]
            # bottom = possible_plate_corners[i][2][1]
            # left = possible_plate_corners[i][0][0]
            # right = possible_plate_corners[i][1][0]
            # rotated_plate = gray[top:bottom, left:right]
            # cols = matched_corners[i][1][2][0]
            # rows = matched_corners[i][1][2][1]
            # possible_plate = cv2.warpPerspective(rotated_plate, M, (cols, rows))


            # Get original image dimensions
            original_height, original_width = img.shape[:2]

            # Calculate the aspect ratio
            aspect_ratio = original_width / original_height

            # Determine the best fit for the screen
            if original_width > screen_width or original_height > screen_height:
                if aspect_ratio > screen_width / screen_height:
                    new_width = screen_width
                    new_height = int(screen_width / aspect_ratio)
                else:
                    new_height = screen_height
                    new_width = int(screen_height * aspect_ratio)
            else:
                # If the image is smaller than the screen, don't resize
                new_width, new_height = original_width, original_height

            for corners in barcode_corners_list:
                # Convert the corner points to integers for drawing
                corners = [(int(x), int(y)) for x, y in corners]
                for j in range(4):
                    cv2.line(img, corners[j], corners[(j + 1) % 4], (0, 255, 0), 2)  # Green lines


            # Resize the image
            resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            
            # Display the resized image
            cv2.imshow("Resized Image", resized_img)
            cv2.waitKey(0)

            # Save the resized image
            cv2.imwrite(output_path, resized_img)
            print(f"Resized and saved: {image_file} to {output_path}")

        # Destroy the window to free resources
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    folder_path = '3_barcodes'
    output_folder = 'out'
    display_and_resize_images(folder_path, output_folder)
    print("Finished processing images.")