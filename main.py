import cv2
import os
import time
from pyzbar.pyzbar import decode
import numpy as np
import math
from collections import defaultdict
from collections import Counter

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
    for angle, length_bins in angle_length_id_map.items():
        lines_in_angle_bins[angle] = 0
        for length, indices in length_bins.items():
            lines_in_angle_bins[angle] += len(indices)

    return line_stats, angle_length_id_map, lines_in_angle_bins

def get_barcode_lines(image):
    barcode_lines = []
    lsd = cv2.createLineSegmentDetector(0)
    cv_lines = lsd.detect(image)[0]
    line_stats, angle_length_bins, lines_in_angle_bins = get_line_stats(cv_lines, image)

    most_seen_angs = max(lines_in_angle_bins, key=lambda key: lines_in_angle_bins[key])
    no_angle_lines = lines_in_angle_bins[most_seen_angs]
    most_seen_lens = max(angle_length_bins[most_seen_angs], key=lambda key: len(angle_length_bins[most_seen_angs][key]))
    no_len_lines = len(angle_length_bins[most_seen_angs][most_seen_lens])
    for index in angle_length_bins[most_seen_angs][most_seen_lens]:
        barcode_lines.append(cv_lines[index])
        
    for angle, length_bins in angle_length_bins.items():
        for length, indices in length_bins.items():
            if len(indices) > no_len_lines / 3 and length != most_seen_lens:
                for index in indices:
                    barcode_lines.append(cv_lines[index])
    return barcode_lines

def get_random_color():
    """
    Generates a random color in BGR format.

    Returns:
        tuple: A tuple representing a random color in BGR format (B, G, R).
    """
    return (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))

def group_nearby_points(points_with_dist_angle, min_angle):
    """
    Groups points based on proximity and angle, where each point has its own
    maximum distance and a normalized angle. Two points are neighbors if their
    squared Euclidean distance is within the maximum of their max_dist values
    AND the absolute difference between their normalized angles is within min_angle.

    Args:
        points_with_dist_angle: A list of lists, where each inner list is
                                 [ [x, y], max_dist, normalized_angle_deg ].
        min_angle: The minimum absolute difference in normalized angles (in degrees)
                   for two points to be considered neighbors.

    Returns:
        A list of lists, where each inner list represents a group of nearby points
        (each element in the inner list will be the [ [x, y], max_dist, normalized_angle_deg ] format).
    """
    if not points_with_dist_angle:
        return []

    # Determine a reasonable cell size based on the maximum potential max_dist
    max_possible_max_dist = max(item[1] for item in points_with_dist_angle) if points_with_dist_angle else 0
    cell_size = max_possible_max_dist

    grid = defaultdict(list)
    for i, ([x, y], max_d, angle) in enumerate(points_with_dist_angle):
        cell_x = int(x // cell_size) if cell_size > 0 else 0
        cell_y = int(y // cell_size) if cell_size > 0 else 0
        grid[(cell_x, cell_y)].append(i)

    n = len(points_with_dist_angle)
    groups = []
    visited = [False] * n

    def squared_distance(p1_coords, p2_coords):
        return (p1_coords[0] - p2_coords[0])**2 + (p1_coords[1] - p2_coords[1])**2
    
    def angle_difference_circular(angle1, angle2):
        """Calculates the minimum difference between two angles in degrees (0-360 range)."""
        diff = abs(angle1 - angle2)
        return min(diff, 180 - diff)

    def get_nearby_indices(point_index):
        coords1, max_d1, angle1 = points_with_dist_angle[point_index]
        x1, y1 = coords1
        cell_x = int(x1 // cell_size) if cell_size > 0 else 0
        cell_y = int(y1 // cell_size) if cell_size > 0 else 0
        nearby_indices = []
        max_relevant_dist_sq = max_d1**2 if max_d1 >= 0 else float('inf') # Ensure non-negative max_dist

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                neighbor_cell = (cell_x + dx, cell_y + dy)
                if neighbor_cell in grid:
                    for neighbor_index in grid[neighbor_cell]:
                        if neighbor_index != point_index and not visited[neighbor_index]:
                            coords2, max_d2, angle2 = points_with_dist_angle[neighbor_index]
                            max_combined_dist_sq = max(max_relevant_dist_sq, max_d2**2 if max_d2 >= 0 else float('inf'))
                            angle_diff = angle_difference_circular(angle1, angle2)
                            length_diff = abs(max_d1 - max_d2)
                            min_length = min(max_d1, max_d2) / 3
                            if squared_distance(coords1, coords2) <= max_combined_dist_sq and angle_diff <= min_angle and length_diff <= min_length:
                                nearby_indices.append(neighbor_index)
        return nearby_indices

    def find_connected_component(start_node):
        component = []
        queue = [start_node]
        visited[start_node] = True
        component.append(points_with_dist_angle[start_node])

        while queue:
            current_node = queue.pop(0)
            for neighbor_index in get_nearby_indices(current_node):
                if not visited[neighbor_index]:
                    visited[neighbor_index] = True
                    component.append(points_with_dist_angle[neighbor_index])
                    queue.append(neighbor_index)
        return component

    for i in range(n):
        if not visited[i]:
            groups.append(find_connected_component(i))

    return groups

def find_furthest_points(group):
    """Finds the two points in a group with the greatest Euclidean distance."""
    if len(group) < 2:
        point = group[0] if group else None
        return point, point

    furthest_dist_sq = 0
    furthest_pair = [None, None]

    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            coords1 = group[i][0]
            coords2 = group[j][0]
            dist_sq = (coords1[0] - coords2[0])**2 + (coords1[1] - coords2[1])**2
            if dist_sq > furthest_dist_sq:
                furthest_dist_sq = dist_sq
                furthest_pair = [group[i], group[j]]
    return furthest_pair

def get_oriented_box_coordinates(point):
    """
    Calculates the coordinates of the four corners of an oriented box.

    Args:
        center: A list or tuple representing the (x, y) coordinates of the box center.
        half_len: Half the length of the box (used for both width and height).
        angle_deg: The normalized angle of the box in degrees (counter-clockwise from the positive x-axis).

    Returns:
        A list of four (x, y) coordinate tuples representing the corners of the box,
        starting from the top-left corner after rotation.
    """
    [center, half_len, angle_deg] = point
    cx, cy = center
    angle_rad = math.radians(-angle_deg)

    # Define the corner offsets relative to the center before rotation
    corners_local = [
        (-half_len/1.5, half_len),  # Top-left
        (half_len/1.5, half_len),   # Top-right
        (half_len/1.5, -half_len),  # Bottom-right
        (-half_len/1.5, -half_len)  # Bottom-left
    ]

    corners_rotated = []
    for lx, ly in corners_local:
        # Rotation matrix
        cos_theta = math.cos(angle_rad)
        sin_theta = math.sin(angle_rad)

        # Rotate the corner coordinates
        rx = cx + lx * cos_theta - ly * sin_theta
        ry = cy + lx * sin_theta + ly * cos_theta
        corners_rotated.append([rx, ry])
    return corners_rotated

def get_barcode_line_stats(barcode_lines):
    line_centers = []
    for i, line in enumerate(barcode_lines):
        x1, y1, x2, y2 = line[0]
        center_point = (int((x1 + x2) / 2), int((y1 + y2) / 2))

        half_len = math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2)) / 2

        opposite = (x1 - x2)
        adjacent = (y1 - y2)
        if adjacent == 0:
            adjacent = 0.00001
        angle_rad = math.atan(opposite / adjacent)
        angle_deg = math.degrees(angle_rad)
        normalized_angle = angle_deg % 180

        line_centers.append([center_point, half_len, normalized_angle])
        # cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    return line_centers

def get_barcode_boxes(gray):
    barcode_lines = get_barcode_lines(gray)
    line_centers = get_barcode_line_stats(barcode_lines)
    grouped_points = group_nearby_points(line_centers, 2)
    boxes = []
    for group in grouped_points:
        if len(group) > 20:
            points = []
            ends = find_furthest_points(group)
            for end in ends:
                corners_rotated = get_oriented_box_coordinates(end)
                points.append(corners_rotated)
                # for i, corner in enumerate(corners_rotated):
                #     cv2.line(image, (int(corner[0]), int(corner[1])), (int(corners_rotated[i - 1][0]), int(corners_rotated[i - 1][1])), (255, 255, 0), 2)

            for i in range(2):
                dists = []
                for point in points[i]:
                    dists.append(math.sqrt(math.pow(point[0] - ends[1-i][0][0], 2) + math.pow(point[1] - ends[1-i][0][1], 2)))
                mnp = dists.index(min(dists))
                points[i].pop(mnp)
                dists.pop(mnp)
                mnp = dists.index(min(dists))
                points[i].pop(mnp)

            if (points[0][0][0] + points[0][1][0]) / 2 > (points[1][0][0] + points[1][1][0]) / 2:
                temp = points.pop(0)
                points.append(temp)
            
            for side in points:
                if side[0][1] > side[1][1]:
                    temp = side.pop(0)
                    side.append(temp)
            box = [points[0][0], points[1][0], points[1][1], points[0][1]]
            boxes.append(box)
    return boxes


def get_gaps(row):
    gap_widths = []
    last = False
    cnt = 0
    for pix in row:
        if bool(pix) != last:
            last = not last
            gap_widths.append(cnt)
            cnt = 0
        cnt += 1
    gap_widths.pop(0)
    gap_widths.pop(-1)

    largest_gap = max(gap_widths)
    smallest_gap = min(gap_widths) # largest_gap/4
    middle_gap = smallest_gap + (largest_gap - smallest_gap)/2
    return largest_gap, smallest_gap, middle_gap, gap_widths


def decode_code128_binary_image(binary_image, encoding_type='128C'):
    encodings_128A = {
        '212222': ' ', '222122': '!', '222221': '"', '121223': '#', '121322': '$',
        '131222': '%', '122213': '&', '122312': "'", '132212': '(', '221213': ')',
        '221312': '*', '231212': '+', '112232': ',', '122132': '-', '122231': '.',
        '113222': '/', '123122': '0', '123221': '1', '223211': '2', '221132': '3',
        '221231': '4', '213212': '5', '223112': '6', '312131': '7', '311222': '8',
        '321122': '9', '321221': ':', '312212': ';', '322112': '<', '322211': '=',
        '212123': '>', '212321': '?', '232121': '@', '111323': 'A', '131123': 'B',
        '131321': 'C', '112313': 'D', '132113': 'E', '132311': 'F', '211313': 'G',
        '231113': 'H', '231311': 'I', '112133': 'J', '112331': 'K', '132131': 'L',
        '113123': 'M', '113321': 'N', '133121': 'O', '313121': 'P', '211331': 'Q',
        '231131': 'R', '213113': 'S', '213311': 'T', '213131': 'U', '311123': 'V',
        '311321': 'W', '331121': 'X', '312113': 'Y', '312311': 'Z', '332111': '[',
        '314111': '\\', '221411': ']', '431111': '^', '111224': '_', '111422': '\x00',
        '121124': '\x01', '121421': '\x02', '141122': '\x03', '141221': '\x04',
        '112214': '\x05', '112412': '\x06', '122114': '\x07', '122411': '\x08',
        '142112': '\x09', '142211': '\x0A', '241211': '\x0B', '221114': '\x0C',
        '413111': '\x0D', '241112': '\x0E', '134111': '\x0F', '111242': '\x10',
        '121142': '\x11', '121241': '\x12', '114212': '\x13', '124112': '\x14',
        '124211': '\x15', '411212': '\x16', '421112': '\x17', '421211': '\x18',
        '212141': '\x19', '214121': '\x1A', '412121': '\x1B', '111143': '\x1C',
        '111341': '\x1D', '131141': '\x1E', '114113': '\x1F', '114311': 'FNC 3',
        '411113': 'FNC 2', '411311': 'Shift B', '113141': 'Code C', '114131': 'Code B',
        '311141': 'FNC 4', '411131': 'FNC 1', '211412': 'Start A', '211214': 'Start B',
        '211232': 'Start C', '2331112': 'Stop'
    }

    encodings_128B = {
        '212222': ' ', '222122': '!', '222221': '"', '121223': '#', '121322': '$',
        '131222': '%', '122213': '&', '122312': "'", '132212': '(', '221213': ')',
        '221312': '*', '231212': '+', '112232': ',', '122132': '-', '122231': '.',
        '113222': '/', '123122': '0', '123221': '1', '223211': '2', '221132': '3',
        '221231': '4', '213212': '5', '223112': '6', '312131': '7', '311222': '8',
        '321122': '9', '321221': ':', '312212': ';', '322112': '<', '322211': '=',
        '212123': '>', '212321': '?', '232121': '@', '111323': 'A', '131123': 'B',
        '131321': 'C', '112313': 'D', '132113': 'E', '132311': 'F', '211313': 'G',
        '231113': 'H', '231311': 'I', '112133': 'J', '112331': 'K', '132131': 'L',
        '113123': 'M', '113321': 'N', '133121': 'O', '313121': 'P', '211331': 'Q',
        '231131': 'R', '213113': 'S', '213311': 'T', '213131': 'U', '311123': 'V',
        '311321': 'W', '331121': 'X', '312113': 'Y', '312311': 'Z', '332111': '[',
        '314111': '\\', '221411': ']', '431111': '^', '111224': '_', '111422': '`',
        '121124': 'a', '121421': 'b', '141122': 'c', '141221': 'd', '112214': 'e',
        '112412': 'f', '122114': 'g', '122411': 'h', '142112': 'i', '142211': 'j',
        '241211': 'k', '221114': 'l', '413111': 'm', '241112': 'n', '134111': 'o',
        '111242': 'p', '121142': 'q', '121241': 'r', '114212': 's', '124112': 't',
        '124211': 'u', '411212': 'v', '421112': 'w', '421211': 'x', '212141': 'y',
        '214121': 'z', '412121': '{', '111143': '|', '111341': '}', '131141': '~',
        '114113': 'DEL', '114311': 'FNC 3', '411113': 'FNC 2', '411311': 'Shift A',
        '113141': 'Code C', '114131': 'FNC 4', '311141': 'Code A', '411131': 'FNC 1',
        '211412': 'Start A', '211214': 'Start B', '211232': 'Start C', '2331112': 'Stop'
    }

    encodings_128C = {
        '212222': '00', '222122': '01', '222221': '02', '121223': '03', '121322': '04',
        '131222': '05', '122213': '06', '122312': '07', '132212': '08', '221213': '09',
        '221312': '10', '231212': '11', '112232': '12', '122132': '13', '122231': '14',
        '113222': '15', '123122': '16', '123221': '17', '223211': '18', '221132': '19',
        '221231': '20', '213212': '21', '223112': '22', '312131': '23', '311222': '24',
        '321122': '25', '321221': '26', '312212': '27', '322112': '28', '322211': '29',
        '212123': '30', '212321': '31', '232121': '32', '111323': '33', '131123': '34',
        '131321': '35', '112313': '36', '132113': '37', '132311': '38', '211313': '39',
        '231113': '40', '231311': '41', '112133': '42', '112331': '43', '132131': '44',
        '113123': '45', '113321': '46', '133121': '47', '313121': '48', '211331': '49',
        '231131': '50', '213113': '51', '213311': '52', '213131': '53', '311123': '54',
        '311321': '55', '331121': '56', '312113': '57', '312311': '58', '332111': '59',
        '314111': '60', '221411': '61', '431111': '62', '111224': '63', '111422': '64',
        '121124': '65', '121421': '66', '141122': '67', '141221': '68', '112214': '69',
        '112412': '70', '122114': '71', '122411': '72', '142112': '73', '142211': '74',
        '241211': '75', '221114': '76', '413111': '77', '241112': '78', '134111': '79',
        '111242': '80', '121142': '81', '121241': '82', '114212': '83', '124112': '84',
        '124211': '85', '411212': '86', '421112': '87', '421211': '88', '212141': '89',
        '214121': '90', '412121': '91', '111143': '92', '111341': '93', '131141': '94',
        '114113': '95', '114311': '96', '411113': '97', '411311': '98', '113141': '99',
        '114131': 'Code B', '311141': 'Code A', '411131': 'FNC 1',
        '211412': 'Start A', '211214': 'Start B', '211232': 'Start C', '2331112': 'Stop'
    }

    encodings = {
        '128A': encodings_128A,
        '128B': encodings_128B,
        '128C': encodings_128C
    }
    encoding_dict = encodings[encoding_type]

    row_cnt = 0
    print("Length:", len(binary_image))
    for row in binary_image:
        print("Row:", row_cnt)
        row_cnt += 1
        largest_gap, smallest_gap, middle_gap, widths = get_gaps(row)
        print("gaps:", largest_gap, smallest_gap, middle_gap)
        print("Widths", widths)

        widths = []
        last = False
        cnt = 0
        for pix in row:
            if bool(pix) != last:
                last = not last
                if cnt >= middle_gap:
                    large_dif = abs(largest_gap - cnt)
                    middle_dif = abs(cnt - middle_gap)
                    if large_dif > middle_dif:
                        widths.append(4)
                        # print("4 - Cnt:", cnt)
                    else:
                        widths.append(3)
                        # print("3 - Cnt:", cnt)
                else:
                    small_dif = abs(cnt - smallest_gap)
                    middle_dif = abs(middle_gap - cnt)
                    if middle_dif > small_dif:
                        # print("2 - Cnt:", cnt)
                        widths.append(2)
                    else:
                        # print("1 - Cnt:", cnt)
                        widths.append(1)
                cnt = 0
            cnt += 1
        # widths.pop(0)
        # widths.pop(-1)

        print("width codes:", widths)

    return "Nada"

def detect_barcodes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------------------------------
    # Use OpenCV's barcode detector
    # ------------------------------------------------------------------------
    barcode_detector = cv2.barcode_BarcodeDetector()

    retval, decoded_info, points, straight_code = barcode_detector.detectAndDecodeMulti(gray)
    if points is not None:
        print(f"    OpenCV detected {len(points)} barcodes...")
        for i, barcode_corners in enumerate(points):
            # Convert the corner points to integers for drawing
            corners = [(int(x), int(y)) for x, y in barcode_corners]
            for j in range(4):
                cv2.line(image, corners[j], corners[(j + 1) % 4], (0, 255, 0), 2)  # Green lines
            if decoded_info[i] == '':
                text = 'Unknown'
            else:
                text = decoded_info[i]
            cv2.putText(image, f"CV: {text}", (corners[0][0], corners[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Blue lines

    # ------------------------------------------------------------------------
    # Detect barcodes using pyzbar  
    # ------------------------------------------------------------------------
    pyzbarcodes = decode(gray)
    print(f"    Pyzbar detected {len(pyzbarcodes)} barcodes...")
    
    for barcode in pyzbarcodes:
        # Extract barcode data and type
        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type

        # Print barcode data and type
        print("     Barcode Data:", barcode_data)
        print("     Barcode Type:", barcode_type)

        # Draw a rectangle around the barcode
        (x, y, w, h) = barcode.rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Put barcode data and type on the image
        cv2.putText(image, f"Pyzbar: {barcode_data} ({barcode_type})",
                    (x - 300, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Blue lines

    # ------------------------------------------------------------------------
    # My barcode detector
    # ------------------------------------------------------------------------
    boxes = get_barcode_boxes(gray)
    print("    I detected", len(boxes),  "barcodes...")
    box_no = 0
    for box in boxes:
        for i in range(len(box)):
            cv2.line(image, (int(box[i][0]), int(box[i][1])), (int(box[i - 1][0]), int(box[i - 1][1])), (0, 0, 255), 2)

        bar_height = int(math.sqrt(math.pow(box[3][0] - box[0][0], 2) + math.pow(box[3][1] - box[0][1], 2)) + (math.sqrt(math.pow(box[2][0] - box[1][0], 2) + math.pow(box[2][1] - box[1][1], 2))) / 2)
        bar_width = int(math.sqrt(math.pow(box[1][0] - box[0][0], 2) + math.pow(box[1][1] - box[0][1], 2)) + (math.sqrt(math.pow(box[2][0] - box[3][0], 2) + math.pow(box[2][1] - box[3][1], 2))) / 2)

        aligned_points = np.array([[0, 0], [bar_width, 0], [bar_width, bar_height], [0, bar_height]])

        x_values = [point[0] for point in box]
        y_values = [point[1] for point in box]

        min_x = int(min(x_values))
        max_x = int(max(x_values))
        min_y = int(min(y_values))
        max_y = int(max(y_values))

        cut_box = np.zeros((4, 2))
        for i, point in enumerate(box):
            cut_box[i] = [point[0]-min_x, point[1]-min_y]

        cut_gray = gray[min_y:max_y, min_x:max_x]
        M, status = cv2.findHomography(cut_box, aligned_points)

        cut_gray = cv2.warpPerspective(cut_gray, M, (bar_width, bar_height))
        crop = 10
        alighed_barcode = cut_gray[crop:bar_height - crop, crop:bar_width - crop]
        (thresh, alighed_barcode) = cv2.threshold(alighed_barcode, 10, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        bar_data = decode_code128_binary_image(alighed_barcode)
        print("My decode: ", bar_data)

        cv2.imshow("alighed_barcode" + str(box_no), alighed_barcode)
        box_no += 1

    return boxes


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

            print("Processing image ", image_file)

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


            # Resize the image
            resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            
            # Display the resized image
            cv2.imshow("Resized Image", resized_img)
            cv2.waitKey(0)

            # Save the resized image
            cv2.imwrite(output_path, resized_img)

        # Destroy the window to free resources
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    folder_path = '3_barcodes'
    output_folder = 'out'
    display_and_resize_images(folder_path, output_folder)
    print("Finished processing images.")