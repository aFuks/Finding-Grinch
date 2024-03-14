import numpy as np
import cv2


def find_contours(binary_image):
    contours = []
    height, width = binary_image.shape

    for y in range(height):
        for x in range(width):
            if binary_image[y, x] == 255:
                contour = [(x, y)]
                binary_image[y, x] = 0
                current_point = (x, y)
                while True:
                    neighbors = [
                        (current_point[0] - 1, current_point[1]),
                        (current_point[0] + 1, current_point[1]),
                        (current_point[0], current_point[1] - 1),
                        (current_point[0], current_point[1] + 1),
                    ]

                    for neighbor in neighbors:
                        nx, ny = neighbor
                        if 0 <= nx < width and 0 <= ny < height and binary_image[ny, nx] == 255:
                            contour.append((nx, ny))
                            binary_image[ny, nx] = 0
                            current_point = (nx, ny)
                            break
                    else:
                        break

                contours.append(np.array(contour))

    return contours


def draw_rectangles(image, bounding_boxes):
    for box in bounding_boxes:
        x, y, w, h = box
        # Rysowanie prostokąta bez wypełnienia
        image[y:y+h, x:x+2] = [0, 255, 0]  # Lewa krawędź
        image[y:y+h, x+w-2:x+w] = [0, 255, 0]  # Prawa krawędź
        image[y:y+2, x:x+w] = [0, 255, 0]  # Górna krawędź
        image[y+h-2:y+h, x:x+w] = [0, 255, 0]  # Dolna krawędź


def contour_area(contour):
    # Wzór Gaussa do obliczania pola wielokąta
    area = 0.5 * abs(sum(x0*y1 - x1*y0 for (x0, y0), (x1, y1) in zip(contour, contour[1:] + [contour[0]])))
    return area


def merge_contours(contours, distance_threshold=120, max_vertical_merge=600, max_horizontal_merge=300, min_contour_area=10):
    merged_contours = []

    for i, current_contour in enumerate(contours):
        if current_contour is None:
            continue

        for j in range(i + 1, len(contours)):
            other_contour = contours[j]
            if other_contour is None:
                continue

            current_min_y = np.min(current_contour[:, 1])
            current_max_y = np.max(current_contour[:, 1])
            other_min_y = np.min(other_contour[:, 1])
            other_max_y = np.max(other_contour[:, 1])

            current_min_x = np.min(current_contour[:, 0])
            current_max_x = np.max(current_contour[:, 0])
            other_min_x = np.min(other_contour[:, 0])
            other_max_x = np.max(other_contour[:, 0])

            distance = np.linalg.norm(np.array(current_contour[0]) - np.array(other_contour[0]))

            if distance <= distance_threshold and \
               abs(current_min_y - other_max_y) <= max_vertical_merge and \
               abs(current_min_x - other_max_x) <= max_horizontal_merge and \
               contour_area(current_contour) + contour_area(other_contour) >= min_contour_area:
                current_contour = np.concatenate((current_contour, other_contour))
                contours[j] = None

        merged_contours.append(current_contour)

    return [contour for contour in merged_contours if contour is not None]


def extract_and_display_largest_contours(image, binary_image, bounding_boxes, num_contours=2):
    largest_bboxes = sorted(bounding_boxes, key=lambda box: box[2] * box[3], reverse=True)[:num_contours]

    largest_contour_areas_image = [image[y:y+h, x:x+w] for x, y, w, h in largest_bboxes]
    largest_contour_areas_binary = [binary_image[y:y+h, x:x+w] for x, y, w, h in largest_bboxes]

    return largest_contour_areas_image, largest_contour_areas_binary


image_path2 = './org.jpg'
image_path1 = './edited.jpg'

image1 = cv2.imread(image_path1)
image2 = cv2.imread(image_path2)

gray1 = np.dot(image1[..., :3], [0.2989, 0.5870, 0.1140])
gray2 = np.dot(image2[..., :3], [0.2989, 0.5870, 0.1140])

diff = np.abs(gray1 - gray2)

threshold = (diff > 25).astype(np.uint8) * 255
threshold_2 = np.copy(threshold)  # threshold_2, bo threshold zwykły jest edytowany

contours = find_contours(threshold)

merged_contours = merge_contours(contours)

merged_bounding_boxes = [
    (np.min(contour[:, 0]), np.min(contour[:, 1]), np.max(contour[:, 0]) - np.min(contour[:, 0]), np.max(contour[:, 1]) - np.min(contour[:, 1]))
    for contour in merged_contours
]

draw_rectangles(image1, merged_bounding_boxes)

# pusty, 4 wymiarowy obrazek
largest_contour_areas_image, largest_contour_areas_binary = extract_and_display_largest_contours(image1, threshold_2, merged_bounding_boxes, num_contours=2)

largest_contour_1 = largest_contour_areas_image[0]
largest_contour_1_binary = largest_contour_areas_binary[0]

# ucinam je, żeby nie było zielonej ramki
largest_contour_1_cropped = largest_contour_1[2:-2, 2:-2]
largest_contour_1_binary_cropped = largest_contour_1_binary[2:-2, 2:-2]

result_image = np.zeros_like(largest_contour_1_cropped, dtype=np.uint8, shape=(largest_contour_1_cropped.shape[0], largest_contour_1_cropped.shape[1], 4))

for y in range(largest_contour_1_cropped.shape[0]):
    for x in range(largest_contour_1_cropped.shape[1]):
        if largest_contour_1_binary_cropped[y, x] == 255:
            result_image[y, x, :3] = largest_contour_1_cropped[y, x]
            result_image[y, x, 3] = 255
        else:
            result_image[y, x] = [0, 0, 0, 0]

cv2.imwrite("grinch.png", result_image)

cv2.imshow('Znalezione Grinche', image1)
cv2.imshow('Wyciety najwiekszy Grinch', largest_contour_1_cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()



