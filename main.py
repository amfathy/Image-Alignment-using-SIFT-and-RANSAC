import sys
import cv2
import numpy as np

# Load images
img1_path = r'D:\Collegematrial\IIAS\picture1.jpg'
img2_path = r'D:\Collegematrial\IIAS\picture2.jpg'

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)

# Check if images are loaded successfully
if img1 is None or img2 is None:
    print("Error: Unable to load image.")
    sys.exit()

# Convert images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Applying SIFT detector
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)  # Compute descriptors along with keypoints
kp2, des2 = sift.detectAndCompute(gray2, None)  # Compute descriptors along with keypoints

def euclidean_distance(descriptor1, descriptor2):
    return np.linalg.norm(descriptor1 - descriptor2)

matches = []
for i, descriptor1 in enumerate(des1):
    min_distance = float('inf')
    min_index = None
    for j, descriptor2 in enumerate(des2):
        distance = euclidean_distance(descriptor1, descriptor2)
        if distance < min_distance:
            min_distance = distance
            min_index = j
    matches.append(cv2.DMatch(i, min_index, min_distance))

def arrange_pairs(keypoints1, keypoints2, matches):
    arranged_kp2 = []
    kp1_matched = []

    for match in matches:
        kp1_idx, kp2_idx = match.queryIdx, match.trainIdx
        if kp1_idx < len(keypoints1) and kp2_idx < len(keypoints2):
            arranged_kp2.append(keypoints2[kp2_idx])
            kp1_matched.append(keypoints1[kp1_idx])

    return arranged_kp2, kp1_matched

new_kp2, kp1_matched = arrange_pairs(kp1, kp2, matches)

def estimate_transformation(setA, setB):
    setA_pts = np.float32([kp.pt for kp in setA])
    setB_pts = np.float32([kp.pt for kp in setB])

    return cv2.estimateAffinePartial2D(setA_pts, setB_pts)[0]

transformation_matrix = estimate_transformation(kp1_matched, new_kp2)

def apply_transformation(T, set):
    transformed_keypoints = []
    for kp in set:
        x, y = kp.pt
        new_x = T[0, 0] * x + T[0, 1] * y + T[0, 2]
        new_y = T[1, 0] * x + T[1, 1] * y + T[1, 2]
        transformed_keypoints.append(cv2.KeyPoint(new_x, new_y, 1))
    return transformed_keypoints

transformed_kp2 = apply_transformation(transformation_matrix, kp2)

def create_panorama(img1, img2, transformation_matrix):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    img2_warped = cv2.warpAffine(img2, transformation_matrix, (w2, h2))

    mask = np.zeros((h1, w1), dtype=np.uint8)
    mask[:, :] = 255
    mask_warped = cv2.warpAffine(mask, transformation_matrix, (w2, h2))

    blended_img = cv2.addWeighted(img1, 0.5, img2_warped, 0.5, 0)

    blended_img[mask_warped == 255] = img2_warped[mask_warped == 255]

    return blended_img

# Create panorama using the transformed keypoints
panorama = create_panorama(img1, img2, transformation_matrix)

# Display result
cv2.imshow('Output', panorama)
cv2.waitKey(0)
cv2.destroyAllWindows()
sys.exit()