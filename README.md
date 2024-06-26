<h1>Image Stitching using SIFT and OpenCV</h1><br>
This Python script stitches two input images into a single panoramic image using the Scale-Invariant Feature Transform (SIFT) algorithm and OpenCV library.<br>

<h2>Explanation</h2><br>
<h3>Loading and Preprocessing Images</h3><br>
The script first loads two input images (img1 and img2) using the cv2.imread() function from OpenCV. It checks if the images are loaded successfully using an if condition.<br> If either image is not loaded, it prints an error message and exits the script using sys.exit().<br>

Next, the script converts the loaded images to grayscale using the cv2.cvtColor() function with the cv2.COLOR_BGR2GRAY argument.<br>

<h3>SIFT Feature Detection and Matching</h3><br>
The script applies the SIFT feature detector using cv2.SIFT_create(). It computes keypoints and descriptors for both grayscale images using the detectAndCompute method.<br>

A custom function euclidean_distance() calculates the Euclidean distance between descriptors to find matches between keypoints. Matches are stored in the matches list.<br>

<h3>Transformation Estimation and Panorama Creation</h3><br>
An affine transformation matrix is estimated to align keypoints between the two images using the cv2.estimateAffinePartial2D() function. The estimated transformation matrix is then applied to keypoints of the second image to align them with the first image.<br>

<h3>Finally</h3><br>
the script creates a panoramic image by blending the two images together using the estimated transformation. The resulting panoramic image is displayed using OpenCV's cv2.imshow() function.
