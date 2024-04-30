#Image Stitching using SIFT and OpenCV
This Python script stitches two input images into a single panoramic image using the Scale-Invariant Feature Transform (SIFT) algorithm and OpenCV library.

##Explanation
####Loading and Preprocessing Images
The script first loads two input images (img1 and img2) using the cv2.imread() function from OpenCV. It checks if the images are loaded successfully using an if condition. If either image is not loaded, it prints an error message and exits the script using sys.exit().

Next, the script converts the loaded images to grayscale using the cv2.cvtColor() function with the cv2.COLOR_BGR2GRAY argument.

####SIFT Feature Detection and Matching
The script applies the SIFT feature detector using cv2.SIFT_create(). It computes keypoints and descriptors for both grayscale images using the detectAndCompute method.

A custom function euclidean_distance() calculates the Euclidean distance between descriptors to find matches between keypoints. Matches are stored in the matches list.

####Transformation Estimation and Panorama Creation
An affine transformation matrix is estimated to align keypoints between the two images using the cv2.estimateAffinePartial2D() function. The estimated transformation matrix is then applied to keypoints of the second image to align them with the first image.

####Finally 
the script creates a panoramic image by blending the two images together using the estimated transformation. The resulting panoramic image is displayed using OpenCV's cv2.imshow() function.
