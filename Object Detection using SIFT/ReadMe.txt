1.) In the project I've implement Homography and RANSAC algorithms to detect test objects from the image. 

2.)Problem was successfully simplified by linearizing the homography equation X2 = H*X1.

3.) 1st section of the problem was implemented using copyTo and adjustROI functions and the core algorithm was developed referring 
knnMatcher for k=2. 

4.) SIFT.operator was successfully used to extract SIFT features.

5.) 2nd section was implemented by selecting 4 random keypoints from the test image and using those points to calculate the homography
matrix and using self-implemented RANSAC algorithm to improve the homography matrix.

6.) Two sets of points are projected in the final images, white point corresponds to original keypoints detected from the original image 
and black set of points correspond to estimated keypoints using the homography matrix.

7.) For the purpose of simplicity, all the images are in the grayscale.
HOMOGRAPHY MATRIX: