In this assignment we perform an affine projections of given images of rectangular solid.
The given data is of 6 images of an object taken from various points which have 8 points in common

Algorithm of the code:
•	Accept image number from user
•	Draw original image for the corresponding image
•	Center the given data to center of mass of the object
•	Compute SVD for the centered data D= U.W.Vt
•	Strip matrix U only to 3 columns, Vt to 3 rows and construct W3 by selecting only 3 eigen values of W. 
•	Decompose W in following way, U3 * W3sqrt and W3sqrt * Vt3, where first matrix would be motion matrix and the latter would be shape matrix
•	Project the 3D points in 2D image using the motion matrix
•	Translate the image origin to the original origin.
