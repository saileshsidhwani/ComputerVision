In this problem I've performed object classification given enough test images for each category. First we train our data with test images 
and then we try to recognize the new test images.
Algorithm of the code:
1.	TRAINING STEP
•	Computed cumulative SIFT features for all training images.
•	Applied PCA on all the features using eigen and covar function of openCV to compute PCA-SIFT features.
•	Applied k-means clustering on these features (for k=100), now the centers matrix of kmeans function are the codewords for our problems.
•	Found histograms using calcHist function using labels as the input for each image.
•	Trained the train data using CvKNeatrrest :: train() pasing histograms of train images as input and response matrix as {1,1,...20times ; 2,2,...20 times and so on}.

2.	TESTING STEP
•	Compute SIFT features for the test image.
•	Used eigen vector of training step to compute PCA features for the test image.
•	Found closest centre for test features using eucildian distance of the features with all the codewords. 
•	Found histogram of test image labels matrix using calcHist.
•	Used CvkNearrest :: find_nearrest() for the histogram of test image.

Step By Step Results:
PCA_DIMENTION : 20
NEIGHBOURS : 12
K(KMEANS) : 50

1.	SIFT features of training data :  [41467 x 128]
2.	Compute eigen vectors EIGEN
3.	PCA SIFT: [41467 x 20]. PCA SIFT = SIFT * EIGEN.
4.	K-means for k = 50 CODEWORDS: [50 x 20]
5.	calcHist for LABELS for each test image. HIST: [100 x 50]
6.	Train data: CvKNearrest on histograms and response matrix RESPONSE : [100 x 1]
7.	SIFT features for TESTING Images : [21959 * 128]
8.	PCASIFT for test images : [21959 * 20]. PCASIFT = SIFT * EIGEN from test data
9.	Compute nearest codewords for all the features using
norm(featuresTest.row(i), centers.row(j), NORM_L2) 
10.	Compute histogram for test images labels for all the codewords.
11.	Find_nearrest(histogramsTest, KNEIGHBOURS, results, neighResponses,dists)
12.	RESULT : [50 x 1] gives us classification for all the test images

TOTAL ACCURACY: 34/50 

DETECTION RATES: 
        butterfly: 40%
        car_side: 100%
        faces: 80%
        watch: 60%
        water_lilly: 60%

CONFUSION MATRIX:
4       3       2       0       1
0       10      0       0       0
0       2       8       0       0
0       2       2       6       0
0       3       1       0       6

CONCLUSION:
Computing PCASIFT features in the training phase and clustering the features are the most time consuming phases in the assignment. The overall algorithm gives us a good level of classification.
The parameters found that gives the best result are:

Accuracy : 34/50 test images classify correctly
Pca_dimention : 20
Neighbours : 12
K(kmeans) : 50

