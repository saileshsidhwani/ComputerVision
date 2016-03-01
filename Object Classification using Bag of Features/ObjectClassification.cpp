#include<iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/ml/ml.hpp"

int const TRAIN = 0;
int const TEST = 1;
int const BUTTERFLY = 1;
int const CAR_SIDE = 2;
int const FACES = 3;
int const WATCH = 4;
int const WATER_LILLY = 5;
int const NUM_TRAINIMAGES = 100;

int const PCA_DIMENTION = 20;
int const CLUSTERS = 50;
int const KNEIGHBOURS = 12;

int const ATTEMPTS = 10;
int const READDATA = 0;

using namespace std;
using namespace cv;

Mat eigenT;

//folder path for training images
map<int, string> trainMap = { { 1, "butterfly/train/" },
							{ 2, "car_side/train/" },
							{ 3, "faces/train/" },
							{ 4, "watch/train/" },
							{ 5, "water_lilly/train/" } };

//folder path for test images
map<int, string> testMap = { { 1, "butterfly/test/" },
							{ 2, "car_side/test/" },
							{ 3, "faces/test/" },
							{ 4, "watch/test/" },
							{ 5, "water_lilly/test/" } };

void calcPCA(Mat &features , Mat &eigenT)
{

	Mat descriptors, covar, mu, eigenValues, eigenVectors, _eigenVectors,transformed;

	calcCovarMatrix(features, covar, mu, CV_COVAR_NORMAL | CV_COVAR_ROWS);
	eigen(covar, eigenValues, eigenVectors);
	_eigenVectors = eigenVectors(Rect(0, 0, eigenVectors.cols, PCA_DIMENTION));
	eigenT = _eigenVectors.t();
	eigenT.convertTo(eigenT, CV_32F);
	features = features * eigenT;
}

/*
This function calculates PCA-SIFT features for the given image and appends the calculated features to features matrix
*/
void calcPCASIFT(Mat trainImg,  Mat &features, vector<int> &sizeVec,int flag)
{
	SIFT sift;
	vector<KeyPoint> keypoints;
	InputArray mask = cv::noArray();
	Mat descriptors;

	//calculates sift discriptors for the image in discriptor matrix 
	sift.operator()(trainImg, mask, keypoints, descriptors, false);
	features.push_back(descriptors);
	sizeVec.push_back(descriptors.rows);
}

void readData(int objectClass, Mat &features, vector<int> &sizeVec , int flag )
{
	string filename;
	Mat trainImg;
	int count = 0;


	if (flag == TRAIN)
		cout << "\tReading Images from " << trainMap[objectClass] << endl;
	else
		cout << "\tReading Images from " << testMap[objectClass] << endl;
	
	//Compute the names of the images to be read
	//image name format : image_4d.jpg
	for (int i = 0; i < 400; i++)
	{
		if (flag == TRAIN)
			filename = trainMap[objectClass];
		else
			filename = testMap[objectClass];
		
		filename += "image_0";
		if (i > 99)
			filename = filename +  to_string(i) + ".jpg";
		else if (i>9)
			filename = filename + "0" + to_string(i) + ".jpg";
		else
			filename = filename + "00" + to_string(i) + ".jpg";

		trainImg = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);		//read the image in gray scale

		//if image not present, look for other next image number
		if (trainImg.empty())
			continue;
		else
		{
			calcPCASIFT(trainImg, features, sizeVec,flag);		//compute SIFT features for each image

			//maximum of 20 images per class
			if (++count >= 20 )
				break;
		}
	}
}

/*
This function calculates histograms of all the codewords(cluster centers) for each test image.
Histogram is computed on labels of features for each image
*/
void calcHistograms(Mat labels, vector<int> sizeVec, Mat &histograms)
{
	float ran[] = { 0, CLUSTERS };
	const float* range[] = { ran };
	int channels[] = { 0 };						//Histogram is 1-D with range of {0-NumClusters} and size = NumClusters
	int histSize[] = { CLUSTERS }, y = 0;

	labels.convertTo(labels, CV_32F);
	for (unsigned i = 0; i < sizeVec.size(); i++)
	{
		Mat hist;
		Mat temp = labels(Rect(0, y, 1, sizeVec[i]));		//Extract labels of each image to compute histogram from all the cumulative labels
		calcHist(&temp, 1, channels, Mat(), hist, 1, histSize, range, true, false);
		y += sizeVec[i];
		hist = hist.t();									//transpose to store histogram of each image in a single row
		hist = hist.mul(float(1.0/sizeVec[i]));				//normalize the histogram calculated to remove the effect of no. of features in each image
		histograms.push_back(hist);
	}
}

/*
This function finds closest center for the test image features*/
void calcLabelsTest(Mat featuresTest, Mat centers, Mat &labelsTest)
{
	for (int i = 0; i < featuresTest.rows;i++)
	{
		double dist = 999999999,tempDist;
		int center = 0;
		for (int j = 0; j < centers.rows; j++)
		{
			tempDist = norm(featuresTest.row(i), centers.row(j), NORM_L2);
			if (tempDist < dist)
			{
				dist = tempDist;
				center = j;
			}
		}
		labelsTest.push_back(center);
	}
}

/*
This function prints statictics for the code
Stats considered are Total Accuracy, Dectection Rates for each class and Confusion Matrix*/
void printStats(Mat results)
{
	Mat stats(5, 5, CV_32F, Scalar(0));
	double detectionRate[5] = { 000 };
	int k = 0, correctCount = 0;

	//Compute Confusion matrix
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 10; j++)
		{
			if (i == results.at<float>(k + j, 0) - 1)
			{
				correctCount++;
				detectionRate[i]++;
			}
			stats.at<float>(i, results.at<float>(k + j, 0) - 1)++;
		}
		k += (results.rows / 5);
	}

	cout << "\nTotal Accuracy: " << correctCount << "/50" << endl;

	//detection rate
	cout << "\nDetection Rate : " << endl;
	for (int i = 1; i <= 5; i++)
	{
		string s = testMap[i];
		s = s.substr(0, s.size() - 6);
		double percent = (detectionRate[i - 1] / 10) * 100;
		cout << "\t" << s << ": " << percent << "%" << endl;
	}

	cout << "\nConfusion Matrix:" << endl;
	//display confusion matrix
	for (int i = 0; i < 5; i++)
	{
		for (int j = 0; j < 5; j++)
		{
			cout << stats.at<float>(i, j) << "\t";
		}
		cout << endl;
	}
}

int main()
{
	Mat featuresTrain, labels, centers, histograms,eigenT;
	vector<int> sizeVec;
	Mat featuresTest, labelsTest, histogramsTest;
	vector<int> sizeVecTest;
	CvKNearest knn;
	KNearest kn;
	
	Mat response,results, neighResponses, dists;
	if (READDATA == 1)
	{
		//Read Training Data 
		cout << "Reading Training Images..." << endl;
		readData(BUTTERFLY, featuresTrain, sizeVec, TRAIN);
		readData(CAR_SIDE, featuresTrain, sizeVec, TRAIN);
		readData(FACES, featuresTrain, sizeVec, TRAIN);
		readData(WATCH, featuresTrain, sizeVec, TRAIN);
		readData(WATER_LILLY, featuresTrain, sizeVec, TRAIN);
		calcPCA(featuresTrain,eigenT);

		//Read Test Data
		cout << "Reading Test Images..." << endl;
		readData(BUTTERFLY, featuresTest, sizeVecTest, TEST);
		readData(CAR_SIDE, featuresTest, sizeVecTest, TEST);
		readData(FACES, featuresTest, sizeVecTest, TEST);
		readData(WATCH, featuresTest, sizeVecTest, TEST);
		readData(WATER_LILLY, featuresTest, sizeVecTest, TEST);
		featuresTest = featuresTest * eigenT;

		//Store Data to file storage for faster use
		cv::FileStorage storage("test.yml", cv::FileStorage::WRITE);
		storage << "featuresTrain" << featuresTrain;
		storage << "sizeTrain" << sizeVec;
		storage << "featuresTest" << featuresTest;
		storage << "sizeTest" << sizeVecTest;
		storage.release();
	}

	else
	{
		//Read stored data
		cv::FileStorage storage1("test.yml", cv::FileStorage::READ);
		storage1["featuresTrain"] >> featuresTrain;
		storage1["sizeTrain"] >> sizeVec;
		storage1["featuresTest"] >> featuresTest;
		storage1["sizeTest"] >> sizeVecTest;
		storage1.release();
	}
	
	//Cluster features of all the test images using k-means clustering
	cout << "Clustering the input data using k-means algorithm... \n" << endl;
	kmeans(featuresTrain, CLUSTERS, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, ATTEMPTS, 0.0001), ATTEMPTS, KMEANS_PP_CENTERS,centers);

	//Find histogram of each training image based on labels of its features 
	calcHistograms(labels, sizeVec,histograms);

	//Compute labels for the features of test image based on euclidian distance from kmeans centers
	calcLabelsTest(featuresTest, centers,labelsTest);

	//Find histogram of each test image based on labels of its features 
	calcHistograms(labelsTest, sizeVecTest, histogramsTest);

	//Create response matrix for training knn object
	for (int i = 0; i < 20; i++)		response.push_back(1);
	for (int i = 20; i < 40; i++)		response.push_back(2);
	for (int i = 40; i < 60; i++)		response.push_back(3);
	for (int i = 60; i < 80; i++)		response.push_back(4);
	for (int i = 80; i < 100; i++)		response.push_back(5);

	//Train the knn object on histograms using response 
	kn.train(histograms, response);

	//Find the nearest neighbour for histograms of each test image
	kn.find_nearest(histogramsTest, KNEIGHBOURS, results, neighResponses,dists);

	printStats(results);
	cout << "\n*****Hit any key to EXIT*****" << endl;
	getchar();
}