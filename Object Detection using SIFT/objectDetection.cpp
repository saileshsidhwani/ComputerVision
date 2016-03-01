#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>		
#include <opencv2/nonfree/features2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

//Number of iterations for RANSAC Algorithm
#define RANSACIterator 100;

//Difference of threshold allowed while comparing descriptors
#define PIXELThreshold 3;

Mat H, image1, image2 ;
vector<Point> cordinatesTestImg,cordinatesObjImg;


void findMatches(Mat descriptorsTestImg, Mat descriptorsObjImg, vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2)
{
vector<double> matches;
vector<int> index;

	for (int j = 0; j < descriptorsTestImg.rows; j++)
	{
		matches.clear();
		//Caculate distance of descriptor in IMG1 with each descriptor in IMG2
		for (int i = 0; i < descriptorsObjImg.rows; i++)
			matches.push_back(norm(descriptorsTestImg.row(j), descriptorsObjImg.row(i), NORM_L2));

		index.clear();
		//Sort  distances in ascending order
		sortIdx(matches, index, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);

		//Reject if matches are close. 
		if (matches[index[0]] < 0.75 * matches[index[1]])
		{
			cordinatesTestImg.push_back(keypoints1[j].pt);
			cordinatesObjImg.push_back(keypoints2[index[0]].pt);
		}
	}
}


void displayMatches()
{
	Size testImgSize = image1.size();
	Size objImgSize = image2.size();

	Mat image3(max(testImgSize.height, objImgSize.height), testImgSize.width + objImgSize.width, CV_LOAD_IMAGE_GRAYSCALE, Scalar::all(0));

	int diff1 = max(0, objImgSize.height - testImgSize.height);
	int diff2 = max(0, testImgSize.height - objImgSize.height);

	// Move right boundary to the left.
	image3.adjustROI(0, -diff1, 0, -objImgSize.width);
	image1.copyTo(image3);
	// Move the left boundary to the right, right boundary to the right.
	image3.adjustROI(0, diff1 - diff2, -testImgSize.width, objImgSize.width);
	image2.copyTo(image3);
	// restore original ROI.
	image3.adjustROI(0, diff2, testImgSize.width, 0);

	// draw lines connecting matched keypoints
	Point shift = Point(testImgSize.width, 0);
	for (unsigned i = 0; i < cordinatesTestImg.size(); i++)
		line(image3, cordinatesTestImg[i], cordinatesObjImg[i] + shift, CV_RGB(255, 0, 0));
	imshow("Matching Pairs", image3);
	imwrite("Image2&5_matching.jpg", image3);
}


Mat calculateHomographyMatrix()
{
	const int num = 4;
	Mat A(0, 8, CV_64F);
	Mat b(0, 1, CV_64F);
	Mat h(8, 1, CV_64F);
	vector<double> hVec;
	Mat H1 = Mat(3, 3, CV_64F);		

	//Linearizing X2 =H*X1
	for (int i = 0; i < num; i++)
	{
		//Select a random entry
		int random = rand() % cordinatesTestImg.size();

		Point p1 = cordinatesTestImg[random], p2 = cordinatesObjImg[random];
		//Compute and append rows of A, b
		b.push_back((double)p2.x);
		b.push_back((double)p2.y);
		Mat X1 = (Mat_<double>(1, 8) << p1.x, p1.y, 1, 0, 0, 0, (-1 * p1.x*p2.x), (-1 * p1.y*p2.x));
		A.push_back(X1);
		Mat X2 = (Mat_<double>(1, 8) << 0, 0, 0, p1.x, p1.y, 1, (-1 * p1.x*p2.y), (-1 * p1.y*p2.y));
		A.push_back(X2);
	}

	// Solve the equation ---> A*h = b 
	if (!solve(A, b, h))
	{
		H1.data = NULL;
		return H1;
	}

	//H33 = 1
	h.push_back((double)1);				

	//Convert H from 1x9 matrix to 3x3 matrix
	/*for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			H.at<double>(i,j) = h.at<double>(i + j);*/
	hVec.assign((double*)h.datastart, (double*)h.dataend);
	//Unroll the vector into a matrix  
	memcpy(H1.data, hVec.data(), hVec.size()*sizeof(double));

	return H1;
}

void countCorrectMatches(Mat H_temp, int *numMatches)
{
	double matches_threshold = PIXELThreshold;		
	for (unsigned i = 0; i < cordinatesTestImg.size(); i++) 
	{
		Point p1 = cordinatesTestImg[i];
		Point p2 = cordinatesObjImg[i];
		Mat P1_hom = (Mat_<double>(3, 1) << p1.x, p1.y, 1);		//Homogenous cordinate for P1
		Mat P2_hom = (Mat_<double>(3, 1) << p2.x, p2.y, 1);		//Homogenous cordinate for P2
		Mat p2Projection_hom = H_temp * P1_hom;					//HOMOGRAPHY
		
		double matches = norm(P2_hom, p2Projection_hom);

		if (matches < matches_threshold)						//Check threshold after projection
			(*numMatches)++;
	}
}

void RANSAC()
{
	int maxMatches = 0;
	int numMatches = 0;
	int iterator = RANSACIterator
	for (int i = 0; i < iterator ; i++)
	{
		Mat H_temp = calculateHomographyMatrix();		//Compute matrix values
		//if Homography matrix is NULL, try again
		if (!H_temp.data)							
			continue;

		//Count maximum number of correct matches found
		countCorrectMatches(H_temp,&numMatches);
		
		if (numMatches >= maxMatches)
		{
			H_temp.copyTo(H);
			maxMatches = numMatches;
		}
	}

}

void displayResult()
{

	vector<Point> coords2_est;

	for (unsigned i = 0; i < cordinatesTestImg.size(); i++) {
		Point p1 = cordinatesTestImg[i];

		Mat mP1_hom = (Mat_<double>(3, 1) << p1.x, p1.y, 1);
		Mat mP2_est_hom = H * mP1_hom;
		Point p2_est;
		p2_est.x = int(mP2_est_hom.at<double>(0, 0) / mP2_est_hom.at<double>(2, 0));
		p2_est.y = int(mP2_est_hom.at<double>(1, 0) / mP2_est_hom.at<double>(2, 0));
		coords2_est.push_back(p2_est);
	}

	Point p;
	//Draw estimated keypoints in black
	for (unsigned i = 0; i < coords2_est.size(); i++) {
		p = coords2_est[i];
		circle(image2, p, 2, Scalar(0, 0, 0), -1, 8);
	}

	//Draw original keypoints in white
	for (unsigned i = 0; i < cordinatesObjImg.size(); i++) {
		p = cordinatesObjImg[i];
		circle(image2, p, 2, Scalar(255, 255, 255), -1, 8);
	}
	imshow("Final Image", image2);
	imwrite("Image2&5_final.jpg", image2);

}

int main()
{
	SIFT sift;
	Mat descriptorsTestImg, descriptorsObjImg;
	vector<KeyPoint> keypoints1, keypoints2;
	InputArray mask = cv::noArray();
	
	//Read Test and object image
	image1 = imread("image_2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	image2 = imread("image_5.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	// Computing SIFT keypoints and descriptors
	sift.operator()(image1, mask, keypoints1, descriptorsTestImg);
	sift.operator()(image2, mask, keypoints2, descriptorsObjImg);

	//Match Descriptors of keypoints
	findMatches(descriptorsTestImg, descriptorsObjImg, keypoints1, keypoints2);
		
	//Display matching descriptors
	displayMatches();
		
	// RANSAC Algorithm
	RANSAC();
	cout << "\nHomogenous Matrix: " << H << endl ;
	
	// Project Final Keypoints in Object Image
	displayResult();

	waitKey(0);
	return 0;
}
