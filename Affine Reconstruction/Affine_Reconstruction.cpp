#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

float shiftX = 0, shiftY = 0;
int main()
{
	int imgNum = 1;

	//given input data points
	/*Row 0 corroosponds to x-cordinates of Image1 and Row 1 is y-cordinate of Image1*/
	/*Row 2 corroosponds to x-cordinates of Image2 and Row 3 is y-cordinate of Image2 and so on*/
	float data[12][8] = {	{ 227, 261, 339, 299, 194, 227, 303, 265 },
							{ 341, 400, 277, 218, 341, 402, 275, 214 },
							{ 150, 112, 192, 234, 175, 137, 214, 256 },
							{ 208, 197, 83, 88, 228, 220, 105, 111 },
							{ 27, 67, 133, 95, 48, 86, 156, 117 },
							{ 225, 225, 126, 123, 247, 246, 147, 145 },
							{ 48, 111, 130, 62, 38, 102, 117, 48 },
							{ 33, 55, 71, 46, 64, 86, 105, 81 },
							{ 85, 126, 235, 197, 76, 117, 227, 188 },
							{ 93, 97, 133, 131, 126, 130, 168, 166 },
							{ 227, 253, 81, 66, 228, 253, 85, 67 },
							{ 52, 52, 57, 57, 93, 95, 102, 100 } };
	
	//given sequence of connectivity of the image points
	int sequence[12][2] = {
			{ 1, 2 }, { 2, 3 }, { 3, 4 }, { 4, 1 }, { 5, 6 }, { 6, 7 }, { 7, 8 }, { 8, 5 }, { 1, 5 }, { 2, 6 }, { 3, 7 }, { 4, 8 }};

	
	
	Mat D(12, 8, CV_32F, &data);
	cout << "Enter the image number to draw affine projection of (1 through 6) : " << endl;
	cin >> imgNum;
	if (imgNum > 6 || imgNum < 1)
		imgNum = 1;

	//Extract image points of required image
	Mat imgMat = D(Rect(0,(imgNum*2)-2,8,2));
	Mat imgPoints = imgMat.t();
	
	//Create a blank white image
	Mat original(512,512,CV_LOAD_IMAGE_COLOR,Scalar(255,255,255));
 
	//Draw the original image in given sequence
	for (int i = 0; i < 12; i++)
	{
			int temp1 = sequence[i][0];
			int temp2 = sequence[i][1];
			Point p1, p2;
			p1.x = imgPoints.at<float>(temp1 - 1, 0);
			p1.y = imgPoints.at<float>(temp1 - 1, 1);
			p2.x = imgPoints.at<float>(temp2 - 1, 0);
			p2.y = imgPoints.at<float>(temp2 - 1, 1);
			line(original, p1, p2, Scalar(0,0,0), 1);
	}
	imshow("Original_image", original);
	waitKey(0);

	//Centering image data
	//Calculate mean of each row and subtract it from all the elements
	for (int i = 0; i < 12; i++)
	{
		float sum = 0;
		for (int j = 0; j < 8; j++)
			sum += data[i][j];
	
		sum = sum / 8;

		if (i == ((imgNum * 2) - 2))
			shiftX = sum;

		if (i == ((imgNum * 2) - 1))
			shiftY = sum;
		
		for (int j = 0; j < 8; j++)
			data[i][j] = data[i][j] - sum;
	}

	Mat D1(12, 8, CV_32F, &data);			//new data matrix with centered data
	Mat U, W, Vt;
	SVD::compute(D1, W, U, Vt);				//Computing SVD
	
	Mat U3 = U(Rect(0, 0, 3, 12));			//Computing U3 matrix
	Mat Vt3 = Vt(Rect(0, 0, 8, 3));			//Computing Vt3 matrix
	Mat W3(3, 3, CV_32F, float(0));
	
	//Computing W3 matrix fom vector W
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			if (i == j)
				W3.at<float>(i, j) = W.at<float>(i, 0);

	Mat sqrtW3(3, 3, CV_32F);
	Mat A0(12, 3, CV_32F);
	Mat P0(3, 8, CV_32F);
	
	sqrt(W3, sqrtW3);				//Calculating sqrt(W3)
	A0 = U3 * sqrtW3;				//Calculating A0 matrix. Post-multiply U3 with Sqrt(W3)
	P0 = sqrtW3 * Vt3;				//Calculating P0 matrix. Pre-multiply Vt3 with Sqrt(W3)
	cout << "Motion matrix (A0): " << endl;
	cout << A0 << endl;

	cout << "Shape Matrix (3D points)  : " << endl;
	cout << P0 << endl;
	Mat projectedPoints(8, 2, CV_32F,float(0));

	//Extract A matrix for requred image from A0
	Mat A_img = A0(Rect(0, (imgNum*2)-2, 3, 2));
	
	//Matrix P0 is a 3x8 matrix. So each column is a single 3D point.
	//Select each point i.e. Mat P, multiply it with corossponding images A matrix to get projected 2D point
	//Store all the 2D points in the projectedPoints Mat, in format as each row would be each 2D point
	for (int i = 0;  i < 8; i++)
	{
		Mat P = P0(Rect(i,0, 1, 3));
		Mat projection = A_img * P;
		projectedPoints.at<float>(i, 0) = projection.at<float>(0, 0);
		projectedPoints.at<float>(i, 1) = projection.at<float>(1, 0);
	}

	//cout << "projectedPoints" << projectedPoints << endl;
	

	//Readjust center of projected 2d points with factor of original transformation value
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 8; j++)
		{
			if (i == 0)
				projectedPoints.at<float>(j, i) += shiftX; 
			if (i == 1)
				projectedPoints.at<float>(j, i) += shiftY;
		}
	}


	//Drawing the projected image from SVD
	for (int i = 0; i < 12; i++)
	{
		int temp1 = sequence[i][0];
		int temp2 = sequence[i][1];
		Point p1, p2;
		p1.x = projectedPoints.at<float>(temp1 - 1, 0);
		p1.y = projectedPoints.at<float>(temp1 - 1, 1);
		p2.x = projectedPoints.at<float>(temp2 - 1, 0);
		p2.y = projectedPoints.at<float>(temp2 - 1, 1);
		line(original, p1, p2, Scalar(0, 0, 0), 1);
	}
	imshow("projected", original);
	string filename = "Camera";
	filename += to_string(imgNum);
	filename += "_affine.jpg";
	imwrite(filename, original);
	
	waitKey(0);
}
