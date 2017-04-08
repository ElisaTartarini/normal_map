// NormalMap.cpp : definisce il punto di ingresso dell'applicazione console.
//

/***************************************************************************
				Compute norm map from single image
1) Apply Sobel filter along x and y directions
2) Compose norm vector using Sobel components and a fixed z component
3) Normalize norm vector
4) Transform normalized norm vector into RGB components for norm map as follows:
	X: -1 to +1 :  Red: 0 to 255
	Y: -1 to +1 :  Green: 0 to 255
	Z:  0 to -1 :  Blue: 128 to 255
****************************************************************************/

#include "stdafx.h"
#include <opencv.hpp>

using namespace cv;

void computeNormals(Mat &xSobel, Mat &ySobel, Mat& normals, float strength, int blurSize, float gaussSigma);

int _tmain(int argc, _TCHAR* argv[])
{
	// Parameters
	std::string filename = "../images/rocks.jpg";			// Input image filename
	//std::string filename = "../images/sphere.png";		// Input image filename
	//std::string filename = "../images/sphere2.jpg";		// Input image filename
	float strength = 30;									// Normal Z component intensity
	int blurSize = 7;										// Kernel size for Gaussian blur
	float gaussSigma = 0;									// Sigma parameter for Gaussian blur (0 means automatically calculated fro kernel size

	// Read input image and show it
	cv::Mat source;
	source = imread(filename);
	imshow("input", source);

	// Transform image to grayscale
	Mat source_gray;
	cvtColor(source, source_gray, CV_BGR2GRAY);

	// Blur input image
	Mat blurred;
	GaussianBlur(source_gray, blurred, Size(blurSize, blurSize), gaussSigma);
	imshow("blurred", blurred);

	// Compute Sobel images
	Mat xSobel, ySobel;
	Sobel(blurred, xSobel, CV_32FC1, 1, 0, 5);
	Sobel(blurred, ySobel, CV_32FC1, 0, 1, 5);
	double minS, maxS;

	// Convert Sobel images in a viewable format and show them
	Mat xSobel_8, ySobel_8;
	minMaxLoc(xSobel, &minS, &maxS);
	xSobel.convertTo(xSobel_8, CV_8UC1, 255. / (maxS - minS), -minS * 255 / (maxS - minS));
	minMaxLoc(ySobel, &minS, &maxS);
	ySobel.convertTo(ySobel_8, CV_8UC1, 255. / (maxS - minS), -minS * 255 / (maxS - minS));
	imshow("xSobel", xSobel_8);
	imshow("ySobel", ySobel_8);

	// Compute normal vector and store it in the normal map
	Mat dest(source.size(), CV_8UC3);
	
	computeNormals(xSobel, ySobel, dest, strength, blurSize, gaussSigma);

	// Show and save norm map image
	imshow("dest", dest);
	imwrite("../images/normals.png", dest);

	waitKey(-1);

	source.release();
	source_gray.release();
	blurred.release();
	xSobel.release();
	ySobel.release();
	xSobel_8.release();
	ySobel_8.release();
	dest.release();

	return 0;
}


void computeNormals(Mat &xSobel, Mat &ySobel, Mat& normals, float strength, int blurSize, float gaussSigma)
{
	assert(xSobel.size() == ySobel.size()
		&& xSobel.size() == normals.size()
		&& xSobel.type() == CV_32FC1
		&& ySobel.type() == CV_32FC1
		&& normals.type() == CV_8UC3);

	Vec3f norm;
	// Init image pointers
	float* xSobel_ptr = (float*)xSobel.data;
	float* ySobel_ptr = (float*)ySobel.data;
	Vec3b* dest_ptr = (Vec3b*)normals.data;

	// Iterate over rows
	for (int r = 0; r < xSobel.rows; r++, xSobel_ptr += xSobel.cols, ySobel_ptr += ySobel.cols, dest_ptr += normals.cols)
	{
		// Init row pointers
		float* xSobel_row = xSobel_ptr;
		float* ySobel_row = ySobel_ptr;
		Vec3b* dest_row = dest_ptr;

		// Iterate over columns
		for (int c = 0; c < xSobel.cols; c++, xSobel_row++, ySobel_row++, dest_row++)
		{
			// Fill norm vector
			norm[0] = *xSobel_row;
			norm[1] = -*ySobel_row;
			norm[2] = strength;

			// Normalize norm vector in range [0,1]
			cv::normalize(norm, norm, 1, NORM_L2);

			// Compute norm RGB components
			norm /= 2.;
			norm += Vec3f(0.5, 0.5, 0.5);
			norm *= 255;

			// Write normal map image
			*dest_row = Vec3b((uchar)round(norm[0]), (uchar)round(norm[1]), (uchar)round(norm[2]));
		}
	}

	// Convert image format from BGR to RGB
	cvtColor(normals, normals, CV_BGR2RGB);

}