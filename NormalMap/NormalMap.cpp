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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace cv;

/**
 * User data to pass to trackbar callback functions
 */
struct TrackbarCallbackData {
	Mat* source;
	Mat* xSobel;
	Mat* ySobel;
	Mat* normals;
	int* strength;
	int* blurSize;
	float gaussSigma;
	std::string outWin;
};

void computeNormals(Mat &xSobel, Mat &ySobel, Mat& normals, int strength);

// Trackbar callbacks
void recomputeNormals(int, void*);
void recomputeBlurAndNormals(int, void*);

int main(int argc, char* argv[])
{
        if(argc <= 1)
        {
            std::cout << "Usage: %s [image-filename]" << argv[0] << std::endl;
            return -1;
        }
        std::string filename = argv[1];
    
	// Parameters
	//std::string filename = "../images/rocks.jpg";			// Input image filename
	//std::string filename = "../images/sphere.png";		// Input image filename
	//std::string filename = "../images/sphere2.jpg";		// Input image filename
	int strength = 300;									// Normal Z component intensity
	int blurSize = 31;										// Kernel size for Gaussian blur
	float gaussSigma = 0;									// Sigma parameter for Gaussian blur (0 means automatically calculated fro kernel size

	std::string resultWin = "Normals";
	Mat source, source_gray, blurred;
	Mat xSobel, ySobel;
	//Mat xSobel_8, ySobel_8;
	Mat dest;

	// Read input image and show it
	source = imread(filename);
        if(source.empty())
        {
            std::cout << "Invalid image file: %s" << filename << std::endl;
        }
	imshow("input", source);

	// Create result image
	dest.create(source.size(), CV_8UC3);

	// Prepare result windows
	namedWindow(resultWin, CV_WINDOW_NORMAL);
        std::cout << "Press a key to exit..." << std::endl;

	// Fill userdata and create trackbars
	TrackbarCallbackData data;
	data.source = &source_gray;
	data.xSobel = &xSobel;
	data.ySobel = &ySobel;
	data.normals = &dest;
	data.strength = &strength;
	data.blurSize = &blurSize;
	data.gaussSigma = gaussSigma;
	data.outWin = resultWin;
	createTrackbar("Depth strength", resultWin, &strength, 1000, recomputeNormals, &data);
	createTrackbar("Blur kernel", resultWin, &blurSize, 60, recomputeBlurAndNormals, &data);

	// Transform image to grayscale
	cvtColor(source, source_gray, CV_BGR2GRAY);

	// Blur input image
	GaussianBlur(source_gray, blurred, Size(blurSize, blurSize), gaussSigma);
	//imshow("blurred", blurred);

	// Compute Sobel images
	Sobel(blurred, xSobel, CV_32FC1, 1, 0, 5);
	Sobel(blurred, ySobel, CV_32FC1, 0, 1, 5);

	// Convert Sobel images in a viewable format and show them
	//double minS, maxS;
	//minMaxLoc(xSobel, &minS, &maxS);
	//xSobel.convertTo(xSobel_8, CV_8UC1, 255. / (maxS - minS), -minS * 255 / (maxS - minS));
	//minMaxLoc(ySobel, &minS, &maxS);
	//ySobel.convertTo(ySobel_8, CV_8UC1, 255. / (maxS - minS), -minS * 255 / (maxS - minS));
	//imshow("xSobel", xSobel_8);
	//imshow("ySobel", ySobel_8);

	// Compute normal vector and store it in the normal map	
	computeNormals(xSobel, ySobel, dest, strength);

	// Show norm map image
	imshow(resultWin, dest);

	waitKey(-1);

	// Save norm map image
        std::string outFile = filename.substr(0,filename.find_last_of('/')+1)+"normals.png";
	imwrite(outFile, dest);
        std::cout << "Result written to: " << outFile << std::endl;
        
	source.release();
	source_gray.release();
	blurred.release();
	xSobel.release();
	ySobel.release();
	//xSobel_8.release();
	//ySobel_8.release();
	dest.release();

	return 0;
}

/**
 * Compute normal map given x-Sobel and y-Sobel derivatives and z strength
 */
void computeNormals(Mat &xSobel, Mat &ySobel, Mat& normals, int strength)
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
			norm[0] = -*xSobel_row;
			norm[1] = *ySobel_row;
			norm[2] = (float)strength;

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

/**
* Re-compute normal map using new data updated in user data param
*/
void recomputeNormals(int, void* data_)
{
	TrackbarCallbackData* data = (TrackbarCallbackData*)data_;

	// Compute normal vector and store it in the normal map	
	computeNormals(*data->xSobel, *data->ySobel, *data->normals, *data->strength);

	// Update normals window
	imshow(data->outWin, *data->normals);
}

/**
* Re-compute Gaussian blur of input image, Sobel derivatives and normal map using new data updated in user data param
*/
void recomputeBlurAndNormals(int, void* data_)
{
	TrackbarCallbackData* data = (TrackbarCallbackData*)data_;

	Mat blurred;

	// Avoid exceptions
	if (*data->blurSize % 2 == 0)
		(*data->blurSize)++;
	
	// Blur input image
	GaussianBlur(*data->source, blurred, Size(*data->blurSize, *data->blurSize), data->gaussSigma);

	// Compute Sobel images
	Sobel(blurred, *data->xSobel, CV_32FC1, 1, 0, 5);
	Sobel(blurred, *data->ySobel, CV_32FC1, 0, 1, 5);

	// Compute normal vector and store it in the normal map	
	computeNormals(*data->xSobel, *data->ySobel, *data->normals, *data->strength);

	// Update normals window
	imshow(data->outWin, *data->normals);

	blurred.release();
}
