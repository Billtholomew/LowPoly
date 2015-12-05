#include "stdafx.h"
#include <iostream>
#include <ctime>
#include <omp.h>

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

void fillDelaunay(Vec6f triangle, Mat oim, Mat nim) {
	vector<Point> verts(3);
	verts[0] = Point(cvRound(triangle[0]), cvRound(triangle[1]));
	verts[1] = Point(cvRound(triangle[2]), cvRound(triangle[3]));
	verts[2] = Point(cvRound(triangle[4]), cvRound(triangle[5]));

	if (abs(verts[0].x) > oim.cols || abs(verts[0].y) > oim.rows ||
		abs(verts[1].x) > oim.cols || abs(verts[1].y) > oim.rows ||
		abs(verts[2].x) > oim.cols || abs(verts[2].y) > oim.rows)
		return;

	// Make a copy of the ROI that is all black
	Mat ROI = Mat::zeros(oim.size[0], oim.size[1], CV_8U);
	// Fill in the new ROI
	fillConvexPoly(ROI, verts, { 255 });
	// Get the mean of the values in the ROI
	Scalar color = mean(oim, ROI);
	// Paint in the ROI of the new image
	fillConvexPoly(nim, verts, color);
}

Mat drawDelaunay(Subdiv2D delaunay, Mat pts, Mat oim, bool par=false) {
	vector<Vec6f> triangleList;
	delaunay.getTriangleList(triangleList);
	Mat nim;
	oim.copyTo(nim);

	Mat BROI;
	pts.copyTo(BROI);
	BROI.setTo(Scalar({0}));
	
	clock_t begin = clock();
	if (par) 
	{	
		int i;
		omp_set_num_threads(4);
		#pragma omp parallel for private(i)
		for (i = 0; i < triangleList.size(); i++)
			fillDelaunay(triangleList[i], oim, nim);
	}
	else 
	{
		for (size_t i = 0; i < triangleList.size(); i++)
		{	
			Vec6f t = triangleList[i];
			fillDelaunay(t, oim, nim);
		}
	}
	clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	std::cout << elapsed_secs << std::endl;
	return nim;
}

Mat getTriangulation(Mat oim, int minT=10, int maxT=55, float ratio=0.05, bool debug = true) {
	Mat pts;
	//blur(oim, oim, Size(9, 9));
	Canny(oim, pts, minT, maxT);
	Mat R;
	pts.copyTo(R);
	randu(R, Scalar::all(0), Scalar::all(255));
	pts = pts.mul((R < ratio*255));
	
	Rect rect(0, 0, pts.size[1], pts.size[0]);
	Subdiv2D delaunay;
	delaunay.initDelaunay(rect);
	findNonZero(pts, R);
	delaunay.insert(R);
	// Get corner cases
	delaunay.insert(Point(0, 0));
	delaunay.insert(Point(0, pts.size[0]-1));
	delaunay.insert(Point(pts.size[1]-1, 0));
	delaunay.insert(Point(pts.size[1]-1, pts.size[0]-1));
	Mat out = drawDelaunay(delaunay, pts, oim, true);
	return out;
}

int fromCamera() {
	CvCapture* capture = 0;
	Mat frame;

	capture = cvCaptureFromCAM(0); //0=default, -1=any camera, 1..99=your camera
	if (!capture) 
		std::cout << "No camera detected" << std::endl;

	cvNamedWindow("MyVideo", CV_WINDOW_AUTOSIZE);

	if (capture)
	{
		std::cout << "In capture ..." << std::endl;
		for (;;)
		{
			IplImage* iplImg = cvQueryFrame(capture);
			frame = iplImg;
			if (frame.empty())
				break;
			int len = max(frame.size[0], frame.size[1]);
			double scale = 500.0 / len;
			resize(frame, frame, Size(0, 0), scale, scale);
			frame = getTriangulation(frame);
			resize(frame, frame, Size(0, 0), 1/scale, 1/scale);
			IplImage out = frame;
			cvShowImage("MyVideo", &out); //show the frame in "MyVideo" window

			if (waitKey(5) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
			{
				std::cout << "esc key is pressed by user" << std::endl;
				break;
			}
		}

		cvDestroyWindow("MyVideo");

		return 0;
	}
	return 0;
}

int fromFile(const char* fName) {
	Mat image;
	image = imread(fName);   // Read the file
	std::cout << fName << std::endl;
	if (!image.data)
	{
		std::cout << "Could not open or find the image" << std::endl;
	}
	int len = max(image.size[0], image.size[1]);
	double scale = 500.0 / len;
	resize(image, image, Size(0, 0), scale, scale);
	image = getTriangulation(image);
	cvNamedWindow("Lo-Poly", WINDOW_AUTOSIZE);
	IplImage out = image;
	cvShowImage("Lo-Poly", &out);
	waitKey(0);
	cvDestroyWindow("Lo-Poly");
	//imwrite("C:/Users/William/Desktop/Art/pel_serial.png",image);
	return 0;
}

int main(int argc, const char** argv)
{
	//return fromFile("C:/Users/William/Desktop/Art/Lenna.jpg");
	return fromCamera();
}