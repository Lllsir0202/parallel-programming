#pragma once
#include<iostream>
#include<chrono>
#include<opencv2/opencv.hpp>
#include<cmath>

using namespace cv;

class xy_filter
{
public:
	//@alignment refers to align bits
	void Gaussion_xy_filter(const Mat& src, Mat& res, const int msize, const double sigma, int alignment);

	//@row col refers to src rows and cols
	//@mask refers to one_dimension window
	//@need to use twice
	void process(uchar** matrix, double* mask, const int row, const int col, const int border);
};