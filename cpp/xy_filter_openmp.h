#pragma once
#include<iostream>
#include<cmath>
#include<opencv.hpp>

using namespace cv;


class omp_filter
{
public:
	void process_omp_row(uchar** matrix, uchar** tmp, double* mask, const int start_row, const int end_row, const int width, const int height, const int border);
	void process_omp_col(uchar** tmp, uchar** res, double* mask, const int start_row, const int end_row, const int width, const int height, const int border);
};