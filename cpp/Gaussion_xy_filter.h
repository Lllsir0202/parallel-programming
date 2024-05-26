#pragma once
#include<iostream>
#include<chrono>
#include<opencv2/opencv.hpp>
#include<cmath>
#include "Gaussion_xy_SIMD.h"
#include "xy_filter_openmp.h"
#include "generate_mask.h"
#include <pthread.h>
//#pragma comment(lib, "D:/pthreads-w32-2-9-1-release/Pre-built.2/lib/x64/pthreadVC2.lib")


using namespace cv;

class xy_filter:public simd , omp_filter
{
public:
	//@alignment refers to align bits
	void Gaussion_xy_filter(const Mat& src, Mat& res, const int msize, const double sigma, int alignment);
	void Gaussion_xy_filter_SSE(const Mat& src, Mat& res, const int msize, const double sigma, int alignment);
	void Gaussion_xy_filter_AVX(const Mat& src, Mat& res, const int msize, const double sigma, int alignment);
	void Gaussion_xy_filter_openmp_SSE(const Mat& src, Mat& res, const int msize, const double sigma, int alignment);
	void Gaussion_xy_filter_openmp_AVX(const Mat& src, Mat& res, const int msize, const double sigma, int alignment);
	void Gaussion_xy_filter_openmp(const Mat& src, Mat& res, const int msize, const double sigma, int alignment, int thread_num);
	void Gaussion_xy_filter_pthread(const Mat& src, Mat& res, const int msize, const double sigma, int alignment);
	//@row col refers to src rows and cols
	//@mask refers to one_dimension window
	//@need to use twice
	void process(uchar** matrix, uchar** tmp , uchar** res , double* mask, const int row, const int col, const int border);
};