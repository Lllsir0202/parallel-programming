#pragma once
#include<tmmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>
#include<immintrin.h>
#include<iostream>
#include<opencv.hpp>
using namespace cv;

class simd
{
public:
	//@row col refers to src rows and cols
	//@mask refers to one_dimension window
	//@need to use twice
	void process_SSE(uchar** matrix, uchar** tmp ,uchar** res, double* mask, const int row, const int col, const int border);
	void process_AVX(uchar** matrix, uchar** tmp, uchar** res, double* mask, const int row, const int col, const int border);
};