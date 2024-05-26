#pragma once
#include<iostream>
#include<opencv.hpp>
#include<cmath>

using namespace cv;

class generate_mask
{
public:
	static void Generate_mask(const Mat& src, double* mask, const int msize, const double sigma);
};