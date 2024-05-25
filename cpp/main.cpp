#pragma once
#include<iostream>
#include<cmath>
#include<cstdint>
#include<opencv2/opencv.hpp>
#include<tmmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>
#include<immintrin.h>
#include <pthread.h>
//#include<pthread.h>
#include <omp.h>
//#include <pthread.h>
#include<chrono>
#include"Gaussion_xy_filter.h"
#include"generate_mask.h"
using namespace cv;

//二维
void Generate_Gaussion_Mask(Mat& mask, Size msize, double sigma)
{
	mask.create(msize, CV_64F);

	//h->height of mask 
	//w->width of mask
	int h = msize.height;
	int w = msize.width;

	//center_w -> center_x of mask
	//center_h -> center_y of mask
	int center_w = (w - 1) / 2;
	int center_h = (h - 1) / 2;

	double sum = 0.0;
	double x, y;
	double g = 0.0;
	for (int i = 0; i < h; i++)
	{
		y = i - center_h;
		y *= y;
		for (int j = 0; j < w; j++)
		{
			x = j - center_w;
			x *= x;
			g = exp(-(x + y) / (2 * sigma * sigma));
			mask.at<double>(i, j) = g;
			sum += g;
		}
	}
	mask = mask / sum;
}

//一维
void Generate_Gaussion_Mask(Mat& mask, int msize, double sigma)
{
	mask.create(1,msize, CV_64F);
	
	int center = (msize - 1) / 2;

	double sum = 0.0;
	double elem;
	double g = 0.0;
	for (int i = 0; i < msize; i++)
	{
		elem = i - center;
		elem *= elem;
		g = exp(-elem / (2 * sigma * sigma));
		mask.at<double>(0,i) = g;
		sum += g;
	}
	mask /= sum;

}

void Gaussion_simp(const Mat& src, Mat& res, const Mat& mask)
{
	//mh -> height of mask 1/2
	//mw -> width of mask 1/2
	//用来设置补充边界的宽度和高度
	int mh = (mask.cols - 1) / 2;
	int mw = (mask.rows - 1) / 2;

	//先将res赋值为0
	res = Mat::zeros(src.size(), src.type());

	Mat new_src;
	copyMakeBorder(src, new_src, mh, mh, mw, mw, BORDER_REPLICATE);

	//进行滤波
	auto start_clock = std::chrono::high_resolution_clock::now();


	for (int i = mh; i < src.rows + mh; i++)
	{
		for (int j = mw; j < src.cols + mw; j++)
		{
			double sum[3] = { 0 };
			for (int r = -mh; r <= mh; r++)
			{
				for (int c = -mw; c <= mw; c++)
				{
					if (src.channels() == 1)
					{
						sum[0] += new_src.at<uchar>(i + r, j + c) * mask.at<double>(r + mh, c + mw);
					}
					else if(src.channels() == 3)
					{
						Vec3b rgb = new_src.at<Vec3b>(i + r, j + c);
						sum[0] += rgb[0] * mask.at<double>(r + mh, c + mw); //B
						sum[1] += rgb[1] * mask.at<double>(r + mh, c + mw); //G
						sum[2] += rgb[2] * mask.at<double>(r + mh, c + mw); //R
					}
				}
			}

			if (new_src.channels() == 3)
			{
				for (int i = 0; i < 3; i++)
				{
					if (sum[i] < 0)
					{
						sum[i] = 0;
					}
					if (sum[i] > 255)
					{
						sum[i] = 255;
					}
				}
			}

			if (new_src.channels() == 1)
			{
				res.at<uchar>(i - mh, j - mw) = sum[0];
			}
			else if (new_src.channels() == 3)
			{
				Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
				res.at<Vec3b>(i - mh, j - mw) = rgb;
			}
		}
	}

	auto finish_clock = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds> (finish_clock - start_clock);
	std::cout << double(duration.count()) << std::endl;
}

void Gaussion_xy(const Mat& src, Mat& res, const int msize , const double sigma)
{
	Mat mask;
	Mat new_src;
	std::vector<Mat> channels_;

	//生成一维核
	Generate_Gaussion_Mask(mask, msize, sigma);
	//先横向处理

	auto start_clock = std::chrono::high_resolution_clock::now();


	int border = (msize - 1) / 2;
	res = Mat::zeros(src.size(), src.type());
	copyMakeBorder(src, new_src, border, border, border, border, BORDER_REPLICATE);


	for (int i = border; i < src.rows + border; i++)
	{
		for (int j = border; j < src.cols + border; j++)
		{
			double sum[3] = { 0 };
			for (int r = -border; r <= border; r++)
			{
				if (src.channels() == 1)
				{
					sum[0] += new_src.ptr<uchar>(i)[j + r] * mask.ptr<double>(0)[r + border];
					//sum[0] += new_src.at<uchar>(i, j + r) * mask.at<double>(0, r + border);
				}
				else if (src.channels() == 3)
				{
					split(new_src, channels_);
					double rgb_0 = channels_.at(0).ptr<uchar>(0)[r + border];
					double rgb_1 = channels_.at(1).ptr<uchar>(0)[r + border];
					double rgb_2 = channels_.at(2).ptr<uchar>(0)[r + border];
					sum[0] += rgb_0 * mask.at<double>(0, r + border);
					sum[1] += rgb_1 * mask.at<double>(0, r + border);
					sum[2] += rgb_2 * mask.at<double>(0, r + border);
				}
			}
			if (new_src.channels() == 3)
			{
				for (int i = 0; i < 3; i++)
				{
					if (sum[i] < 0)
					{
						sum[i] = 0;
					}
					if (sum[i] > 255)
					{
						sum[i] = 255;
					}
				}
			}

			if (new_src.channels() == 1)
			{
				res.at<uchar>(i - border, j - border) = sum[0];
			}
			else if (new_src.channels() == 3)
			{
				Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
				res.at<Vec3b>(i - border, j - border) = rgb;
			}
		}
	}

	copyMakeBorder(res, new_src, border, border, border, border, BORDER_REPLICATE);
	for (int i = border; i < src.rows + border; i++)
	{
		for (int j = border; j < src.cols + border; j++)
		{
			double sum[3] = { 0 };
			for (int c = -border; c <= border; c++)
			{
				if (src.channels() == 1)
				{
					sum[0] += new_src.ptr<uchar>(i + c)[j] * mask.ptr<double>(0)[c + border];
					//sum[0] += new_src.at<uchar>(i + c, j) * mask.at<double>(0, c + border);
				}
				else if (src.channels() == 3)
				{
					split(new_src, channels_);
					double rgb_0 = channels_.at(0).ptr<uchar>(0)[c + border];
					double rgb_1 = channels_.at(1).ptr<uchar>(0)[c + border];
					double rgb_2 = channels_.at(2).ptr<uchar>(0)[c + border];
					//Vec3b rgb = new_src.at<Vec3b>(i + c, j);
					sum[0] += rgb_0 * mask.at<double>(0, c + border);
					sum[1] += rgb_1 * mask.at<double>(0, c + border);
					sum[2] += rgb_2 * mask.at<double>(0, c + border);
				}
			}
			if (new_src.channels() == 3)
			{
				for (int i = 0; i < 3; i++)
				{
					if (sum[i] < 0)
					{
						sum[i] = 0;
					}
					if (sum[i] > 255)
					{
						sum[i] = 255;
					}
				}
			}

			if (new_src.channels() == 1)
			{
				res.at<uchar>(i - border, j - border) = sum[0];
			}
			else if (new_src.channels() == 3)
			{
				Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
				res.at<Vec3b>(i - border, j - border) = rgb;
			}

		}
	}
	auto finish_clock = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds> (finish_clock - start_clock);
	std::cout << "xy分离情况"<< double(duration.count()) << std::endl;
}


void Gaussion_xy_SSE(const Mat& src, Mat& res, const int msize, const double sigma)
{
	Mat mask;
	Mat new_src;
	std::vector<Mat> channels_;

	Generate_Gaussion_Mask(mask, msize, sigma);

	auto start_clock = std::chrono::high_resolution_clock::now();

	int border = (msize - 1) / 2;
	res = Mat::zeros(src.size(), src.type());

	// 使用对齐内存分配函数分配对齐的内存
	float* aligned_new_src = static_cast<float*>(_aligned_malloc((new_src.cols + 2 * border) * src.channels() * sizeof(float), 16));
	if (aligned_new_src == nullptr)
	{
		// 内存分配失败的处理
		return;
	}

	copyMakeBorder(src, new_src, border, border, border, border, BORDER_REPLICATE);

	for (int i = border; i < src.rows + border; i++)
	{
		for (int j = border; j < src.cols + border; j++)
		{
			__m128 sum[3] = { _mm_setzero_ps() };

			for (int r = -border; r <= border; r++)
			{
				if (src.channels() == 1)
				{
					// 使用对齐的加载操作
					__m128 src_val = _mm_load_ps(aligned_new_src + (i * new_src.cols + j + r));

					__m128 mask_val = _mm_set1_ps(mask.at<float>(0, r + border));

					sum[0] = _mm_add_ps(sum[0], _mm_mul_ps(src_val, mask_val));
				}
				else if (src.channels() == 3)
				{
					split(new_src, channels_);

					// 使用对齐的加载操作
					__m128 rgb_0 = _mm_load_ps(channels_.at(0).ptr<float>(0) + (i * new_src.cols + j + r));
					__m128 rgb_1 = _mm_load_ps(channels_.at(1).ptr<float>(0) + (i * new_src.cols + j + r));
					__m128 rgb_2 = _mm_load_ps(channels_.at(2).ptr<float>(0) + (i * new_src.cols + j + r));

					__m128 mask_val = _mm_set1_ps(mask.at<float>(0, r + border));

					sum[0] = _mm_add_ps(sum[0], _mm_mul_ps(rgb_0, mask_val));
					sum[1] = _mm_add_ps(sum[1], _mm_mul_ps(rgb_1, mask_val));
					sum[2] = _mm_add_ps(sum[2], _mm_mul_ps(rgb_2, mask_val));
				}
			}

			if (new_src.channels() == 3)
			{
				for (int i = 0; i < 3; i++)
				{
					sum[i] = _mm_max_ps(sum[i], _mm_setzero_ps());
					sum[i] = _mm_min_ps(sum[i], _mm_set1_ps(255.0f));
				}
			}

			if (new_src.channels() == 1)
			{
				res.at<float>(i - border, j - border) = _mm_cvtss_f32(sum[0]);
			}
			else if (new_src.channels() == 3)
			{
				Vec3f rgb = { _mm_cvtss_f32(sum[0]), _mm_cvtss_f32(sum[1]), _mm_cvtss_f32(sum[2]) };
				res.at<Vec3f>(i - border, j - border) = rgb;
			}
		}
	}

	// 释放对齐内存
	_aligned_free(aligned_new_src);

	auto finish_clock = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds> (finish_clock - start_clock);
	std::cout << "SSE优化情况: " << double(duration.count()) << std::endl;
}





void Gaussion_xy_omp(const Mat& src, Mat& res, const int msize, const double sigma)
{
	Mat mask;
	Mat new_src;
	std::vector<Mat> channels_;
	// Generating the 1D mask
	Generate_Gaussion_Mask(mask, msize, sigma);

	// Padding the source image
	int border = (msize - 1) / 2;
	copyMakeBorder(src, new_src, border, border, border, border, BORDER_REPLICATE);

	// Initializing the result image
	res = Mat::zeros(src.size(), src.type());
	int sum = 0;
#pragma omp parallel num_threads(4)
	{
		int num_threads = omp_get_num_threads();
		//std::cout << num_threads << std::endl;
		int thread_id = omp_get_thread_num();
		int rows_per_thread = src.rows / num_threads;
		int start_row = thread_id * rows_per_thread;
		int end_row = (thread_id == num_threads - 1) ? src.rows : start_row + rows_per_thread;
		for (int i = 0; i < 1000000; i++)
		{
			sum += i;
		}
	}
	auto start_clock = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(4)
	{
		int num_threads = omp_get_num_threads();
		//std::cout << num_threads << std::endl;
		int thread_id = omp_get_thread_num();

		// Divide the rows among threads
		int rows_per_thread = src.rows / num_threads;
		int start_row = thread_id * rows_per_thread;
		int end_row = (thread_id == num_threads - 1) ? src.rows : start_row + rows_per_thread;

		// Process the assigned rows
		for (int i = start_row + border; i < end_row + border; i++)
		{
			for (int j = border; j < src.cols + border; j++)
			{
				double sum[3] = { 0 };
				for (int r = -border; r <= border; r++)
				{
					if (src.channels() == 1)
					{
						sum[0] += new_src.ptr<uchar>(i)[j + r] * mask.ptr<double>(0)[r + border];
						//sum[0] += new_src.at<uchar>(i, j + r) * mask.at<double>(0, r + border);
					}
					else if (src.channels() == 3)
					{
						split(new_src, channels_);
						double rgb_0 = channels_.at(0).ptr<double>(0)[r + border];
						double rgb_1 = channels_.at(1).ptr<double>(0)[r + border];
						double rgb_2 = channels_.at(2).ptr<double>(0)[r + border];
						sum[0] += rgb_0 * mask.at<double>(0, r + border);
						sum[1] += rgb_1 * mask.at<double>(0, r + border);
						sum[2] += rgb_2 * mask.at<double>(0, r + border);
					}
				}
				if (new_src.channels() == 3)
				{
					for (int i = 0; i < 3; i++)
					{
						if (sum[i] < 0)
						{
							sum[i] = 0;
						}
						if (sum[i] > 255)
						{
							sum[i] = 255;
						}
					}
				}

				if (new_src.channels() == 1)
				{
					res.at<uchar>(i - border, j - border) = sum[0];
				}
				else if (new_src.channels() == 3)
				{
					Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
					res.at<Vec3b>(i - border, j - border) = rgb;
				}
			}
		}
	}

	// Padding the result image
	copyMakeBorder(res, new_src, border, border, border, border, BORDER_REPLICATE);

#pragma omp parallel num_threads(4)
	{
		int num_threads = omp_get_num_threads();
		int thread_id = omp_get_thread_num();

		// Divide the rows among threads
		int rows_per_thread = src.rows / num_threads;
		int start_row = thread_id * rows_per_thread;
		int end_row = (thread_id == num_threads - 1) ? src.rows : start_row + rows_per_thread;

		// Process the assigned rows
		for (int i = start_row + border; i < end_row + border; i++)
		{
			for (int j = border; j < src.cols + border; j++)
			{
				double sum[3] = { 0 };
				uchar* matrix = new_src.data;
				int height = new_src.rows;
				int width = new_src.cols;
				for (int c = -border; c <= border; c++)
				{
					if (src.channels() == 1)
					{
						sum[0] += new_src.ptr<uchar>(i + c)[j] * mask.ptr<double>(0)[c + border];
						//sum[0] += new_src.at<uchar>(i + c, j) * mask.at<double>(0, c + border);
					}
					else if (src.channels() == 3)
					{
						split(new_src, channels_);
						double rgb_0 = channels_.at(0).ptr<double>(0)[c + border];
						double rgb_1 = channels_.at(1).ptr<double>(0)[c + border];
						double rgb_2 = channels_.at(2).ptr<double>(0)[c + border];
						//Vec3b rgb = new_src.at<Vec3b>(i + c, j);
						sum[0] += rgb_0 * mask.at<double>(0, c + border);
						sum[1] += rgb_1 * mask.at<double>(0, c + border);
						sum[2] += rgb_2 * mask.at<double>(0, c + border);
					}
				}
				if (new_src.channels() == 3)
				{
					for (int i = 0; i < 3; i++)
					{
						if (sum[i] < 0)
						{
							sum[i] = 0;
						}
						if (sum[i] > 255)
						{
							sum[i] = 255;
						}
					}
				}

				if (new_src.channels() == 1)
				{
					res.at<uchar>(i - border, j - border) = sum[0];
				}
				else if (new_src.channels() == 3)
				{
					Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
					res.at<Vec3b>(i - border, j - border) = rgb;
				}
			}
		}
	}

	auto finish_clock = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds> (finish_clock - start_clock);
	std::cout << "omp优化" << double(duration.count()) << std::endl;
}

void Gaussion_xy_omp_SIMD(const Mat& src, Mat& res, const int msize, const double sigma)
{
	Mat mask;
	Mat new_src;

	// Generating the 1D mask
	Generate_Gaussion_Mask(mask, msize, sigma);

	// Padding the source image
	int border = (msize - 1) / 2;
	copyMakeBorder(src, new_src, border, border, border, border, BORDER_REPLICATE);

	// Initializing the result image
	res = Mat::zeros(src.size(), src.type());

	auto start_clock = std::chrono::high_resolution_clock::now();

#pragma omp parallel num_threads(4)
	{
		int num_threads = omp_get_num_threads();
		//std::cout << num_threads << std::endl;
		int thread_id = omp_get_thread_num();

		// Divide the rows among threads
		int rows_per_thread = src.rows / num_threads;
		int start_row = thread_id * rows_per_thread;
		int end_row = (thread_id == num_threads - 1) ? src.rows : start_row + rows_per_thread;

		// Process the assigned rows
		for (int i = start_row + border; i < end_row + border; i++)
		{
			for (int j = border; j < src.cols + border; j++)
			{
				double sum[3] = { 0 };
				for (int r = -border; r <= border; r++)
				{
					if (src.channels() == 1)
					{
						sum[0] += new_src.at<uchar>(i, j + r) * mask.at<double>(0, r + border);
					}
					else if (src.channels() == 3)
					{
						Vec3b rgb = new_src.at<Vec3b>(i, j + r);
						sum[0] += rgb[0] * mask.at<double>(0, r + border);
						sum[1] += rgb[1] * mask.at<double>(0, r + border);
						sum[2] += rgb[2] * mask.at<double>(0, r + border);
					}
				}
				if (new_src.channels() == 3)
				{
					for (int i = 0; i < 3; i++)
					{
						if (sum[i] < 0)
						{
							sum[i] = 0;
						}
						if (sum[i] > 255)
						{
							sum[i] = 255;
						}
					}
				}

				if (new_src.channels() == 1)
				{
					res.at<uchar>(i - border, j - border) = sum[0];
				}
				else if (new_src.channels() == 3)
				{
					Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
					res.at<Vec3b>(i - border, j - border) = rgb;
				}
			}
		}
	}

	// Padding the result image
	copyMakeBorder(res, new_src, border, border, border, border, BORDER_REPLICATE);

#pragma omp parallel num_threads(4)
	{
		int num_threads = omp_get_num_threads();
		int thread_id = omp_get_thread_num();

		// Divide the rows among threads
		int rows_per_thread = src.rows / num_threads;
		int start_row = thread_id * rows_per_thread;
		int end_row = (thread_id == num_threads - 1) ? src.rows : start_row + rows_per_thread;

		// Process the assigned rows
		for (int i = start_row + border; i < end_row + border; i++)
		{
			for (int j = border; j < src.cols + border; j++)
			{
				double sum[3] = { 0 };
				for (int c = -border; c <= border; c++)
				{
					if (src.channels() == 1)
					{
						sum[0] += new_src.at<uchar>(i + c, j) * mask.at<double>(0, c + border);
					}
					else if (src.channels() == 3)
					{
						Vec3b rgb = new_src.at<Vec3b>(i + c, j);
						sum[0] += rgb[0] * mask.at<double>(0, c + border);
						sum[1] += rgb[1] * mask.at<double>(0, c + border);
						sum[2] += rgb[2] * mask.at<double>(0, c + border);
					}
				}
				if (new_src.channels() == 3)
				{
					for (int i = 0; i < 3; i++)
					{
						if (sum[i] < 0)
						{
							sum[i] = 0;
						}
						if (sum[i] > 255)
						{
							sum[i] = 255;
						}
					}
				}

				if (new_src.channels() == 1)
				{
					res.at<uchar>(i - border, j - border) = sum[0];
				}
				else if (new_src.channels() == 3)
				{
					Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
					res.at<Vec3b>(i - border, j - border) = rgb;
				}
			}
		}
	}

	auto finish_clock = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds> (finish_clock - start_clock);
	std::cout << double(duration.count()) << std::endl;
}


//struct ThreadData {
//	Mat* src;
//	Mat* res;
//	Mat* new_src;
//	Mat* mask;
//	int msize;
//	int border;
//	int start_row;
//	int end_row;
//};
//
//void* processRows(void* arg) {
//	ThreadData* data = static_cast<ThreadData*>(arg);
//	Mat& src = *(data->src);
//	Mat& res = *(data->res);
//	Mat& new_src = *(data->new_src);
//	Mat& mask = *(data->mask);
//	int msize = data->msize;
//	int border = data->border;
//	int start_row = data->start_row;
//	int end_row = data->end_row;
//
//	std::vector<Mat> channels_;
//
//	for (int i = start_row; i < end_row; i++) {
//		for (int j = border; j < src.cols + border; j++) {
//			double sum[3] = { 0 };
//			for (int r = -border; r <= border; r++) {
//				if (src.channels() == 1) {
//					sum[0] += new_src.ptr<uchar>(i)[j + r] * mask.ptr<double>(0)[r + border];
//				}
//				else if (src.channels() == 3) {
//					split(new_src, channels_);
//					double rgb_0 = channels_.at(0).ptr<double>(0)[r + border];
//					double rgb_1 = channels_.at(1).ptr<double>(0)[r + border];
//					double rgb_2 = channels_.at(2).ptr<double>(0)[r + border];
//					sum[0] += rgb_0 * mask.at<double>(0, r + border);
//					sum[1] += rgb_1 * mask.at<double>(0, r + border);
//					sum[2] += rgb_2 * mask.at<double>(0, r + border);
//				}
//			}
//			if (new_src.channels() == 3) {
//				for (int i = 0; i < 3; i++) {
//					if (sum[i] < 0) {
//						sum[i] = 0;
//					}
//					if (sum[i] > 255) {
//						sum[i] = 255;
//					}
//				}
//			}
//
//			if (new_src.channels() == 1) {
//				res.at<uchar>(i - border, j - border) = static_cast<uchar>(sum[0]);
//			}
//			else if (new_src.channels() == 3) {
//				Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
//				res.at<Vec3b>(i - border, j - border) = rgb;
//			}
//		}
//	}
//
//	return nullptr;
//}
//
//void Gaussion_xy_pthread(Mat& src, Mat& res, const int msize, const double sigma) {
//	Mat mask;
//	Mat new_src;
//	std::vector<Mat> channels_;
//
//	// 生成一维核
//	Generate_Gaussion_Mask(mask, msize, sigma);
//
//	int border = (msize - 1) / 2;
//	res = Mat::zeros(src.size(), src.type());
//	copyMakeBorder(src, new_src, border, border, border, border, BORDER_REPLICATE);
//
//	int num_threads = std::min(static_cast<int>(src.rows), 8); // 设置线程数量
//	std::vector<pthread_t> threads(num_threads);
//	std::vector<ThreadData> thread_data(num_threads);
//
//	int rows_per_thread = src.rows / num_threads;
//	int remaining_rows = src.rows % num_threads;
//	int start_row = 0;
//	int end_row = rows_per_thread;
//
//	for (int i = 0; i < num_threads; i++) {
//		if (i == num_threads - 1) {
//			end_row += remaining_rows;
//		}
//
//		thread_data[i] = { &src, &res, &new_src, &mask, msize, border, start_row, end_row };
//		pthread_create(&threads[i], nullptr, processRows, &thread_data[i]);
//
//		start_row = end_row;
//		end_row += rows_per_thread;
//	}
//
//	for (int i = 0; i < num_threads; i++) {
//		pthread_join(threads[i], nullptr);
//	}
//
//	copyMakeBorder(res, new_src, border, border, border, border, BORDER_REPLICATE);
//
//	for (int i = border; i < src.rows + border; i++) {
//		for (int j = border; j < src.cols + border; j++) {
//			double sum[3] = { 0 };
//			for (int c = -border; c <= border; c++) {
//				if (src.channels() == 1) {
//					sum[0] += new_src.ptr<uchar>(i + c)[j] * mask.ptr<double>(0)[c + border];
//				}
//				else if (src.channels() == 3) {
//					split(new_src, channels_);
//					double rgb_0 = channels_.at(0).ptr<double>(0)[c + border];
//					double rgb_1 = channels_.at(1).ptr<double>(0)[c + border];
//					double rgb_2 = channels_.at(2).ptr<double>(0)[c + border];
//					sum[0] += rgb_0 * mask.at<double>(0, c + border);
//					sum[1] += rgb_1 * mask.at<double>(0, c + border);
//					sum[2] += rgb_2 * mask.at<double>(0, c + border);
//				}
//			}
//			if (new_src.channels() == 3) {
//				for (int i = 0; i < 3; i++) {
//					if (sum[i] < 0) {
//						sum[i] = 0;
//					}
//					if (sum[i] > 255) {
//						sum[i] = 255;
//					}
//				}
//			}
//
//			if (new_src.channels() == 1) {
//				res.at<uchar>(i - border, j - border) = static_cast<uchar>(sum[0]);
//			}
//			else if (new_src.channels() == 3) {
//				Vec3b rgb = { static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2]) };
//				res.at<Vec3b>(i - border, j - border) = rgb;
//			}
//		}
//	}
//
//	std::cout << "pthread情况" << std::endl;
//}





int main()
{
	Mat src = imread("D://parallel/woman.webp");
	Mat mask;
	Generate_Gaussion_Mask(mask, Size(5,5), 3.5);
	//std::cout << mask << std::endl;
	Mat res1;
	Mat res2;
	Mat res3;
	Mat res4;
	Mat res5;
	//res1 = imread("D://parallel/test.png");
	//imshow("origin", res1);
	imshow("原版", src);
	
	/*std::vector<Mat> v;
	split(src, v);
	imshow("B", v.at(0));
	imshow("G", v.at(1));
	imshow("R", v.at(2));
	merge(v, res2);
	imshow("RGB", res2);*/

	xy_filter xy_filter_;
	xy_filter_.Gaussion_xy_filter(src, res1, 5, 3.5, 8);
	imshow("new", res1);


	//Gaussion_simp(src, res1, mask);
	//imshow("模糊", res1);
	//Gaussion_xy(src, res2, 5, 3.5);
	//imshow("xy", res2);
	//Gaussion_xy_SSE(src, res3, 5, 3.5);
	//imshow(",,,", res3);
	//Gaussion_xy_omp(src, res4, 5, 3.5);
	//imshow("xxx", res4);
	//Gaussion_xy_omp_SSE(src, res5, 5, 3.5);
	//imshow("yyy", res5);
	//Gaussion_xy_pthread(src, res5, 5, 3.5);
	//imshow("pthread", res5);
	waitKey();
	return 0;
}