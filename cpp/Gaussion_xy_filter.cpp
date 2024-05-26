#include "Gaussion_xy_filter.h"
#include<omp.h>
using namespace cv;

//@alignment refers to align bits
void xy_filter::Gaussion_xy_filter(const Mat& src, Mat& res, const int msize, const double sigma, int alignment)
{
	res = Mat::zeros(src.size(), src.type());
	int border = (msize - 1) / 2;//border_size 用于填充边界
	Mat new_src;
	int height = src.rows;
	int width = src.cols;
	double* mask = (double*)_aligned_malloc(sizeof(double) * msize, alignment);
	generate_mask::Generate_mask(src, mask, msize, sigma);

	copyMakeBorder(src, new_src, border, border, border, border, BORDER_REPLICATE);


	int nwidth = new_src.cols;
	int nheight = new_src.rows;

	//对齐分配内存
	uchar** matrix = (uchar**)_aligned_malloc(sizeof(uchar*) * nwidth, alignment);
	for (int i = 0; i < nwidth; i++)
	{
		matrix[i] = (uchar*)_aligned_malloc(sizeof(uchar) * nheight, alignment);
	}

	uchar** res_ = (uchar**)_aligned_malloc(sizeof(uchar*) * nwidth, alignment);
	for (int i = 0; i < nwidth; i++)
	{
		res_[i] = (uchar*)_aligned_malloc(sizeof(uchar) * nheight, alignment);
	}

	uchar** tmp = (uchar**)_aligned_malloc(sizeof(uchar*) * nwidth, alignment);
	for (int i = 0; i < nwidth; i++)
	{
		tmp[i] = (uchar*)_aligned_malloc(sizeof(uchar) * nheight, alignment);
	}

	std::vector<Mat> channels_;
	split(new_src, channels_);

	if (new_src.channels() == 1)
	{
		for (int i = 0; i < nwidth; i++)
		{
			for (int j = 0; j < nheight; j++)
			{
				matrix[i][j] = channels_.at(0).ptr<uchar>(j)[i];
				tmp[i][j] = channels_.at(0).ptr<uchar>(j)[i];
			}
		}
		//process
		auto start_clock = std::chrono::high_resolution_clock::now();
		process(matrix ,tmp, res_, mask, width, height, border);
		auto finish_clock = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds> (finish_clock - start_clock);
		std::cout << "xy_filter_one_channel:" << double(duration.count()) << "ns" << std::endl;
		//process(matrix, mask, width, height, border);

		//just need to rewrite
		for (int i = 0; i < width; i++)
		{
			for (int j = 0; j < height; j++)
			{
				res.ptr<uchar>(j)[i] = res_[i + border][j + border];
			}
		}
	}
	else if (new_src.channels() == 3)
	{
		//std::vector<Mat> res_channel(3, Mat::zeros(src.size(), channels_.at(0).type()));
		std::vector<Mat> res_channel = channels_;
		//0 -> B
		//1 -> G
		//2 -> R
		auto start_clock = std::chrono::high_resolution_clock::now();
		for (int w = 0; w < 3; w++)
		{
			for (int i = 0; i < nwidth; i++)
			{
				for (int j = 0; j < nheight; j++)
				{
					matrix[i][j] = channels_.at(w).ptr<uchar>(j)[i];
					tmp[i][j] = channels_.at(w).ptr<uchar>(j)[i];
				}
			}
			//process
			process(matrix,tmp, res_, mask, width, height, border);
			//process(matrix, mask, width, height, border);

			//need merge
			for (int i = 0; i < width; i++)
			{
				for (int j = 0; j < height; j++)
				{
					res_channel.at(w).ptr<uchar>(j)[i] = res_[i + border][j + border];
				}
			}
		}
		auto finish_clock = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds> (finish_clock - start_clock);
		std::cout << "xy_filter_three_channels:" << double(duration.count()) << "ns" << std::endl;
		merge(res_channel, res);
	}

	for (int i = 0; i < nwidth; i++)
	{
		_aligned_free(matrix[i]);
	}
	_aligned_free(matrix);

	for (int i = 0; i < nwidth; i++)
	{
		_aligned_free(res_[i]);
	}
	_aligned_free(res_);

	for (int i = 0; i < nwidth; i++)
	{
		_aligned_free(tmp[i]);
	}
	_aligned_free(tmp);

	_aligned_free(mask);
}


void xy_filter::Gaussion_xy_filter_SSE(const Mat& src, Mat& res, const int msize, const double sigma, int alignment = 8)
{
	res = Mat::zeros(src.size(), src.type());
	int border = (msize - 1) / 2;//border_size 用于填充边界
	Mat new_src;
	int height = src.rows;
	int width = src.cols;
	double* mask = (double*)_aligned_malloc(sizeof(double) * msize, alignment);
	generate_mask::Generate_mask(src, mask, msize, sigma);

	copyMakeBorder(src, new_src, border, border, border, border, BORDER_REPLICATE);

	int nwidth = new_src.cols;
	int nheight = new_src.rows;

	//对齐分配内存
	uchar** matrix = (uchar**)_aligned_malloc(sizeof(uchar*) * nwidth, alignment);
	for (int i = 0; i < nwidth; i++)
	{
		matrix[i] = (uchar*)_aligned_malloc(sizeof(uchar) * nheight, alignment);
	}

	uchar** res_ = (uchar**)_aligned_malloc(sizeof(uchar*) * nwidth, alignment);
	for (int i = 0; i < nwidth; i++)
	{
		res_[i] = (uchar*)_aligned_malloc(sizeof(uchar) * nheight, alignment);
	}

	uchar** tmp = (uchar**)_aligned_malloc(sizeof(uchar*) * nwidth, alignment);
	for (int i = 0; i < nwidth; i++)
	{
		tmp[i] = (uchar*)_aligned_malloc(sizeof(uchar) * nheight, alignment);
	}

	std::vector<Mat> channels_;
	split(new_src, channels_);
	if (new_src.channels() == 1)
	{
		for (int i = 0; i < nwidth; i++)
		{
			for (int j = 0; j < nheight; j++)
			{
				matrix[i][j] = channels_.at(0).ptr<uchar>(j)[i];
				tmp[i][j] = channels_.at(0).ptr<uchar>(j)[i];
			}
		}
		//process
		auto start_clock = std::chrono::high_resolution_clock::now();
		process_SSE(matrix, tmp, res_, mask, width, height, border);
		auto finish_clock = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds> (finish_clock - start_clock);
		std::cout << "xy_filter_SSE_one_channel:" << double(duration.count()) << "ns" << std::endl;
		//process(matrix, mask, width, height, border);

		//just need to rewrite
		for (int i = 0; i < width; i++)
		{
			for (int j = 0; j < height; j++)
			{
				res.ptr<uchar>(j)[i] = res_[i + border][j + border];
			}
		}
	}
	else if (new_src.channels() == 3)
	{
		std::vector<Mat> res_channel = channels_;
		//0 -> B
		//1 -> G
		//2 -> R
		auto start_clock = std::chrono::high_resolution_clock::now();
		for (int w = 0; w < 3; w++)
		{
			for (int i = 0; i < nwidth; i++)
			{
				for (int j = 0; j < nheight; j++)
				{
					matrix[i][j] = channels_.at(w).ptr<uchar>(j)[i];
					tmp[i][j] = channels_.at(w).ptr<uchar>(j)[i];
				}
			}
			//process
			process_SSE(matrix,tmp, res_, mask, width, height, border);
			//process(matrix, mask, width, height, border);

			//need merge
			for (int i = 0; i < width; i++)
			{
				for (int j = 0; j < height; j++)
				{
					res_channel.at(w).ptr<uchar>(j)[i] = res_[i + border][j + border];
				}
			}
		}
		auto finish_clock = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds> (finish_clock - start_clock);
		std::cout << "xy_filter_SSE_three_channels:" << double(duration.count()) << "ns" << std::endl;

		merge(res_channel, res);
	}

	for (int i = 0; i < nwidth; i++)
	{
		_aligned_free(matrix[i]);
	}
	_aligned_free(matrix);

	for (int i = 0; i < nwidth; i++)
	{
		_aligned_free(res_[i]);
	}
	_aligned_free(res_);

	for (int i = 0; i < nwidth; i++)
	{
		_aligned_free(tmp[i]);
	}
	_aligned_free(tmp);

	_aligned_free(mask);
}

void xy_filter::Gaussion_xy_filter_AVX(const Mat& src, Mat& res, const int msize, const double sigma, int alignment = 16)
{
	res = Mat::zeros(src.size(), src.type());
	int border = (msize - 1) / 2;//border_size 用于填充边界
	Mat new_src;
	int height = src.rows;
	int width = src.cols;
	double* mask = (double*)_aligned_malloc(sizeof(double) * msize, alignment);
	generate_mask::Generate_mask(src, mask, msize, sigma);

	copyMakeBorder(src, new_src, border, border, border, border, BORDER_REPLICATE);

	int nwidth = new_src.cols;
	int nheight = new_src.rows;

	//对齐分配内存
	uchar** matrix = (uchar**)_aligned_malloc(sizeof(uchar*) * nwidth, alignment);
	for (int i = 0; i < nwidth; i++)
	{
		matrix[i] = (uchar*)_aligned_malloc(sizeof(uchar) * nheight, alignment);
	}

	uchar** res_ = (uchar**)_aligned_malloc(sizeof(uchar*) * nwidth, alignment);
	for (int i = 0; i < nwidth; i++)
	{
		res_[i] = (uchar*)_aligned_malloc(sizeof(uchar) * nheight, alignment);
	}

	uchar** tmp = (uchar**)_aligned_malloc(sizeof(uchar*) * nwidth, alignment);
	for (int i = 0; i < nwidth; i++)
	{
		tmp[i] = (uchar*)_aligned_malloc(sizeof(uchar) * nheight, alignment);
	}

	std::vector<Mat> channels_;
	split(new_src, channels_);
	if (new_src.channels() == 1)
	{
		for (int i = 0; i < nwidth; i++)
		{
			for (int j = 0; j < nheight; j++)
			{
				matrix[i][j] = channels_.at(0).ptr<uchar>(j)[i];
				tmp[i][j] = channels_.at(0).ptr<uchar>(j)[i];
			}
		}
		//process
		auto start_clock = std::chrono::high_resolution_clock::now();
		process_AVX(matrix, tmp, res_, mask, width, height, border);
		auto finish_clock = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds> (finish_clock - start_clock);
		std::cout << "xy_filter_AVX_one_channel:" << double(duration.count()) << "ns" << std::endl;
		//process(matrix, mask, width, height, border);

		//just need to rewrite
		for (int i = 0; i < width; i++)
		{
			for (int j = 0; j < height; j++)
			{
				res.ptr<uchar>(j)[i] = res_[i + border][j + border];
			}
		}
	}
	else if (new_src.channels() == 3)
	{
		std::vector<Mat> res_channel = channels_;
		//0 -> B
		//1 -> G
		//2 -> R
		auto start_clock = std::chrono::high_resolution_clock::now();
		for (int w = 0; w < 3; w++)
		{
			for (int i = 0; i < nwidth; i++)
			{
				for (int j = 0; j < nheight; j++)
				{
					matrix[i][j] = channels_.at(w).ptr<uchar>(j)[i];
					tmp[i][j] = channels_.at(w).ptr<uchar>(j)[i];
				}
			}
			//process
			process_AVX(matrix, tmp, res_, mask, width, height, border);
			//process(matrix, mask, width, height, border);

			//need merge
			for (int i = 0; i < width; i++)
			{
				for (int j = 0; j < height; j++)
				{
					res_channel.at(w).ptr<uchar>(j)[i] = res_[i + border][j + border];
				}
			}
		}
		auto finish_clock = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds> (finish_clock - start_clock);
		std::cout << "xy_filter_AVX_three_channels:" << double(duration.count()) << "ns" << std::endl;

		merge(res_channel, res);
	}

	for (int i = 0; i < nwidth; i++)
	{
		_aligned_free(matrix[i]);
	}
	_aligned_free(matrix);

	for (int i = 0; i < nwidth; i++)
	{
		_aligned_free(res_[i]);
	}
	_aligned_free(res_);

	for (int i = 0; i < nwidth; i++)
	{
		_aligned_free(tmp[i]);
	}
	_aligned_free(tmp);

	_aligned_free(mask);
}


void xy_filter::Gaussion_xy_filter_openmp_SSE(const Mat& src, Mat& res, const int msize, const double sigma, int alignment)
{
	int sum = 0;
#pragma parallel omp threads(4)
	{
		int thread_num = omp_get_num_threads();
		int thread_id = omp_get_thread_num();
		int per = 1000000 / thread_num;
		int start_ = thread_id * per;
		int end_ = (thread_id == thread_num) ? 1000000 : per * (thread_id + 1);
		for (int i = start_; i < end_; i++)
		{
			sum++;
		}
	}
	res = Mat::zeros(src.size(), src.type());
	int border = (msize - 1) / 2;//border_size 用于填充边界
	Mat new_src;
	int height = src.rows;
	int width = src.cols;
	double* mask = (double*)_aligned_malloc(sizeof(double) * msize, alignment);
	generate_mask::Generate_mask(src, mask, msize, sigma);

	copyMakeBorder(src, new_src, border, border, border, border, BORDER_REPLICATE);

	int nwidth = new_src.cols;
	int nheight = new_src.rows;

	//对齐分配内存
	uchar** matrix = (uchar**)_aligned_malloc(sizeof(uchar*) * nwidth, alignment);
	for (int i = 0; i < nwidth; i++)
	{
		matrix[i] = (uchar*)_aligned_malloc(sizeof(uchar) * nheight, alignment);
	}

	uchar** res_ = (uchar**)_aligned_malloc(sizeof(uchar*) * nwidth, alignment);
	for (int i = 0; i < nwidth; i++)
	{
		res_[i] = (uchar*)_aligned_malloc(sizeof(uchar) * nheight, alignment);
	}

	uchar** tmp = (uchar**)_aligned_malloc(sizeof(uchar*) * nwidth, alignment);
	for (int i = 0; i < nwidth; i++)
	{
		tmp[i] = (uchar*)_aligned_malloc(sizeof(uchar) * nheight, alignment);
	}

	std::vector<Mat> channels_;
	split(new_src, channels_);

	if (new_src.channels() == 1)
	{
		for (int i = 0; i < nwidth; i++)
		{
			for (int j = 0; j < nheight; j++)
			{
				matrix[i][j] = channels_.at(0).ptr<uchar>(j)[i];
				tmp[i][j] = channels_.at(0).ptr<uchar>(j)[i];
			}
		}
		//process
		auto start_clock = std::chrono::high_resolution_clock::now();
#pragma omp parallel num_threads(4)
		{
			int num_threads = omp_get_num_threads();
			int thread_id = omp_get_thread_num();
			int rows_per_thread = src.rows / num_threads;
			int start_row = thread_id * rows_per_thread;
			int end_row = (thread_id == num_threads - 1) ? src.rows : start_row + rows_per_thread;
#pragma omp parallel for
			for (int i = start_row; i < end_row; i++)
			{
				int j = 0;
				for (j = border; j + 4 < height + border; j += 4)
				{
					double sum[4];
					__m128d sum_ = _mm_set1_pd(0.0);
					int r;
					for (r = -border; r + 4 <= border; r += 4)
					{
						//mask vec
						__m128d mask_val = _mm_load_pd(&mask[r + border]);

						__m128i matrix_val = _mm_load_si128((__m128i*) & matrix[i][j]);

						//transformed matrix_val vec
						__m128d val = _mm_cvtepi32_pd(matrix_val);

						sum_ = _mm_add_pd(sum_, _mm_mul_pd(mask_val, val));
					}
					_mm_store1_pd(sum, sum_);
					for (; r <= border; r++)
					{
						sum[0] += matrix[i + r][j] * mask[r + border];
					}
					if (sum[0] < 0.0)
					{
						sum[0] = 0.0;
					}
					else if (sum[0] > 255.0)
					{
						sum[0] = 255.0;
					}
					tmp[i][j] = sum[0];
				}
				for (; j < height + border; j++)
				{
					double sum = 0.0;
					for (int r = -border; r <= border; r++)
					{
						sum += matrix[i + r][j] * mask[r + border];
					}
					if (sum < 0.0)
					{
						sum = 0.0;
					}
					else if (sum > 255.0)
					{
						sum = 255.0;
					}
					tmp[i][j] = sum;
				}
			}
			//process(matrix, mask, width, height, border);
		}

#pragma omp parallel num_threads(4)
		{
			int num_threads = omp_get_num_threads();
			int thread_id = omp_get_thread_num();
			int rows_per_thread = src.rows / num_threads;
			int start_row = thread_id * rows_per_thread;
			int end_row = (thread_id == num_threads - 1) ? src.rows : start_row + rows_per_thread;
#pragma omp parallel for
			for (int i = start_row; i < end_row; i++)
				{
					int j = 0;
					for (j = border; j + 4 < height + border; j += 4)
					{
						double sum[4];
						__m128d sum_ = _mm_set1_pd(0.0);
						int r;
						for (r = -border; r + 4 <= border; r += 4)
						{
							//mask vec
							__m128d mask_val = _mm_load_pd(&mask[r + border]);

							__m128i matrix_val = _mm_load_si128((__m128i*) & tmp[i][j]);

							//transformed matrix_val vec
							__m128d val = _mm_cvtepi32_pd(matrix_val);

							sum_ = _mm_add_pd(sum_, _mm_mul_pd(mask_val, val));
						}
						_mm_store1_pd(sum, sum_);
						for (; r <= border; r++)
						{
							sum[0] += tmp[i + r][j] * mask[r + border];
						}
						if (sum[0] < 0.0)
						{
							sum[0] = 0.0;
						}
						else if (sum[0] > 255.0)
						{
							sum[0] = 255.0;
						}
						res_[i][j] = sum[0];
					}
					for (; j < height + border; j++)
					{
						double sum = 0.0;
						for (int r = -border; r <= border; r++)
						{
							sum += tmp[i + r][j] * mask[r + border];
						}
						if (sum < 0.0)
						{
							sum = 0.0;
						}
						else if (sum > 255.0)
						{
							sum = 255.0;
						}
						res_[i][j] = sum;
					}
				}			
			//process(matrix, mask, width, height, border);
		}
		auto finish_clock = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds> (finish_clock - start_clock);
		std::cout << "xy_filter_omp_one_channel:" << double(duration.count()) << "ns" << std::endl;
		//just need to rewrite
		for (int i = 0; i < width; i++)
		{
			for (int j = 0; j < height; j++)
			{
				res.ptr<uchar>(j)[i] = res_[i + border][j + border];
			}
		}
	}
	else if (new_src.channels() == 3)
	{
		//std::vector<Mat> res_channel(3, Mat::zeros(src.size(), channels_.at(0).type()));
		std::vector<Mat> res_channel = channels_;
		//0 -> B
		//1 -> G
		//2 -> R
		auto start_clock = std::chrono::high_resolution_clock::now();
		for (int w = 0; w < 3; w++)
		{
			for (int i = 0; i < nwidth; i++)
			{
				for (int j = 0; j < nheight; j++)
				{
					matrix[i][j] = channels_.at(w).ptr<uchar>(j)[i];
					tmp[i][j] = channels_.at(w).ptr<uchar>(j)[i];
				}
			}
			//process

#pragma omp parallel num_threads(4)
			{
				int num_threads = omp_get_num_threads();
				int thread_id = omp_get_thread_num();
				int rows_per_thread = src.rows / num_threads;
				int start_row = thread_id * rows_per_thread;
				int end_row = (thread_id == num_threads - 1) ? src.rows : start_row + rows_per_thread;
#pragma omp parallel for
					for (int i = start_row; i < end_row; i++)
					{
						int j = 0;
						for (j = border; j + 4 < height + border; j += 4)
						{
							double sum[4];
							__m128d sum_ = _mm_set1_pd(0.0);
							int r;
							for (r = -border; r + 4 <= border; r += 4)
							{
								//mask vec
								__m128d mask_val = _mm_load_pd(&mask[r + border]);

								__m128i matrix_val = _mm_load_si128((__m128i*) & matrix[i][j]);

								//transformed matrix_val vec
								__m128d val = _mm_cvtepi32_pd(matrix_val);

								sum_ = _mm_add_pd(sum_, _mm_mul_pd(mask_val, val));
							}
							_mm_store1_pd(sum, sum_);
							for (; r <= border; r++)
							{
								sum[0] += matrix[i + r][j] * mask[r + border];
							}
							if (sum[0] < 0.0)
							{
								sum[0] = 0.0;
							}
							else if (sum[0] > 255.0)
							{
								sum[0] = 255.0;
							}
							tmp[i][j] = sum[0];
						}
						for (; j < height + border; j++)
						{
							double sum = 0.0;
							//for (int r = -border; r <= border; r++)
							//{
							//	sum += matrix[i + r][j] * mask[r + border];
							//}
							if (sum < 0.0)
							{
								sum = 0.0;
							}
							else if (sum > 255.0)
							{
								sum = 255.0;
							}
							tmp[i][j] = sum;
						}
					}				
				//process(matrix, mask, width, height, border);
			}

#pragma omp parallel num_threads(4)
			{
				int num_threads = omp_get_num_threads();
				int thread_id = omp_get_thread_num();
				int rows_per_thread = src.rows / num_threads;
				int start_row = thread_id * rows_per_thread;
				int end_row = (thread_id == num_threads - 1) ? src.rows : start_row + rows_per_thread;
#pragma omp parallel for
					for (int i = start_row; i < end_row; i++)
					{
						int j = 0;
						for (j = border; j + 4 < height + border; j += 4)
						{
							double sum[4];
							__m128d sum_ = _mm_set1_pd(0.0);
							int r;
							for (r = -border; r + 4 <= border; r += 4)
							{
								//mask vec
								__m128d mask_val = _mm_load_pd(&mask[r + border]);

								__m128i matrix_val = _mm_load_si128((__m128i*) & tmp[i][j]);

								//transformed matrix_val vec
								__m128d val = _mm_cvtepi32_pd(matrix_val);

								sum_ = _mm_add_pd(sum_, _mm_mul_pd(mask_val, val));
							}
							_mm_store1_pd(sum, sum_);
							//for (; r <= border; r++)
							//{
							//	sum[0] += tmp[i + r][j] * mask[r + border];
							//}
							if (sum[0] < 0.0)
							{
								sum[0] = 0.0;
							}
							else if (sum[0] > 255.0)
							{
								sum[0] = 255.0;
							}
							res_[i][j] = sum[0];
						}
						for (; j < height + border; j++)
						{
							double sum = 0.0;
							//for (int r = -border; r <= border; r++)
							//{
							//	sum += tmp[i + r][j] * mask[r + border];
							//}
							if (sum < 0.0)
							{
								sum = 0.0;
							}
							else if (sum > 255.0)
							{
								sum = 255.0;
							}
							res_[i][j] = sum;
						}
					}		
				//process(matrix, mask, width, height, border);
			}
			//need merge
			for (int i = 0; i < width; i++)
			{
				for (int j = 0; j < height; j++)
				{
					res_channel.at(w).ptr<uchar>(j)[i] = res_[i + border][j + border];
				}
			}
		}
		auto finish_clock = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds> (finish_clock - start_clock);
		std::cout << "xy_filter_omp_three_channels_SSE:" << double(duration.count()) << "ns" << std::endl;
		merge(res_channel, res);
	}

	for (int i = 0; i < nwidth; i++)
	{
		_aligned_free(matrix[i]);
	}
	_aligned_free(matrix);

	for (int i = 0; i < nwidth; i++)
	{
		_aligned_free(res_[i]);
	}
	_aligned_free(res_);

	for (int i = 0; i < nwidth; i++)
	{
		_aligned_free(tmp[i]);
	}
	_aligned_free(tmp);

	_aligned_free(mask);
}


void xy_filter::Gaussion_xy_filter_openmp_AVX(const Mat& src, Mat& res, const int msize, const double sigma, int alignment)
{
	int sum = 0;
#pragma parallel omp threads(4)
	{
		int thread_num = omp_get_num_threads();
		int thread_id = omp_get_thread_num();
		int per = 1000000 / thread_num;
		int start_ = thread_id * per;
		int end_ = (thread_id == thread_num) ? 1000000 : per * (thread_id + 1);
		for (int i = start_; i < end_; i++)
		{
			sum++;
		}
	}
	res = Mat::zeros(src.size(), src.type());
	int border = (msize - 1) / 2;//border_size 用于填充边界
	Mat new_src;
	int height = src.rows;
	int width = src.cols;
	double* mask = (double*)_aligned_malloc(sizeof(double) * msize, alignment);
	generate_mask::Generate_mask(src, mask, msize, sigma);

	copyMakeBorder(src, new_src, border, border, border, border, BORDER_REPLICATE);

	int nwidth = new_src.cols;
	int nheight = new_src.rows;

	//对齐分配内存
	uchar** matrix = (uchar**)_aligned_malloc(sizeof(uchar*) * nwidth, alignment);
	for (int i = 0; i < nwidth; i++)
	{
		matrix[i] = (uchar*)_aligned_malloc(sizeof(uchar) * nheight, alignment);
	}

	uchar** res_ = (uchar**)_aligned_malloc(sizeof(uchar*) * nwidth, alignment);
	for (int i = 0; i < nwidth; i++)
	{
		res_[i] = (uchar*)_aligned_malloc(sizeof(uchar) * nheight, alignment);
	}

	uchar** tmp = (uchar**)_aligned_malloc(sizeof(uchar*) * nwidth, alignment);
	for (int i = 0; i < nwidth; i++)
	{
		tmp[i] = (uchar*)_aligned_malloc(sizeof(uchar) * nheight, alignment);
	}

	std::vector<Mat> channels_;
	split(new_src, channels_);

	if (new_src.channels() == 1)
	{
		for (int i = 0; i < nwidth; i++)
		{
			for (int j = 0; j < nheight; j++)
			{
				matrix[i][j] = channels_.at(0).ptr<uchar>(j)[i];
				tmp[i][j] = channels_.at(0).ptr<uchar>(j)[i];
			}
		}
		//process
		auto start_clock = std::chrono::high_resolution_clock::now();
#pragma omp parallel num_threads(4)
		{
			int num_threads = omp_get_num_threads();
			int thread_id = omp_get_thread_num();
			int rows_per_thread = src.rows / num_threads;
			int start_row = thread_id * rows_per_thread;
			int end_row = (thread_id == num_threads - 1) ? src.rows : start_row + rows_per_thread;
#pragma omp parallel for
			for (int i = border; i < width + border; i++)
			{
				int j = 0;
				for (j = border; j + 8 < height + border; j += 8)
				{
					double sum[8];
					__m256d sum_ = _mm256_set1_pd(0.0);
					int r;
					for (r = -border; r + 8 <= border; r += 8)
					{
						//mask vec
						__m256d mask_val = _mm256_load_pd(&mask[r + border]);

						__m256i matrix_val = _mm256_load_si256((__m256i*) & matrix[i][j]);

						//transformed matrix_val vec
						__m256d val = _mm256_castsi256_pd(matrix_val);
						//__m256d val = _mm256_cvtepu64_pd(matrix_val);

						sum_ = _mm256_add_pd(sum_, _mm256_mul_pd(mask_val, val));
					}
					_mm256_storeu_pd(sum, sum_);
					for (; r <= border; r++)
					{
						sum[0] += matrix[i + r][j] * mask[r + border];
					}
					if (sum[0] < 0.0)
					{
						sum[0] = 0.0;
					}
					else if (sum[0] > 255.0)
					{
						sum[0] = 255.0;
					}
					matrix[i][j] = sum[0];
				}
				for (; j < height + border; j++)
				{
					double sum = 0.0;
					for (int r = -border; r <= border; r++)
					{
						sum += matrix[i + r][j] * mask[r + border];
					}
					if (sum < 0.0)
					{
						sum = 0.0;
					}
					else if (sum > 255.0)
					{
						sum = 255.0;
					}
					tmp[i][j] = sum;
				}
			}
			//process(matrix, mask, width, height, border);
		}

#pragma omp parallel num_threads(4)
		{
			int num_threads = omp_get_num_threads();
			int thread_id = omp_get_thread_num();
			int rows_per_thread = src.rows / num_threads;
			int start_row = thread_id * rows_per_thread;
			int end_row = (thread_id == num_threads - 1) ? src.rows : start_row + rows_per_thread;
#pragma omp parallel for
			for (int i = border; i < width + border; i++)
			{
				int j = 0;
				for (j = border; j + 8 < height + border; j += 8)
				{
					double sum[8];
					__m256d sum_ = _mm256_set1_pd(0.0);
					int r;
					for (r = -border; r + 8 <= border; r += 8)
					{
						//mask vec
						__m256d mask_val = _mm256_load_pd(&mask[r + border]);

						__m256i matrix_val = _mm256_load_si256((__m256i*) & matrix[i][j]);

						//transformed matrix_val vec
						__m256d val = _mm256_castsi256_pd(matrix_val);
						//__m256d val = _mm256_cvtepu64_pd(matrix_val);

						sum_ = _mm256_add_pd(sum_, _mm256_mul_pd(mask_val, val));
					}
					_mm256_storeu_pd(sum, sum_);
					for (; r <= border; r++)
					{
						sum[0] += tmp[i + r][j] * mask[r + border];
					}
					if (sum[0] < 0.0)
					{
						sum[0] = 0.0;
					}
					else if (sum[0] > 255.0)
					{
						sum[0] = 255.0;
					}
					res_[i][j] = sum[0];
				}
				for (; j < height + border; j++)
				{
					double sum = 0.0;
					for (int r = -border; r <= border; r++)
					{
						sum += tmp[i + r][j] * mask[r + border];
					}
					if (sum < 0.0)
					{
						sum = 0.0;
					}
					else if (sum > 255.0)
					{
						sum = 255.0;
					}
					res_[i][j] = sum;
				}
			}
			//process(matrix, mask, width, height, border);
		}
		auto finish_clock = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds> (finish_clock - start_clock);
		std::cout << "xy_filter_omp_one_channel_AVX:" << double(duration.count()) << "ns" << std::endl;
		//just need to rewrite
		for (int i = 0; i < width; i++)
		{
			for (int j = 0; j < height; j++)
			{
				res.ptr<uchar>(j)[i] = res_[i + border][j + border];
			}
		}
	}
	else if (new_src.channels() == 3)
	{
		//std::vector<Mat> res_channel(3, Mat::zeros(src.size(), channels_.at(0).type()));
		std::vector<Mat> res_channel = channels_;
		//0 -> B
		//1 -> G
		//2 -> R
		auto start_clock = std::chrono::high_resolution_clock::now();
		for (int w = 0; w < 3; w++)
		{
			for (int i = 0; i < nwidth; i++)
			{
				for (int j = 0; j < nheight; j++)
				{
					matrix[i][j] = channels_.at(w).ptr<uchar>(j)[i];
					tmp[i][j] = channels_.at(w).ptr<uchar>(j)[i];
				}
			}
			//process

#pragma omp parallel num_threads(4)
			{
				int num_threads = omp_get_num_threads();
				int thread_id = omp_get_thread_num();
				int rows_per_thread = src.rows / num_threads;
				int start_row = thread_id * rows_per_thread;
				int end_row = (thread_id == num_threads - 1) ? src.rows : start_row + rows_per_thread;
#pragma omp parallel for
				for (int i = border; i < width + border; i++)
				{
					int j = 0;
					for (j = border; j + 8 < height + border; j += 8)
					{
						double sum[8];
						__m256d sum_ = _mm256_set1_pd(0.0);
						int r;
						for (r = -border; r + 8 <= border; r += 8)
						{
							//mask vec
							__m256d mask_val = _mm256_load_pd(&mask[r + border]);

							__m256i matrix_val = _mm256_load_si256((__m256i*) & matrix[i][j]);

							//transformed matrix_val vec
							__m256d val = _mm256_castsi256_pd(matrix_val);
							//__m256d val = _mm256_cvtepu64_pd(matrix_val);

							sum_ = _mm256_add_pd(sum_, _mm256_mul_pd(mask_val, val));
						}
						_mm256_storeu_pd(sum, sum_);
						for (; r <= border; r++)
						{
							sum[0] += matrix[i + r][j] * mask[r + border];
						}
						if (sum[0] < 0.0)
						{
							sum[0] = 0.0;
						}
						else if (sum[0] > 255.0)
						{
							sum[0] = 255.0;
						}
						matrix[i][j] = sum[0];
					}
					for (; j < height + border; j++)
					{
						double sum = 0.0;
						for (int r = -border; r <= border; r++)
						{
							sum += matrix[i + r][j] * mask[r + border];
						}
						if (sum < 0.0)
						{
							sum = 0.0;
						}
						else if (sum > 255.0)
						{
							sum = 255.0;
						}
						tmp[i][j] = sum;
					}
				}
				//process(matrix, mask, width, height, border);
			}

#pragma omp parallel num_threads(4)
			{
				int num_threads = omp_get_num_threads();
				int thread_id = omp_get_thread_num();
				int rows_per_thread = src.rows / num_threads;
				int start_row = thread_id * rows_per_thread;
				int end_row = (thread_id == num_threads - 1) ? src.rows : start_row + rows_per_thread;
#pragma omp parallel for
				for (int i = border; i < width + border; i++)
				{
					int j = 0;
					for (j = border; j + 8 < height + border; j += 8)
					{
						double sum[8];
						__m256d sum_ = _mm256_set1_pd(0.0);
						int r;
						for (r = -border; r + 8 <= border; r += 8)
						{
							//mask vec
							__m256d mask_val = _mm256_load_pd(&mask[r + border]);

							__m256i matrix_val = _mm256_load_si256((__m256i*) & matrix[i][j]);

							//transformed matrix_val vec
							__m256d val = _mm256_castsi256_pd(matrix_val);
							//__m256d val = _mm256_cvtepu64_pd(matrix_val);

							sum_ = _mm256_add_pd(sum_, _mm256_mul_pd(mask_val, val));
						}
						_mm256_storeu_pd(sum, sum_);
						for (; r <= border; r++)
						{
							sum[0] += matrix[i + r][j] * mask[r + border];
						}
						if (sum[0] < 0.0)
						{
							sum[0] = 0.0;
						}
						else if (sum[0] > 255.0)
						{
							sum[0] = 255.0;
						}
						matrix[i][j] = sum[0];
					}
					for (; j < height + border; j++)
					{
						double sum = 0.0;
						for (int r = -border; r <= border; r++)
						{
							sum += matrix[i + r][j] * mask[r + border];
						}
						if (sum < 0.0)
						{
							sum = 0.0;
						}
						else if (sum > 255.0)
						{
							sum = 255.0;
						}
						res_[i][j] = sum;
					}
				}
				//need merge
				for (int i = 0; i < width; i++)
				{
					for (int j = 0; j < height; j++)
					{
						res_channel.at(w).ptr<uchar>(j)[i] = res_[i + border][j + border];
					}
				}
			}
		}
		auto finish_clock = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds> (finish_clock - start_clock);
		std::cout << "xy_filter_omp_three_channels_AVX:" << double(duration.count()) << "ns" << std::endl;
		merge(res_channel, res);

		for (int i = 0; i < nwidth; i++)
		{
			_aligned_free(matrix[i]);
		}
		_aligned_free(matrix);

		for (int i = 0; i < nwidth; i++)
		{
			_aligned_free(res_[i]);
		}
		_aligned_free(res_);

		for (int i = 0; i < nwidth; i++)
		{
			_aligned_free(tmp[i]);
		}
		_aligned_free(tmp);

		_aligned_free(mask);
	}
}



void xy_filter::Gaussion_xy_filter_openmp(const Mat& src, Mat& res, const int msize, const double sigma, int alignment , int thread_num)
{
	int sum = 0;
#pragma parallel omp threads(thread_num)
	{
		int thread_num = omp_get_num_threads();
		int thread_id = omp_get_thread_num();
		int per = 1000000 / thread_num;
		int start_ = thread_id * per;
		int end_ = (thread_id == thread_num) ? 1000000 : per * (thread_id + 1);
		for (int i = start_; i < end_; i++)
		{
			sum++;
		}
	}
	res = Mat::zeros(src.size(), src.type());
	int border = (msize - 1) / 2;//border_size 用于填充边界
	Mat new_src;
	int height = src.rows;
	int width = src.cols;
	double* mask = (double*)_aligned_malloc(sizeof(double) * msize, alignment);
	generate_mask::Generate_mask(src, mask, msize, sigma);

	copyMakeBorder(src, new_src, border, border, border, border, BORDER_REPLICATE);

	int nwidth = new_src.cols;
	int nheight = new_src.rows;

	//对齐分配内存
	uchar** matrix = (uchar**)_aligned_malloc(sizeof(uchar*) * nwidth, alignment);
	for (int i = 0; i < nwidth; i++)
	{
		matrix[i] = (uchar*)_aligned_malloc(sizeof(uchar) * nheight, alignment);
	}

	uchar** res_ = (uchar**)_aligned_malloc(sizeof(uchar*) * nwidth, alignment);
	for (int i = 0; i < nwidth; i++)
	{
		res_[i] = (uchar*)_aligned_malloc(sizeof(uchar) * nheight, alignment);
	}

	uchar** tmp = (uchar**)_aligned_malloc(sizeof(uchar*) * nwidth, alignment);
	for (int i = 0; i < nwidth; i++)
	{
		tmp[i] = (uchar*)_aligned_malloc(sizeof(uchar) * nheight, alignment);
	}

	std::vector<Mat> channels_;
	split(new_src, channels_);

	if (new_src.channels() == 1)
	{
		for (int i = 0; i < nwidth; i++)
		{
			for (int j = 0; j < nheight; j++)
			{
				matrix[i][j] = channels_.at(0).ptr<uchar>(j)[i];
				tmp[i][j] = channels_.at(0).ptr<uchar>(j)[i];
			}
		}
		//process
		auto start_clock = std::chrono::high_resolution_clock::now();
#pragma omp parallel num_threads(thread_num)
		{
			int num_threads = omp_get_num_threads();
			int thread_id = omp_get_thread_num();
			int rows_per_thread = src.rows / num_threads;
			int start_row = thread_id * rows_per_thread;
			int end_row = (thread_id == num_threads - 1) ? src.rows : start_row + rows_per_thread;
#pragma omp parallel for
			for (int i = border; i < width + border; i++)
			{
				for (int j = border; j < height + border; j++)
				{
					double sum = 0.0;
					for (int r = -border; r <= border; r++)
					{
						sum += matrix[i][j + r] * mask[r + border];
					}
					if (sum < 0.0)
					{
						sum = 0.0;
					}
					else if (sum > 255.0)
					{
						sum = 255.0;
					}
					tmp[i][j] = sum;
				}
			}
			//process(matrix, mask, width, height, border);
		}

#pragma omp parallel num_threads(thread_num)
		{
			int num_threads = omp_get_num_threads();
			int thread_id = omp_get_thread_num();
			int rows_per_thread = src.rows / num_threads;
			int start_row = thread_id * rows_per_thread;
			int end_row = (thread_id == num_threads - 1) ? src.rows : start_row + rows_per_thread;
#pragma omp parallel for
			for (int i = border; i < width + border; i++)
			{
				for (int j = border; j < height + border; j++)
				{
					double sum = 0.0;
					for (int r = -border; r <= border; r++)
					{
						sum += tmp[i][j + r] * mask[r + border];
					}
					if (sum < 0.0)
					{
						sum = 0.0;
					}
					else if (sum > 255.0)
					{
						sum = 255.0;
					}
					res_[i][j] = sum;
				}
			}
			//process(matrix, mask, width, height, border);
		}
		auto finish_clock = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds> (finish_clock - start_clock);
		std::cout << thread_num << "###" << "xy_filter_omp_one_channel:" << double(duration.count()) << "ns" << std::endl;
		//just need to rewrite
		for (int i = 0; i < width; i++)
		{
			for (int j = 0; j < height; j++)
			{
				res.ptr<uchar>(j)[i] = res_[i + border][j + border];
			}
		}
	}
	else if (new_src.channels() == 3)
	{
		//std::vector<Mat> res_channel(3, Mat::zeros(src.size(), channels_.at(0).type()));
		std::vector<Mat> res_channel = channels_;
		//0 -> B
		//1 -> G
		//2 -> R
		//auto start_clock = std::chrono::high_resolution_clock::now();
		for (int w = 0; w < 3; w++)
		{
			for (int i = 0; i < nwidth; i++)
			{
				for (int j = 0; j < nheight; j++)
				{
					matrix[i][j] = channels_.at(w).ptr<uchar>(j)[i];
					tmp[i][j] = channels_.at(w).ptr<uchar>(j)[i];
				}
			}
			//process

#pragma omp parallel num_threads(thread_num)
			{
				auto start_clock = std::chrono::high_resolution_clock::now();
				int num_threads = omp_get_num_threads();
				int thread_id = omp_get_thread_num();
				int rows_per_thread = src.rows / num_threads;
				int start_row = thread_id * rows_per_thread;
				int end_row = (thread_id == num_threads - 1) ? src.rows : start_row + rows_per_thread;
#pragma omp parallel for
				for(int i = border; i < width + border; i++)
				{
					for (int j = border; j < height + border; j++)
					{
						double sum = 0.0;
						for (int r = -border; r <= border; r++)
						{
							sum += matrix[i][j + r] * mask[r + border];
						}
						if (sum < 0.0)
						{
							sum = 0.0;
						}
						else if (sum > 255.0)
						{
							sum = 255.0;
						}
						tmp[i][j] = sum;
					}
				}
				auto finish_clock = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::nanoseconds> (finish_clock - start_clock);
				std::cout << thread_num << "###" << "xy_filter_omp_three_channels:" << double(duration.count()) << "ns" << std::endl;
				//process(matrix, mask, width, height, border);
			}

#pragma omp parallel num_threads(thread_num)
			{
				auto start_clock = std::chrono::high_resolution_clock::now();
				int num_threads = omp_get_num_threads();
				int thread_id = omp_get_thread_num();
				int rows_per_thread = src.rows / num_threads;
				int start_row = thread_id * rows_per_thread;
				int end_row = (thread_id == num_threads - 1) ? src.rows : start_row + rows_per_thread;
#pragma omp parallel for
				for (int i = border; i < width + border; i++)
				{
					for (int j = border; j < height + border; j++)
					{
						double sum = 0.0;
						for (int r = -border; r <= border; r++)
						{
							sum += tmp[i][j + r] * mask[r + border];
						}
						if (sum < 0.0)
						{
							sum = 0.0;
						}
						else if (sum > 255.0)
						{
							sum = 255.0;
						}
						res_[i][j] = sum;
					}
				}
				auto finish_clock = std::chrono::high_resolution_clock::now();
				auto duration = std::chrono::duration_cast<std::chrono::nanoseconds> (finish_clock - start_clock);
				std::cout << thread_num << "###" << "xy_filter_omp_three_channels:" << double(duration.count()) << "ns" << std::endl;
				//need merge
				for (int i = 0; i < width; i++)
				{
					for (int j = 0; j < height; j++)
					{
						res_channel.at(w).ptr<uchar>(j)[i] = res_[i + border][j + border];
					}
				}
			}
		}
		//auto finish_clock = std::chrono::high_resolution_clock::now();
		//auto duration = std::chrono::duration_cast<std::chrono::nanoseconds> (finish_clock - start_clock);
		//std::cout << thread_num << "###" << "xy_filter_omp_three_channels:" << double(duration.count()) << "ns" << std::endl;
		merge(res_channel, res);

		for (int i = 0; i < nwidth; i++)
		{
			_aligned_free(matrix[i]);
		}
		_aligned_free(matrix);

		for (int i = 0; i < nwidth; i++)
		{
			_aligned_free(res_[i]);
		}
		_aligned_free(res_);

		for (int i = 0; i < nwidth; i++)
		{
			_aligned_free(tmp[i]);
		}
		_aligned_free(tmp);

		_aligned_free(mask);
	}
}


struct ThreadData
{
	uchar** matrix;
	uchar** res_;
	uchar** tmp;
	double* mask;
	int start_row;
	int end_row;
	int width;
	int height;
	int border;
};

void* process_rows(void* thread_data)
{
	ThreadData* data = (ThreadData*)thread_data;
	uchar** matrix = data->matrix;
	uchar** res_ = data->res_;
	uchar** tmp = data->tmp;
	double* mask = data->mask;
	int start_row = data->start_row;
	int end_row = data->end_row;
	int width = data->width;
	int height = data->height;
	int border = data->border;

	for (int row = start_row; row < end_row; row++)
	{
		for (int col = border; col < height + border; col++)
		{
				double sum = 0.0;
				for (int r = -border; r <= border; r++)
				{
					sum += matrix[row + r][col] * mask[r + border];
				}
				if (sum < 0.0)
				{
					sum = 0.0;
				}
				else if (sum > 255.0)
				{
					sum = 255.0;
				}
				tmp[row][col] = sum;
		}
	}


	pthread_exit(NULL);
	return NULL;
}

void xy_filter::Gaussion_xy_filter_pthread(const Mat& src, Mat& res, const int msize, const double sigma, int alignment)
{
	res = Mat::zeros(src.size(), src.type());
	int border = (msize - 1) / 2;//border_size 用于填充边界
	Mat new_src;
	int height = src.rows;
	int width = src.cols;
	double* mask = (double*)_aligned_malloc(sizeof(double) * msize, alignment);
	generate_mask::Generate_mask(src, mask, msize, sigma);

	copyMakeBorder(src, new_src, border, border, border, border, BORDER_REPLICATE);

	int nwidth = new_src.cols;
	int nheight = new_src.rows;

	//对齐分配内存
	uchar** matrix = (uchar**)_aligned_malloc(sizeof(uchar*) * nwidth, alignment);
	for (int i = 0; i < nwidth; i++)
	{
		matrix[i] = (uchar*)_aligned_malloc(sizeof(uchar) * nheight, alignment);
	}

	uchar** tmp = (uchar**)_aligned_malloc(sizeof(uchar*) * nwidth, alignment);
	for (int i = 0; i < nwidth; i++)
	{
		tmp[i] = (uchar*)_aligned_malloc(sizeof(uchar) * nheight, alignment);
	}

	uchar** res_ = (uchar**)_aligned_malloc(sizeof(uchar*) * nwidth, alignment);
	for (int i = 0; i < nwidth; i++)
	{
		res_[i] = (uchar*)_aligned_malloc(sizeof(uchar) * nheight, alignment);
	}


	std::vector<Mat> channels_;
	split(new_src, channels_);


	if (new_src.channels() == 1)
	{
		for (int i = 0; i < nwidth; i++)
		{
			for (int j = 0; j < nheight; j++)
			{
				matrix[i][j] = channels_.at(0).ptr<uchar>(j)[i];
				tmp[i][j] = channels_.at(0).ptr<uchar>(j)[i];
			}
		}
		pthread_t threads[4];
		ThreadData thread_data[4];

		for (int i = 0; i < 4; i++)
		{
			thread_data[i].matrix = matrix;
			thread_data[i].res_ = res_;
			thread_data[i].tmp = tmp;
			thread_data[i].mask = mask;
			thread_data[i].start_row = i * (height / 4) + border;
			thread_data[i].end_row = (i == 3) ? height + border : (i + 1) * (height / 4) + border;
			thread_data[i].width = width;
			thread_data[i].height = height;
			thread_data[i].border = border;

			pthread_create(&threads[i], NULL, process_rows, (void*)&thread_data[i]);
		}

		for (int i = 0; i < 4; i++)
		{
			pthread_join(threads[i], NULL);
		}
	}
	else if (new_src.channels() == 3)
	{
		std::vector<Mat> res_channel = channels_;
		//0 -> B
		//1 -> G
		//2 -> R
		auto start_clock = std::chrono::high_resolution_clock::now();
		for (int w = 0; w < 3; w++)
		{
			for (int i = 0; i < nwidth; i++)
			{
				for (int j = 0; j < nheight; j++)
				{
					matrix[i][j] = channels_.at(w).ptr<uchar>(j)[i];
					tmp[i][j] = channels_.at(w).ptr<uchar>(j)[i];
				}
			}
			pthread_t threads[2];
			ThreadData thread_data[2];

			for (int i = 0; i < 2; i++)
			{
				thread_data[i].matrix = matrix;
				thread_data[i].res_ = res_;
				thread_data[i].tmp = tmp;
				thread_data[i].mask = mask;
				thread_data[i].start_row = border + i * (width / 2);
				thread_data[i].end_row = (i == 1) ? border + width : border + (i + 1) * (width / 2);
				thread_data[i].width = width;
				thread_data[i].height = height;
				thread_data[i].border = border;

				pthread_create(&threads[i], NULL, process_rows, (void*)&thread_data[i]);
			}

			for (int i = 0; i < 2; i++)
			{
				pthread_join(threads[i], NULL);
			}
			//need merge
			for (int i = 0; i < width; i++)
			{
				for (int j = 0; j < height; j++)
				{
					res_channel.at(w).ptr<uchar>(j)[i] = res_[i + border][j + border];
				}
			}
		}
		auto finish_clock = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds> (finish_clock - start_clock);
		std::cout << "xy_filter_pthread_three_channels:" << double(duration.count()) << "ns" << std::endl;
		merge(res_channel, res);

		for (int i = 0; i < nwidth; i++)
		{
			_aligned_free(matrix[i]);
		}
		_aligned_free(matrix);

		for (int i = 0; i < nwidth; i++)
		{
			_aligned_free(res_[i]);
		}
		_aligned_free(res_);

		for (int i = 0; i < nwidth; i++)
		{
			_aligned_free(tmp[i]);
		}
		_aligned_free(tmp);

		_aligned_free(mask);
	}
}


//@row col refers to src rows and cols
	//@mask refers to one_dimension window
	//@need to use twice
void xy_filter::process(uchar** matrix, uchar** tmp, uchar** res ,double* mask, const int width, const int height, const int border)
{
	for (int i = border; i < width + border; i++)
	{
		for (int j = border; j < height + border; j++)
		{
			double sum = 0.0;
			for (int r = -border; r <= border; r++)
			{
				sum += matrix[i + r][j] * mask[r + border];
			}
			if (sum < 0.0)
			{
				sum = 0.0;
			}
			else if (sum > 255.0)
			{
				sum = 255.0;
			}
			tmp[i][j] = sum;
		}
	}

	for (int i = border; i < width + border; i++)
	{
		for (int j = border; j < height + border; j++)
		{
			double sum = 0.0;
			for (int r = -border; r <= border; r++)
			{
				sum += tmp[i][j + r] * mask[r + border];
			}
			if (sum < 0.0)
			{
				sum = 0.0;
			}
			else if (sum > 255.0)
			{
				sum = 255.0;
			}
			res[i][j] = sum;
		}
	}
}