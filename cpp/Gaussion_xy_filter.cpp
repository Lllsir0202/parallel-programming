#include "Gaussion_xy_filter.h"
#include "generate_mask.h"
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

	std::vector<Mat> channels_;
	split(new_src, channels_);
	if (new_src.channels() == 1)
	{
		for (int i = 0; i < nwidth; i++)
		{
			for (int j = 0; j < nheight; j++)
			{
				matrix[i][j] = channels_.at(0).ptr<uchar>(j)[i];
			}
		}
		//process

		process(matrix, mask, width, height, border);
		//process(matrix, mask, width, height, border);

		//just need to rewrite
		for (int i = 0; i < width; i++)
		{
			for (int j = 0; j < height; j++)
			{
				res.ptr<uchar>(j)[i] = matrix[i + border][j + border];
			}
		}
	}
	else if (new_src.channels() == 3)
	{
		std::vector<Mat> res_channel(3, Mat::zeros(src.size(), CV_8UC1));
		//0 -> B
		//1 -> G
		//2 -> R
		for (int w = 0; w < 3; w++)
		{
			for (int i = 0; i < nwidth; i++)
			{
				for (int j = 0; j < nheight; j++)
				{
					matrix[i][j] = channels_.at(w).ptr<uchar>(j)[i];
				}
			}
			//process
			process(matrix, mask, width, height, border);
			//process(matrix, mask, width, height, border);

			//need merge
			for (int i = 0; i < width; i++)
			{
				for (int j = 0; j < height; j++)
				{
					res_channel.at(w).ptr<uchar>(j)[i] = matrix[i + border][j + border];
				}
			}
		}
		merge(res_channel, res);
	}

	for (int i = 0; i < nwidth; i++)
	{
		_aligned_free(matrix[i]);
	}
	_aligned_free(matrix);
	_aligned_free(mask);
}


//@row col refers to src rows and cols
	//@mask refers to one_dimension window
	//@need to use twice
void xy_filter::process(uchar** matrix, double* mask, const int width, const int height, const int border)
{
	auto start_clock = std::chrono::high_resolution_clock::now();
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
			matrix[i][j] = sum;
		}
	}

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
			matrix[i][j] = sum;
		}
	}
	auto finish_clock = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::nanoseconds> (finish_clock - start_clock);
	std::cout << "情况" << double(duration.count()) << std::endl;
}