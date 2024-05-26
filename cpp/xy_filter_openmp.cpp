#include"xy_filter_openmp.h"

using namespace cv;

void omp_filter::process_omp_row(uchar** matrix, uchar** tmp, double* mask, const int start_row , const int end_row, const int width, const int height, const int border)
{
	//const __m128d zero = _mm_set1_pd(0.0);
	//const __m128d max_val = _mm_set1_pd(255.0);


	for (int i = start_row ; i < end_row; i++)
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
}

void omp_filter::process_omp_col(uchar** tmp, uchar** res, double* mask, const int start_row, const int end_row, const int width, const int height, const int border)
{
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
			res[i][j] = sum[0];
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
			res[i][j] = sum;
		}
	}
}