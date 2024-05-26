#include"generate_mask.h"

using namespace cv;


void generate_mask::Generate_mask(const Mat& src, double* mask, const int msize, const double sigma)
{
	int center = (msize - 1) / 2;

	double sum = 0.0;
	double elem = 0.0;
	double g = 0.0;

	for (int i = 0; i < msize; i++)
	{
		elem = i - center;
		elem *= elem;
		g = exp(-elem / 2 * sigma * sigma);
		mask[i] = g;
		sum += g;
	}

}