#include<iostream>
#include<vector>
#include<cmath>
#include<algorithm>
#include<ctime>
#include<chrono>
#include<tmmintrin.h>
#include<xmmintrin.h>
#include<emmintrin.h>
#include<pmmintrin.h>
#include<smmintrin.h>
#include<nmmintrin.h>
#include<immintrin.h>


using namespace std;
using namespace chrono;

void Gaussion(float** oarr, int n)
{
    float** arr = new float* [n];
    for (int i = 0; i < n; i++)
    {
        arr[i] = new float[n];
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            arr[i][j] = oarr[i][j];
        }
    }
    auto start_wt = steady_clock::now();
    float factor = 0.0;
    //消去过程
    for (int k = 0; k < n - 1; k++)
    {
        for (int i = k + 1; i < n; i++)
        {
            if (arr[k][k] == 0)
            {
                continue;
            }
            factor = arr[i][k] / arr[k][k];
            for (int j = k; j < n; j++)
            {
                arr[i][j] = arr[i][j] - factor * arr[k][j];
            }
        }
    }
    //
    auto finish_wt = steady_clock::now();
    auto duration = duration_cast<nanoseconds> (finish_wt - start_wt);
    cout << "平凡算法运行时间" << double(duration.count()) << "纳秒" << endl;
    for (int i = 0; i < n; i++)
    {
        delete[] arr[i];
    }
    delete[] arr;
}

void Gaussion_SSE(float** oarr, int n)
{
    float** arr = new float* [n];
    for (int i = 0; i < n; i++)
    {
        arr[i] = new float[n];
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            arr[i][j] = oarr[i][j];
        }
    }
    auto start_wt = steady_clock::now();
    float factor = 0.0;

    for (int k = 0; k < n - 1; k++) {
        __m128 t1 = _mm_set1_ps(arr[k][k]);
        int j = 0;
        for (j = k + 1; j + 4 <= n; j += 4) {
            __m128 t2 = _mm_loadu_ps(&arr[k][j]);   //未对齐，用loadu和storeu指令
            t2 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(&arr[k][j], t2);
        }
        for (; j < n; j++) {
            arr[k][j] = arr[k][j] / arr[k][k];
        }
        arr[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            __m128 vik = _mm_set1_ps(arr[i][k]);
            for (j = k + 1; j + 4 <= n; j += 4) {
                __m128 vkj = _mm_loadu_ps(&arr[k][j]);
                __m128 vij = _mm_loadu_ps(&arr[i][j]);
                __m128 vx = _mm_mul_ps(vik, vkj);
                vij = _mm_sub_ps(vij, vx);
                _mm_storeu_ps(&arr[i][j], vij);
            }
            for (; j < n; j++) {
                arr[i][j] = arr[i][j] - arr[i][k] * arr[k][j];
            }
            arr[i][k] = 0;
        }
    }


    auto finish_wt = steady_clock::now();
    auto duration = duration_cast<nanoseconds>(finish_wt - start_wt);
    cout << "SSE非对齐运行时间：" << double(duration.count()) << "纳秒" <<endl;
    for (int i = 0; i < n; i++)
    {
        delete[] arr[i];
    }
    delete[] arr;
}

void Gaussion_SSE_aligned(float** oarr, int n , int aligment)
{
    float** arr = (float**)_aligned_malloc(sizeof(float*) * n, aligment);
    for (int i = 0; i < n; i++) 
    {
        arr[i] = (float*)_aligned_malloc(sizeof(float) * n, aligment);
        //使得矩阵每一行在内存中按照alignment对齐，SSE为16，AVX为32
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            arr[i][j] = oarr[i][j];
        }
    }

    auto start_wt = steady_clock::now();
    float factor = 0.0;
    for (int k = 0; k < n - 1; k++) {
        __m128 t1 = _mm_set1_ps(arr[k][k]);
        int j = k + 1;

        //cout << &m[k][j];
        while ((int)(&arr[k][j]) % 16 && j < n)
        {
            arr[k][j] = arr[k][j] / arr[k][k];
            j++;
        }
        //cout << &m[k][j]<<endl;
        for (; j + 4 <= n; j += 4) {
            __m128 t2 = _mm_load_ps(&arr[k][j]);   //已对齐，用load和store指令
            t2 = _mm_div_ps(t2, t1);
            _mm_store_ps(&arr[k][j], t2);
        }
        for (; j < n; j++) {
            arr[k][j] = arr[k][j] / arr[k][k];
        }
        arr[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            __m128 vik = _mm_set1_ps(arr[i][k]);
            j = k + 1;
            while ((int)(&arr[k][j]) % 16 && j < n)
            {
                arr[i][j] = arr[i][j] - arr[i][k] * arr[k][j];
                j++;
            }
            for (; j + 4 <= n; j += 4) {
                __m128 vkj = _mm_load_ps(&arr[k][j]);
                __m128 vij = _mm_load_ps(&arr[i][j]);
                __m128 vx = _mm_mul_ps(vik, vkj);
                vij = _mm_sub_ps(vij, vx);
                _mm_store_ps(&arr[i][j], vij);
            }
            for (; j < n; j++) {
                arr[i][j] = arr[i][j] - arr[i][k] * arr[k][j];
            }
            arr[i][k] = 0;
        }
    }


    auto finish_wt = steady_clock::now();
    auto duration = duration_cast<nanoseconds>(finish_wt - start_wt);
    std::cout << "SSE对齐运行时间：" << double(duration.count()) << "纳秒"<< endl;
    for (int i = 0; i < n; i++)
    {
        _aligned_free(arr[i]);
    }
    _aligned_free(arr);
}

void Gaussion_AVX_16(float** oarr, int n, int aligment)
{
    float** arr = new float* [n];
    for (int i = 0; i < n; i++)
    {
        arr[i] = new float[n];
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            arr[i][j] = oarr[i][j];
        }
    }
    auto start_wt = steady_clock::now();
    float factor = 0.0;

    for (int k = 0; k < n; k++) {
        __m256 t1 = _mm256_set1_ps(arr[k][k]);
        int j = 0;
        for (j = k + 1; j + 16 <= n; j += 16) {
            __m256 t2 = _mm256_loadu_ps(&arr[k][j]);   //未对齐，用loadu和storeu指令
            t2 = _mm256_div_ps(t2, t1);
            _mm256_storeu_ps(&arr[k][j], t2);
        }
        for (; j < n; j++) {
            arr[k][j] = arr[k][j] / arr[k][k];
        }
        arr[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            __m256 vik = _mm256_set1_ps(arr[i][k]);
            for (j = k + 1; j + 16 <= n; j += 16) {
                __m256 vkj = _mm256_loadu_ps(&arr[k][j]);
                __m256 vij = _mm256_loadu_ps(&arr[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&arr[i][j], vij);
            }
            for (; j < n; j++) {
                arr[i][j] = arr[i][j] - arr[i][k] * arr[k][j];
            }
            arr[i][k] = 0;
        }
    }


    auto finish_wt = steady_clock::now();
    auto duration = duration_cast<nanoseconds>(finish_wt - start_wt);
    cout << "AVX非对齐_16运行时间：" << double(duration.count()) << "纳秒" << endl;
    for (int i = 0; i < n; i++)
    {
        delete[] arr[i];
    }
    delete[] arr;
}

void Gaussion_AVX_aligned_16(float** oarr, int n, int aligment)
{
    float** arr = (float**)_aligned_malloc(sizeof(float*) * n, aligment);
    for (int i = 0; i < n; i++)
    {
        arr[i] = (float*)_aligned_malloc(sizeof(float) * n, aligment);
        //使得矩阵每一行在内存中按照alignment对齐，SSE为16，AVX为32
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            arr[i][j] = oarr[i][j];
        }
    }

    auto start_wt = steady_clock::now();
    float factor = 0.0;
    for (int k = 0; k < n; k++) {
        __m256 t1 = _mm256_set1_ps(arr[k][k]);
        int j = k + 1;

        //cout << &m[k][j];
        while ((int)(&arr[k][j]) % 32 && j < n)
        {
            arr[k][j] = arr[k][j] / arr[k][k];
            j++;
        }
        //cout << &m[k][j]<<endl;
        for (; j + 16 <= n; j += 16) {
            __m256 t2 = _mm256_load_ps(&arr[k][j]);   //已对齐，用load和store指令
            t2 = _mm256_div_ps(t2, t1);
            _mm256_store_ps(&arr[k][j], t2);
        }
        for (; j < n; j++) {
            arr[k][j] = arr[k][j] / arr[k][k];
        }
        arr[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            __m256 vik = _mm256_set1_ps(arr[i][k]);
            j = k + 1;
            while ((int)(&arr[k][j]) % 32 && j < n)
            {
                arr[i][j] = arr[i][j] - arr[i][k] * arr[k][j];
                j++;
            }
            for (; j + 16 <= n; j += 16) {
                __m256 vkj = _mm256_load_ps(&arr[k][j]);
                __m256 vij = _mm256_load_ps(&arr[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_store_ps(&arr[i][j], vij);
            }
            for (; j < n; j++) {
                arr[i][j] = arr[i][j] - arr[i][k] * arr[k][j];
            }
            arr[i][k] = 0;
        }
    }

    auto finish_wt = steady_clock::now();
    auto duration = duration_cast<nanoseconds>(finish_wt - start_wt);
    std::cout << "AVX对齐_16运行时间：" << double(duration.count()) << "纳秒" << endl;
    for (int i = 0; i < n; i++)
    {
        _aligned_free(arr[i]);
    }
    _aligned_free(arr);
}

void Gaussion_AVX_8(float** oarr, int n, int aligment)
{
    float** arr = new float* [n];
    for (int i = 0; i < n; i++)
    {
        arr[i] = new float[n];
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            arr[i][j] = oarr[i][j];
        }
    }
    auto start_wt = steady_clock::now();
    float factor = 0.0;

    for (int k = 0; k < n; k++) {
        __m256 t1 = _mm256_set1_ps(arr[k][k]);
        int j = 0;
        for (j = k + 1; j + 8 <= n; j += 8) {
            __m256 t2 = _mm256_loadu_ps(&arr[k][j]);   //未对齐，用loadu和storeu指令
            t2 = _mm256_div_ps(t2, t1);
            _mm256_storeu_ps(&arr[k][j], t2);
        }
        for (; j < n; j++) {
            arr[k][j] = arr[k][j] / arr[k][k];
        }
        arr[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            __m256 vik = _mm256_set1_ps(arr[i][k]);
            for (j = k + 1; j + 8 <= n; j += 8) {
                __m256 vkj = _mm256_loadu_ps(&arr[k][j]);
                __m256 vij = _mm256_loadu_ps(&arr[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_storeu_ps(&arr[i][j], vij);
            }
            for (; j < n; j++) {
                arr[i][j] = arr[i][j] - arr[i][k] * arr[k][j];
            }
            arr[i][k] = 0;
        }
    }


    auto finish_wt = steady_clock::now();
    auto duration = duration_cast<nanoseconds>(finish_wt - start_wt);
    cout << "AVX非对齐_8运行时间：" << double(duration.count()) << "纳秒" << endl;
    for (int i = 0; i < n; i++)
    {
        delete[] arr[i];
    }
    delete[] arr;
}

void Gaussion_AVX_aligned_8(float** oarr, int n, int aligment)
{
    float** arr = (float**)_aligned_malloc(sizeof(float*) * n, aligment);
    for (int i = 0; i < n; i++)
    {
        arr[i] = (float*)_aligned_malloc(sizeof(float) * n, aligment);
        //使得矩阵每一行在内存中按照alignment对齐，SSE为16，AVX为32
    }
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            arr[i][j] = oarr[i][j];
        }
    }

    auto start_wt = steady_clock::now();
    float factor = 0.0;
    for (int k = 0; k < n ; k++) {
        __m256 t1 = _mm256_set1_ps(arr[k][k]);
        int j = k + 1;

        //cout << &m[k][j];
        while ((int)(&arr[k][j]) % 32 && j < n)
        {
            arr[k][j] = arr[k][j] / arr[k][k];
            j++;
        }
        //cout << &m[k][j]<<endl;
        for (; j + 8 <= n; j += 8) {
            __m256 t2 = _mm256_load_ps(&arr[k][j]);   //已对齐，用load和store指令
            t2 = _mm256_div_ps(t2, t1);
            _mm256_store_ps(&arr[k][j], t2);
        }
        for (; j < n; j++) {
            arr[k][j] = arr[k][j] / arr[k][k];
        }
        arr[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            __m256 vik = _mm256_set1_ps(arr[i][k]);
            j = k + 1;
            while ((int)(&arr[k][j]) % 32 && j < n)
            {
                arr[i][j] = arr[i][j] - arr[i][k] * arr[k][j];
                j++;
            }
            for (; j + 8 <= n; j += 8) {
                __m256 vkj = _mm256_load_ps(&arr[k][j]);
                __m256 vij = _mm256_load_ps(&arr[i][j]);
                __m256 vx = _mm256_mul_ps(vik, vkj);
                vij = _mm256_sub_ps(vij, vx);
                _mm256_store_ps(&arr[i][j], vij);
            }
            for (; j < n; j++) {
                arr[i][j] = arr[i][j] - arr[i][k] * arr[k][j];
            }
            arr[i][k] = 0;
        }
    }


    auto finish_wt = steady_clock::now();
    auto duration = duration_cast<nanoseconds>(finish_wt - start_wt);
    std::cout << "AVX对齐_8运行时间：" << double(duration.count()) << "纳秒" << endl;
    for (int i = 0; i < n; i++)
    {
       _aligned_free(arr[i]);
    }
    _aligned_free(arr);
}

//void Gaussion_NEON(float** oarr, int n)
//{
//    float** arr = new float* [n];
//    for (int i = 0; i < n; i++)
//    {
//        arr[i] = new float[n];
//    }
//
//    for (int i = 0; i < n; i++)
//    {
//        for (int j = 0; j < n; j++)
//        {
//            arr[i][j] = oarr[i][j];
//        }
//    }
//
//    auto start_wt = std::chrono::steady_clock::now();
//    float factor = 0.0;
//    //消去过程
//    for (int k = 0; k < n - 1; k++)
//    {
//        // 使用NEON优化的向量化计算
//        float32x4_t diagonal = vdupq_n_f32(arr[k][k]);
//        for (int i = k + 1; i < n; i++)
//        {
//            if (arr[k][k] == 0)
//            {
//                continue;
//            }
//            factor = arr[i][k] / arr[k][k];
//            float32x4_t scalingFactor = vdupq_n_f32(factor);
//            for (int j = k; j < n; j += 4)
//            {
//                float32x4_t currentRow = vld1q_f32(arr[i] + j);
//                float32x4_t diagonalElements = vld1q_f32(arr[k] + j);
//                float32x4_t updatedRow = vmlaq_f32(currentRow, scalingFactor, diagonalElements);
//                vst1q_f32(arr[i] + j, updatedRow);
//            }
//        }
//    }
//    //
//    auto finish_wt = std::chrono::steady_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds> (finish_wt - start_wt);
//    std::cout << "NEON算法运行时间" << double(duration.count()) << "纳秒" << std::endl;
//    for (int i = 0; i < n; i++)
//    {
//        delete[] arr[i];
//    }
//    delete[] arr;
//}

int main()
{
    /*for(int i = 0 ; i < 8; i++)
    {*/
        srand(time(0));
        int n = 0;
        /*if (i == 0)
        {
            n = 50;
        }
        else if (i == 1)
        {
            n = 100;
        }
        else if (i == 2)
        {
            n = 200;
        }
        else if (i == 3)
        {
            n = 300;
        }
        else if (i == 4)
        {
            n = 500;
        }
        else if (i == 5)
        {
            n = 1000;
        }
        else if (i == 6)
        {
            n = 2000;
        }
        else
        {
            n = 3000;
        }*/
        //cin >> n;
        n = 300;
        std::cout << "问题规模:" << n << endl;
        //vector<vector<float>> arr(n, vector<float>(n));
        float** arr;
        arr = new float* [n];
        for (int i = 0; i < n; i++)
        {
            arr[i] = new float[n];
        }

        //生成随机数据
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                arr[i][j] = rand() % 1000;
            }
        }

        Gaussion(arr, n);
        Gaussion_SSE(arr, n);
        Gaussion_SSE_aligned(arr, n, 16);
        Gaussion_AVX_16(arr, n, 32);
        Gaussion_AVX_aligned_16(arr, n, 32);
        Gaussion_AVX_8(arr, n, 32);
        Gaussion_AVX_aligned_8(arr, n, 32);


        for (int i = 0; i < n; i++)
        {
            delete[] arr[i];
        }
        delete[] arr;
   // }
    return 0;
}
