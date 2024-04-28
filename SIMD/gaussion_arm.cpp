#include <iostream>
#include<sys/time.h>
#include <arm_neon.h>
using namespace std;

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
    float factor = 0.0;
    timeval* start = new timeval();
    timeval* stop = new timeval();
    double durationTime = 0.0;
    gettimeofday(start, NULL);
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
    gettimeofday(stop, NULL);
    durationTime = stop->tv_sec * 1000 + double(stop->tv_usec) / 1000 - start->tv_sec * 1000 - double(start->tv_usec) / 1000;
    cout << " ParallelAlgorithm time: " << double(durationTime) << " ms" << endl;
    for (int i = 0; i < n; i++)
    {
        delete[] arr[i];
    }
    delete[] arr;
}



int main()
{
    for (int i = 0; i < 8; i++)
    {
        int n = 0;
        if (i == 0)
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
        }

        float** a = new float* [n];
        for (int i = 0; i < n; i++)
        {
            a[i] = new float[n];
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                a[i][j] = rand();
            }
        }

        timeval* start = new timeval();
        timeval* stop = new timeval();
        double durationTime = 0.0;
        gettimeofday(start, NULL);
        float32x4_t t0, t1, t2, t3;
        for (int k = 0; k < n; k++)
        {
            t0 = vld1q_dup_f32(a[k] + k);
            int j;

            for (j = k + 1; j + 7 < n; j += 8)
            {
                t1 = vld1q_f32(a[k] + j);
                t2 = vdivq_f32(t1, t0);
                vst1q_f32(a[k] + j, t2);
            }
            for (; j < n; j++)
                a[k][j] /= a[k][k];

            a[k][k] = 1.0;
            for (int i = k + 1; i < n; i++)
            {
                t0 = vld1q_dup_f32(a[i] + k);
                int j;
                for (j = k + 1; j + 7 < n; j += 8)
                {
                    t1 = vld1q_f32(a[k] + j);
                    t2 = vld1q_f32(a[i] + j);
                    t3 = vmulq_f32(t0, t1);
                    t2 = vsubq_f32(t2, t3);
                    vst1q_f32(a[i] + j, t2);
                }
                for (; j < n; j++)
                    a[i][j] -= a[i][k] * a[k][j];
                a[i][k] = 0.0;
            }
        }

        gettimeofday(stop, NULL);
        durationTime = stop->tv_sec * 1000 + double(stop->tv_usec) / 1000 - start->tv_sec * 1000 - double(start->tv_usec) / 1000;
        cout << " ParallelAlgorithm_4 time: " << double(durationTime) << " ms" << endl;
        for (int i = 0; i < n; i++)
        {
            delete[] a[i];
        }
        delete[] a;
    }

    for (int i = 0; i < 8; i++)
    {
        int n = 0;
        if (i == 0)
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
        }

        float** a = new float* [n];
        for (int i = 0; i < n; i++)
        {
            a[i] = new float[n];
        }

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                a[i][j] = rand();
            }
        }

        timeval* start = new timeval();
        timeval* stop = new timeval();
        double durationTime = 0.0;
        gettimeofday(start, NULL);
        float32x4_t t0, t1, t2, t3;
        for (int k = 0; k < n; k++)
        {
            t0 = vld1q_dup_f32(a[k] + k);
            int j;

            for (j = k + 1; j + 7 < n; j += 8)
            {
                t1 = vld1q_f32(a[k] + j);
                t2 = vdivq_f32(t1, t0);
                vst1q_f32(a[k] + j, t2);
            }
            for (; j < n; j++)
                a[k][j] /= a[k][k];

            a[k][k] = 1.0;
            for (int i = k + 1; i < n; i++)
            {
                t0 = vld1q_dup_f32(a[i] + k);
                int j;
                for (j = k + 1; j + 7 < n; j += 8)
                {
                    t1 = vld1q_f32(a[k] + j);
                    t2 = vld1q_f32(a[i] + j);
                    t3 = vmulq_f32(t0, t1);
                    t2 = vsubq_f32(t2, t3);
                    vst1q_f32(a[i] + j, t2);
                }
                for (; j < n; j++)
                    a[i][j] -= a[i][k] * a[k][j];
                a[i][k] = 0.0;
            }
        }

        gettimeofday(stop, NULL);
        durationTime = stop->tv_sec * 1000 + double(stop->tv_usec) / 1000 - start->tv_sec * 1000 - double(start->tv_usec) / 1000;
        cout << " ParallelAlgorithm_8 time: " << double(durationTime) << " ms" << endl;
        for (int i = 0; i < n; i++)
        {
            delete[] a[i];
        }
        delete[] a;
    }

    for (int i = 0; i < 8; i++)
    {
        int n = 0;
        if (i == 0)
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
        }

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


        for (int i = 0; i < n; i++)
        {
            delete[] arr[i];
        }
        delete[] arr;
    }

    return 0;
}
