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

// Function to perform special Gauss elimination
void specialGaussElimination(vector<vector<int>>& omatrix) {
    vector<vector<int>> matrix = omatrix;
    auto start_wt = steady_clock::now();
    int numColumns = matrix[0].size();
    int currentBatch = 0;
    vector<vector<int>> eliminatedRows;

    while (currentBatch < matrix.size()) {
        int numRowsInBatch = min(1000, static_cast<int>(matrix.size()) - currentBatch);  // Batch size
        // Read the current batch of rows into memory
        vector<vector<int>> batch(matrix.begin() + currentBatch, matrix.begin() + currentBatch + numRowsInBatch);

        for (int i = 0; i < numRowsInBatch; i++) {
            bool foundPivot = false;
            int pivotColumnIndex = -1;

            // Find the pivot column index (first non-zero element in current row)
            for (int j = 0; j < numColumns; j++) {
                if (batch[i][j] == 1) {
                    pivotColumnIndex = j;
                    foundPivot = true;
                    break;
                }
            }

            if (!foundPivot)
                continue;

            // Subtract the pivot row from other rows in the current batch
            for (int j = 0; j < numRowsInBatch; j++) {
                if (batch[j][pivotColumnIndex] == 1 && i != j) {
                    for (int k = 0; k < numColumns; k++) {
                        batch[j][k] ^= batch[i][k];
                    }
                }
            }

            // Check if the current row becomes all zeros (empty row)
            if (all_of(batch[i].begin(), batch[i].end(), [](int element) { return element == 0; })) {
                eliminatedRows.push_back(batch[i]);  // Add the empty row to the eliminated rows
                continue;
            }

            // Check if the pivot element is covered by the current batch
            if (pivotColumnIndex < currentBatch + numColumns) {
                // Check if the pivot element has a corresponding elimination row
                bool hasEliminationRow = any_of(eliminatedRows.begin(), eliminatedRows.end(), [&](const vector<int>& row) {
                    for (int k = 0; k < numColumns; k++) {
                        if (row[k] == 0 && batch[i][k] == 1)
                            return false;
                    }
                    return true;
                    });

                if (!hasEliminationRow) {
                    // Convert the current row into an elimination row
                    eliminatedRows.push_back(batch[i]);
                    continue;
                }
            }
        }

        currentBatch += numRowsInBatch;
    }

    auto finish_wt = steady_clock::now();
    auto duration = duration_cast<nanoseconds>(finish_wt - start_wt);
    cout << "特殊高斯消去计算运行时间：" << double(duration.count()) << " 纳秒" << endl;
}

// Function to perform the special Gauss elimination using SSE instructions
void specialGaussEliminationSSE(std::vector<std::vector<int>>& omatrix) {
    vector<vector<int>> matrix = omatrix;
    auto start_wt = steady_clock::now();
    int numColumns = matrix[0].size();
    int currentBatch = 0;
    vector<vector<int>> eliminatedRows;

    while (currentBatch < matrix.size()) {
        int numRowsInBatch = min(1000, static_cast<int>(matrix.size()) - currentBatch);  // Batch size

        // Read the current batch of rows into memory
        vector<vector<int>> batch(matrix.begin() + currentBatch, matrix.begin() + currentBatch + numRowsInBatch);

        for (int i = 0; i < numRowsInBatch; i++) {
            bool foundPivot = false;
            int pivotColumnIndex = -1;

            // Find the pivot column index (first non-zero element in current row)
            for (int j = 0; j < numColumns; j++) {
                if (batch[i][j] == 1) {
                    pivotColumnIndex = j;
                    foundPivot = true;
                    break;
                }
            }

            if (!foundPivot)
                continue;

            // Subtract the pivot row from other rows in the current batch
            for (int j = 0; j < numRowsInBatch; j++) {
                if (batch[j][pivotColumnIndex] == 1 && i != j) {
                    int k;
                    for(k = 0 ; k  < numColumns - numColumns % 4; k += 4)
                    {
                        // Load the pivot row and current row into SIMD registries.
                        __m128i pivotRow = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&batch[i][k]));
                        __m128i currentRow = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&batch[j][k]));

                        // Perform vectorized XOR operation.
                        __m128i xorResult = _mm_xor_si128(pivotRow, currentRow);
                        _mm_storeu_si128(reinterpret_cast<__m128i*>(&batch[j][k]), xorResult);
                    }
                    for (; k < numColumns; k++)
                    {
                        batch[j][k] ^= batch[i][k];
                    }
                }
            }

            // Check if the current row becomes all zeros (empty row)
            if (all_of(batch[i].begin(), batch[i].end(), [](int element) { return element == 0; })) {
                eliminatedRows.push_back(batch[i]);  // Add the empty row to the eliminated rows
                continue;
            }

            // Check if the pivot element is covered by the current batch
            if (pivotColumnIndex < currentBatch + numColumns) {
                // Check if the pivot element has a corresponding elimination row
                bool hasEliminationRow = any_of(eliminatedRows.begin(), eliminatedRows.end(), [&](const vector<int>& row) {
                    for (int k = 0; k < numColumns; k++) {
                        if (row[k] == 0 && batch[i][k] == 1)
                            return false;
                    }
                    return true;
                    });

                if (!hasEliminationRow) {
                    // Convert the current row into an elimination row
                    eliminatedRows.push_back(batch[i]);
                    continue;
                }
            }
        }

        currentBatch += numRowsInBatch;
    }

    auto finish_wt = steady_clock::now();
    auto duration = duration_cast<nanoseconds>(finish_wt - start_wt);
    cout << "SSE计算运行时间：" << double(duration.count()) << " 纳秒" << endl;
}

void specialGaussEliminationAVX_8(std::vector<std::vector<int>>& omatrix) {
    vector<vector<int>> matrix = omatrix;
    auto start_wt = steady_clock::now();
    int numColumns = matrix[0].size();
    int currentBatch = 0;
    vector<vector<int>> eliminatedRows;

    while (currentBatch < matrix.size()) {
        int numRowsInBatch = min(1000, static_cast<int>(matrix.size()) - currentBatch);  // Batch size

        // Read the current batch of rows into memory
        vector<vector<int>> batch(matrix.begin() + currentBatch, matrix.begin() + currentBatch + numRowsInBatch);

        for (int i = 0; i < numRowsInBatch; i++) {
            bool foundPivot = false;
            int pivotColumnIndex = -1;

            // Find the pivot column index (first non-zero element in current row)
            for (int j = 0; j < numColumns; j++) {
                if (batch[i][j] == 1) {
                    pivotColumnIndex = j;
                    foundPivot = true;
                    break;
                }
            }

            if (!foundPivot)
                continue;

            // Subtract the pivot row from other rows in the current batch
            for (int j = 0; j < numRowsInBatch; j++) {
                if (batch[j][pivotColumnIndex] == 1 && i != j) {
                    int k;
                    for(k = 0 ; k < numColumns - numColumns % 8 ; k+= 8)
                    {
                        // Load the pivot row and current row into SIMD registries.
                        __m256i pivotRow = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&batch[i][k]));
                        __m256i currentRow = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&batch[j][k]));

                        // Perform vectorized XOR operation.
                        __m256i xorResult = _mm256_xor_si256(pivotRow, currentRow);
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&batch[j][k]), xorResult);
                    }
                    for (; k < numColumns; k++)
                    {
                        batch[j][k] ^= batch[i][k];
                    }
                }
            }

            // Check if the current row becomes all zeros (empty row)
            if (all_of(batch[i].begin(), batch[i].end(), [](int element) { return element == 0; })) {
                eliminatedRows.push_back(batch[i]);  // Add the empty row to the eliminated rows
                continue;
            }

            // Check if the pivot element is covered by the current batch
            if (pivotColumnIndex < currentBatch + numColumns) {
                // Check if the pivot element has a corresponding elimination row
                bool hasEliminationRow = any_of(eliminatedRows.begin(), eliminatedRows.end(), [&](const vector<int>& row) {
                    for (int k = 0; k < numColumns; k++) {
                        if (row[k] == 0 && batch[i][k] == 1)
                            return false;
                    }
                    return true;
                    });

                if (!hasEliminationRow) {
                    // Convert the current row into an elimination row
                    eliminatedRows.push_back(batch[i]);
                    continue;
                }
            }
        }

        currentBatch += numRowsInBatch;
    }

    auto finish_wt = steady_clock::now();
    auto duration = duration_cast<nanoseconds>(finish_wt - start_wt);
    cout << "AVX_8计算运行时间：" << double(duration.count()) << " 纳秒" << endl;
}

void specialGaussEliminationAVX_16(std::vector<std::vector<int>>& omatrix) {
    vector<vector<int>> matrix = omatrix;
    auto start_wt = steady_clock::now();
    int numColumns = matrix[0].size();
    int currentBatch = 0;
    vector<vector<int>> eliminatedRows;

    while (currentBatch < matrix.size()) {
        int numRowsInBatch = min(1000, static_cast<int>(matrix.size()) - currentBatch);  // Batch size

        // Read the current batch of rows into memory
        vector<vector<int>> batch(matrix.begin() + currentBatch, matrix.begin() + currentBatch + numRowsInBatch);

        for (int i = 0; i < numRowsInBatch; i++) {
            bool foundPivot = false;
            int pivotColumnIndex = -1;

            // Find the pivot column index (first non-zero element in current row)
            for (int j = 0; j < numColumns; j++) {
                if (batch[i][j] == 1) {
                    pivotColumnIndex = j;
                    foundPivot = true;
                    break;
                }
            }

            if (!foundPivot)
                continue;

            // Subtract the pivot row from other rows in the current batch
            for (int j = 0; j < numRowsInBatch; j++) {
                if (batch[j][pivotColumnIndex] == 1 && i != j) {
                    int k;
                    for (k = 0; k < numColumns - numColumns % 16; k += 16)
                    {
                        // Load the pivot row and current row into SIMD registries.
                        __m256i pivotRow = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&batch[i][k]));
                        __m256i currentRow = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&batch[j][k]));

                        // Perform vectorized XOR operation.
                        __m256i xorResult = _mm256_xor_si256(pivotRow, currentRow);
                        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&batch[j][k]), xorResult);
                    }
                    for (; k < numColumns; k++)
                    {
                        batch[j][k] ^= batch[i][k];
                    }
                }
            }

            // Check if the current row becomes all zeros (empty row)
            if (all_of(batch[i].begin(), batch[i].end(), [](int element) { return element == 0; })) {
                eliminatedRows.push_back(batch[i]);  // Add the empty row to the eliminated rows
                continue;
            }

            // Check if the pivot element is covered by the current batch
            if (pivotColumnIndex < currentBatch + numColumns) {
                // Check if the pivot element has a corresponding elimination row
                bool hasEliminationRow = any_of(eliminatedRows.begin(), eliminatedRows.end(), [&](const vector<int>& row) {
                    for (int k = 0; k < numColumns; k++) {
                        if (row[k] == 0 && batch[i][k] == 1)
                            return false;
                    }
                    return true;
                    });

                if (!hasEliminationRow) {
                    // Convert the current row into an elimination row
                    eliminatedRows.push_back(batch[i]);
                    continue;
                }
            }
        }

        currentBatch += numRowsInBatch;
    }

    auto finish_wt = steady_clock::now();
    auto duration = duration_cast<nanoseconds>(finish_wt - start_wt);
    cout << "AVX_16计算运行时间：" << double(duration.count()) << " 纳秒" << endl;
}


int main() {
    // Example usage
    srand(time(0));
    int n = 1000;
    /*for (int i = 0; i < 6; i++)
    {
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
        }*/
        /*else if (i == 6)
        {
            n = 2000;
        }
        else
        {
            n = 3000;
        }*/
        cout << "问题规模" << n << endl;
        vector<vector<int>> matrix(n, vector<int>(n));
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                matrix[i][j] = rand() % 2;
            }
        }


        specialGaussElimination(matrix);

        specialGaussEliminationSSE(matrix);

        specialGaussEliminationAVX_8(matrix);

        specialGaussEliminationAVX_16(matrix);

    //}
    return 0;
}