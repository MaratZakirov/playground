#include <iostream>
#include <vector>
#include <map>

//g++ utils.cpp -O3 -fPIC -std=c++11 -shared -o cpputils.so

using namespace std;

extern "C"  //Tells the compile to use C-linkage for the next scope.
{
    void getColPairs(int N, int * nums, int * X, int * Y, int * Pa, int * Pb)
    {
        map<tuple<int, int>, int> D;
        int cnt = 0;

        for (int i = 0; i < N; i++)
        {
            int true_i = nums[i];
            auto key_i = std::make_tuple (X[true_i], Y[true_i]);
            if (D.count(key_i) == 0 || D[key_i] == -1)
            {
                D[key_i] = true_i;
            } else
            {
                int true_j = D[key_i];
                Pa[cnt] = true_i;
                Pb[cnt] = true_j;
                D[key_i] = -1;
                cnt++;
            }
        }
    }
}