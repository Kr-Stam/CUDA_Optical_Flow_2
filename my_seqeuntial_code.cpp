#include <stdlib.h>
// #include <opencv2/cudev.hpp>
using namespace std;

void grayscaleAvg(unsigned char* src, unsigned char* dest, int w, int h){
    unsigned char* result = (unsigned char*) malloc(w * h * 3 * sizeof(unsigned char));

    for(int i = 0; i < h; i++){
        for(int j = 0; j < w; j++){
            int pos = i*w + j;
            int tmp = (src[pos*3] + src[pos*3 + 1] + src[pos*3 +2]) / 3;
            result[pos*3] = result[pos*3 + 1] = result[pos*3 + 2] = tmp;
        }
    }
}