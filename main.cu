#include "kernels.hpp"
#include "OptFlowCpu.hpp"
#include "OptFlowUtils.hpp"

#include <stdio.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "main.h"

#include "OptFlowGpu.cuh"

using namespace std;

void showTest(cv::Mat src, unsigned char** prev_pyramid, unsigned char** pyramid, int testing_levels, bool test_x, bool test_y, bool test_t)
{
		char title[] = {'L', 'e', 'v', 'e', 'l', ' ', 'x', ' ', '\0'};
		char title1[] = {'L', 'e', 'v', 'e', 'l', ' ', 'x', ' ',' ', 'o', 'r', 'i', 'g', 'i', 'n', 'a', 'l', '\0'};
		int w, h;
		for (int k = 0; k < testing_levels; k++)
		{
				cv::Mat transfomed_scaled = cv::Mat(src.rows, src.cols, CV_8UC1);
				cv::Mat original_scaled = cv::Mat(src.rows, src.cols, CV_8UC3);
				w = src.cols;
				h = src.rows;
				w = w >> k;
				h = h >> k;

				if(test_x){
						title[7] = 'x';
						title1[7] = ' ';
						unsigned char *transformed_unscaled = (unsigned char *)malloc(w * h * sizeof(unsigned char));
						gpu::conv_3ch_1ch_tiled(pyramid[k], w, h, transformed_unscaled, Dx_3x3, 3, 3);
						utils::cleanup_outliers(transformed_unscaled, w, h);
						utils::upscale_1ch(transformed_unscaled, w, h, k, transfomed_scaled.data);
						title[6] = '0' + (char)k;
						title1[6] = '0' + (char)k;
						cv::imshow(title, transfomed_scaled);
						utils::upscale_3ch(pyramid[k], w, h, k, original_scaled.data);
						cv::imshow(title1, original_scaled);
						free(transformed_unscaled);
				}
				if(test_t){
						title[7] = 't';
						title1[7] = ' ';
						unsigned char *transformed_unscaledt2 = (unsigned char *)malloc(w * h * sizeof(unsigned char));
						unsigned char *transformed_unscaledt1 = (unsigned char *)malloc(w * h * sizeof(unsigned char));
						gpu::conv_3ch_1ch_tiled(pyramid[k], w, h, transformed_unscaledt2, Dt_3x3_n, 3, 3);
						gpu::conv_3ch_1ch_tiled(prev_pyramid[k], w, h, transformed_unscaledt1, Dt_3x3_n, 3, 3);
						cpu::sub_arr(transformed_unscaledt2, transformed_unscaledt1, w *h, transformed_unscaledt1);

						utils::cleanup_outliers(transformed_unscaledt1, w, h);
						utils::upscale_1ch(transformed_unscaledt1, w, h, k, transfomed_scaled.data);
						title[6] = '0' + (char)k;
						title1[6] = '0' + (char)k;
						cv::imshow(title, transfomed_scaled);
						utils::upscale_3ch(pyramid[k], w, h, k, original_scaled.data);
						cv::imshow(title1, original_scaled);
						free(transformed_unscaledt1);
						free(transformed_unscaledt2);
				}
				if(test_y){
						title[7] = 'y';
						title1[7] = ' ';
						unsigned char *transformed_unscaled = (unsigned char *)malloc(w * h * sizeof(unsigned char));
						gpu::conv_3ch_1ch_tiled(pyramid[k], w, h, transformed_unscaled, Dy_3x3, 3, 3);
						utils::cleanup_outliers(transformed_unscaled, w, h);
						utils::upscale_1ch(transformed_unscaled, w, h, k, transfomed_scaled.data);
						title[6] = '0' + (char)k;
						title1[6] = '0' + (char)k;
						cv::imshow(title, transfomed_scaled);
						utils::upscale_3ch(pyramid[k], w, h, k, original_scaled.data);
						cv::imshow(title1, original_scaled);
						free(transformed_unscaled);
				}
		}
}
	
//TODO: smeni go ova so CUDA vektor tipovi float2 i uchar3
template<typename T, int ch> void alloc_pyramid(T*** pyramid, int w, int h, int levels)
{
    *pyramid = (T**) malloc(levels * sizeof(T *));
    for (int k = 0; k < levels; k++)
    {
        (*pyramid)[k] = (T*) malloc(w * h * ch * sizeof(T));
        w = w >> 1;
        h = h >> 1;
    }
}

//ova realno nema nikoja prichina zoshto da ne e void*, ama realno dava type safety takada neka sedi
template<typename T> void free_pyramid(T*** pyramid, int levels)
{
		for (int i = 0; i < levels; i++){
				free((*pyramid)[i]);
		}
		free(*pyramid);
}

void visualizeFlowField(cv::Mat src, unsigned char** pyramid, float** flow_pyramid, int levels, int level, int arrowRes)
{
	int h = src.rows >> level;
	int w = src.cols >> level;
	int offset = w / arrowRes;

	cv::Mat test(src.rows, src.cols, CV_8UC3);
	utils::upscale_3ch(pyramid[level], w, h, level, test.data);
	for (int i = 0; i < h; i += offset)
	{
		for (int j = 0; j < w; j += offset)
		{
				// For every point of prev
				//int pos = i * w + j;
				// Find cumulative flow of all previous levels
				float u, v;
				u = v = 0;
				for (int k = levels - 1; k >= level; k--)
				{
						int scale = k - level;
						int tmp_i = i >> scale;
						int tmp_j = j >> scale;
						int tmp_pos = tmp_i * (w >> scale) + tmp_j;
						int multiplier = 1 << scale;
						u += (double)multiplier * flow_pyramid[k][tmp_pos * 2];
						v += (double)multiplier * flow_pyramid[k][tmp_pos * 2 + 1];
				}
				
				//Ova e staveno za skaliranje na strelkite za podobra vizuelizacija
				if(u > offset){
						u = offset;
				}else if(u < - offset){
						u = - offset;
				}
				if(v > offset){
						v = offset;
				}else if(v < - offset){
						v = - offset;
				}
				int tmp_ni = (int)(v + i);
				int tmp_nj = (int)(u + j);

				//Ova e za da se izvadat nekoi problematichni presmetki
				if(tmp_ni < 0 || tmp_nj < 0) continue;

				cv::arrowedLine(test, cv::Point(j, i), cv::Point(tmp_nj, tmp_ni), cv::Scalar(0, 0, 255), 1, 8, 0, 0.4);
		}
	}
	cv::imshow("Optical Flow Field", test);

}

int main()
{
    printf("Optical Flow\n=================\n");

    cv::Mat frame;
    cv::VideoCapture camera(0);

    camera.set(cv::VideoCaptureProperties::CAP_PROP_FRAME_WIDTH, 640);
    camera.set(cv::VideoCaptureProperties::CAP_PROP_FRAME_HEIGHT, 480);

    if (!camera.isOpened())
    {
        printf("Camera is not opened changed device number\n");
        return 0;
    }

    int levels = 4;
    // initialize prev_pyramid
    camera.read(frame);
    cv::Mat src = frame;
    cv::Mat gray(src.rows, src.cols, CV_8UC3);
    
    gpu::grayscale_avg(src.data, gray.data, src.rows, src.cols);
		//cpu::grayscale_avg_cpu(src.data, gray.data, src.rows, src.cols);

		//Allocate Pyramids
		unsigned char** prev_pyramid;
		alloc_pyramid<unsigned char, 3>(&prev_pyramid, src.cols, src.rows, levels);
		unsigned char** pyramid;
		alloc_pyramid<unsigned char, 3>(&pyramid, src.cols, src.rows, levels);

		//init prev_pyramid
		memcpy(prev_pyramid[0], gray.data, gray.cols * gray.rows * sizeof(unsigned char) * 3);
    gpu::gauss_pyramid(prev_pyramid, gray.cols, gray.rows, levels, GAUS_KERNEL_3x3, 3, 3);

    // camera.set(cv::CAP_PROP_FPS, 60);
    bool test = true;
    bool test_x = true;
    bool test_y = true;
    bool test_t = true;
    int testing_levels = 3;

		int w, h;
		float** flow_pyramid;
		alloc_pyramid<float, 2>(&flow_pyramid, src.cols, src.rows, levels);

    while (true)
    {
        camera.read(frame);
        src = frame;
        cv::Mat dest(src.rows, src.cols, CV_8UC3);

        cv::imshow("Source", src);

        // Grayscale the image
        gray = cv::Mat(src.rows, src.cols, CV_8UC3);
        gpu::grayscale_avg(src.data, gray.data, src.rows, src.cols);

        w = src.cols;
        h = src.rows;
        int ww = 9;
        int wh = 9;
        cv::Mat filtered(src.rows, src.cols, CV_8UC3);
        // cpu::bilinear_filter_3ch(gray.data, gray.data, filtered.data, w, h, ww, wh, 10, 20);
        gpu::bilinear_filter(gray.data, gray.data, filtered.data, w, h, ww, wh, 2, 5);

        // Calculate optical flow

				//init pyramid
				//pyramid[0] = filtered.data;
				memcpy(pyramid[0], filtered.data, filtered.cols * filtered.rows * sizeof(unsigned char) * 3);
				//pyramid[0] = gray.data;
        // cpu::gauss_pyramid(gray.data, src.cols, src.rows, levels, pyramid);
        // gpu::gauss_pyramid(gray.data, src.cols, src.rows, levels, pyramid);
        gpu::gauss_pyramid(pyramid, src.cols, src.rows, levels, GAUS_KERNEL_3x3, 3, 3);
        //cpu::gauss_pyramid(pyramid, src.cols, src.rows, levels, GAUS_KERNEL_3x3, 3, 3);

        // prikaz na nivoata na gauziskata piramida
				if(test) showTest(src, prev_pyramid, pyramid, testing_levels, test_x, test_y, test_t);

				for(int k = levels - 1; k >= 0; k--)
				{
					int tmp_h = src.rows >> k;
					int tmp_w = src.cols >> k;
					gpu::calc_opt_flow(prev_pyramid[k], pyramid[k], tmp_w, tmp_h, flow_pyramid, k, levels);
					//cpu::calc_optical_flow(prev_pyramid[k], pyramid[k], tmp_w, tmp_h, flow_pyramid, k, levels);
				}

        // Visualize flow field
				int level = 0;
				int arrowRes = 30;
				visualizeFlowField(src, pyramid, flow_pyramid, levels, level, arrowRes);

				// Assign the current flow pyramid to the previous one
				unsigned char** swap = prev_pyramid;
				prev_pyramid = pyramid;
				pyramid = swap;

        if (cv::waitKey(5) == 27) return 0;
    }

		free_pyramid(&prev_pyramid, levels);
		free_pyramid(&pyramid, levels);
		free_pyramid(&flow_pyramid, levels);

    return 0;
}
