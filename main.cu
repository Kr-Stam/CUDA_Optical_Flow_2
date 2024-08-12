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
    // initialize prevPyramid
    camera.read(frame);
    cv::Mat src = frame;
    cv::Mat gray(src.rows, src.cols, CV_8UC3);
    
    gpu::grayscale_avg(src.data, gray.data, src.rows, src.cols);
		//cpu::grayscale_avg_cpu(src.data, gray.data, src.rows, src.cols);

    unsigned char **prevPyramid = (unsigned char **)malloc(levels * sizeof(unsigned char *));
    int w = src.cols;
    int h = src.cols;
    for (int k = 0; k < levels; k++)
    {
        prevPyramid[k] = (unsigned char *)malloc(w * h * 3 * sizeof(unsigned char));
        w = w >> 1;
        h = h >> 1;
    }
		memcpy(prevPyramid[0], gray.data, gray.cols * gray.rows * sizeof(unsigned char) * 3);
    gpu::gauss_pyramid(prevPyramid, gray.cols, gray.rows, levels, GAUS_KERNEL_3x3, 3, 3);

    // camera.set(cv::CAP_PROP_FPS, 60);
    bool test = true;
    bool test_x = true;
    bool test_y = true;
    bool test_t = true;
    int testing_levels = 3;

		bool prevPyramidAlloc = true;
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
        w = src.cols;
        h = src.rows;
        unsigned char **pyramid = (unsigned char **)malloc(levels * sizeof(unsigned char *));
				pyramid[0] = filtered.data;
				//pyramid[0] = gray.data;
        for (int k = 1; k < levels; k++)
        {
            pyramid[k] = (unsigned char *)malloc(w * h * 3 * sizeof(unsigned char));
            w = w >> 1;
            h = h >> 1;
        }
        // cpu::gauss_pyramid(gray.data, src.cols, src.rows, levels, pyramid);
        // gpu::gauss_pyramid(gray.data, src.cols, src.rows, levels, pyramid);
        gpu::gauss_pyramid(pyramid, src.cols, src.rows, levels, GAUS_KERNEL_3x3, 3, 3);
        //cpu::gauss_pyramid(pyramid, src.cols, src.rows, levels, GAUS_KERNEL_3x3, 3, 3);

        // prikaz na nivoata na gauziskata piramida
        if (test)
        {
            char title[] = {'L', 'e', 'v', 'e', 'l', ' ', 'x', ' ', '\0'};
            char title1[] = {'L', 'e', 'v', 'e', 'l', ' ', 'x', ' ',' ', 'o', 'r', 'i', 'g', 'i', 'n', 'a', 'l', '\0'};
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
                    gpu::conv_3ch_1ch_tiled(prevPyramid[k], w, h, transformed_unscaledt1, Dt_3x3_n, 3, 3);
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
        
        float **flowPyramid = (float **)malloc(levels * sizeof(float *));
        for (int k = levels - 1; k >= 0; k--)
        {
            int tmp_h = src.rows >> k;
            int tmp_w = src.cols >> k;
            flowPyramid[k] = (float *)malloc(tmp_h * tmp_w * 2 * sizeof(float));
            gpu::calc_opt_flow(prevPyramid[k], pyramid[k], tmp_w, tmp_h, flowPyramid, k, levels);
            //cpu::calc_optical_flow(prevPyramid[k], pyramid[k], tmp_w, tmp_h, flowPyramid, k, levels);
        }

        // Visualize flow field
        int level = 0;
        h = src.rows >> level;
        w = src.cols >> level;
        int offset = w / 30;
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
                    u += (double)multiplier * flowPyramid[k][tmp_pos * 2];
                    v += (double)multiplier * flowPyramid[k][tmp_pos * 2 + 1];
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
                if(tmp_ni < 0 || tmp_nj < 0){
                    continue;
                }

                cv::arrowedLine(test, cv::Point(j, i), cv::Point(tmp_nj, tmp_ni), cv::Scalar(0, 0, 255), 1, 8, 0, 0.4);
            }
        }
        cv::imshow("Optical Flow Field", test);

        // Free malloc
				// index 0 is src
				if(prevPyramidAlloc){
					free(prevPyramid[0]);
					prevPyramidAlloc = false;
				}
        for (int i = 1; i < levels; i++)
        {
            free(prevPyramid[i]);
        }
        free(prevPyramid);
        prevPyramid = pyramid;
        for (int i = 1; i < levels; i++)
        {
            free(flowPyramid[i]);
        }
        free(flowPyramid);

        if (cv::waitKey(5) == 27)
        {
            return 0;
        }
    }

    return 0;
}
