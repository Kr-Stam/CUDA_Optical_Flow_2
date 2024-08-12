#pragma once

namespace gpu {

	void grayscale_avg(const unsigned char *src_h, unsigned char *dest_h, int h, int w);

	void conv_3ch_2d(const unsigned char *src_h, unsigned char *dest_h, int w, int h, const float *mask_t, int mw, int mh);

	void conv_3ch_2d_constant(const unsigned char *src_h, unsigned char *dest_h, int w, int h, const float *mask_t, int mw, int mh);

	void conv_3ch_tiled(const unsigned char *src_h, unsigned char *dest_h, int w, int h, const float *mask_t, int mw, int mh);

	void conv_3ch_1ch_constant(const unsigned char *src_h, int w, int h, unsigned char *dest_h, const float *mask_t, int mw, int mh);

	void conv_3ch_1ch_tiled(const unsigned char *src_h, int w, int h, unsigned char *dest_h, const float *mask_t, int mw, int mh);

	void conv_3ch_1ch_tiled_uchar_float(const unsigned char *src_h, int w, int h, float *dest_h, const float *mask_t, int mw, int mh);

	void conv_1d_3ch(unsigned char* src_h, int w, int h, unsigned char* dest_h);

	void gauss_pyramid(unsigned char ** pyramid, int w, int h, int levels, const float* mask, int mw, int mh);

	void srm_1ch(const unsigned char *arr1_h, const unsigned char *arr2_h, int w, int h, int ww, int wh, int *dest_h);

	void srm_1ch_float(const float *arr1_h, const float *arr2_h, int w, int h, int ww, int wh, float *dest_h);

	void srm_1ch_tiled(const unsigned char *arr1_h, const unsigned char *arr2_h, int w, int h, int ww, int wh, int *dest_h);

	void inverse_matrix(int *sumIx2, int *sumIy2, int *sumIxIy, int *sumIxIt, int *sumIyIt, float **optFlowPyramid, int level, int w, int h);

	void inverse_matrix_float(float *sumIx2, float *sumIy2, float *sumIxIy, float *sumIxIt, float *sumIyIt, float **optFlowPyramid, int level, int w, int h);

	void calc_opt_flow(const unsigned char *prev, unsigned char *next, int w, int h, float **optFlowPyramid, int level, int maxLevel);

	void bilinear_filter(unsigned char *src, unsigned char *gray, unsigned char *dest, int w, int h, int ww, int wh, double sigmaS, double sigmaB);
}
