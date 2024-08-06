#include <string.h>
#include <stdlib.h>
#include "OptFlowCpu.hpp"
#include "kernels.hpp"

namespace cpu {
	
	//TODO: prefrli go ova vo util ili samo refaktoriraj da ne se koristi?
	void sub_arr(unsigned char *arr1, unsigned char *arr2, int n, unsigned char *dest)
	{
			for (int i = 0; i < n; i++)
			{
					dest[i] = arr1[i] - arr2[i];
			}
	}

	void grayscale_avg_cpu(const unsigned char *src, unsigned char *dest, int w, int h)
	{
			int pos, tmp;
			for (int i = 0; i < h; i++)
			{
					for (int j = 0; j < w; j++)
					{
							pos = i * w + j;
							tmp = (src[pos * 3] + src[pos * 3 + 1] + src[pos * 3 + 2]) / 3;
							dest[pos * 3] = dest[pos * 3 + 1] = dest[pos * 3 + 2] = tmp;
					}
			}
	}

	void conv_3ch(const unsigned char *src, const float *mask, unsigned char *dest, int w, int h, int mw, int mh)
	{
			int hmh = mh >> 1;
			int hmw = mw >> 1;

			int pos;
			int tmp[3];
			for (int y = 0; y < h; y++)
			{
					for (int x = 0; x < w; x++)
					{
							pos = y * w + x;

							int start_x = x - hmw;
							int start_y = y - hmh;
							int tmp_pos, mask_pos, tmp_x, tmp_y;

							tmp[0] = tmp[1] = tmp[2] = 0;

							for (int i = 0; i < mh; i++)
							{
									for (int j = 0; j < mw; j++)
									{
											tmp_x = start_x + j;
											tmp_y = start_y + i;
											if (tmp_x >= 0 && tmp_x < w && tmp_y >= 0 && tmp_y < h)
											{
													tmp_pos = tmp_y * w + tmp_x;
													mask_pos = i * mw + j;
													tmp[0] += src[tmp_pos * 3] * mask[mask_pos];
													tmp[1] += src[tmp_pos * 3 + 1] * mask[mask_pos];
													tmp[2] += src[tmp_pos * 3 + 2] * mask[mask_pos];
											}
									}
							}
							dest[pos * 3] = (unsigned char)tmp[0];
							dest[pos * 3 + 1] = (unsigned char)tmp[1];
							dest[pos * 3 + 2] = (unsigned char)tmp[2];
					}
			}
	}

	void conv_3ch_to_1ch(const unsigned char *src, int w, int h, unsigned char *dest, const float *mask, int mw, int mh)
	{
			int hmh = mh >> 1;
			int hmw = mw >> 1;

			int pos, tmp;
			for (int y = 0; y < h; y++)
			{
					for (int x = 0; x < w; x++)
					{
							pos = y * w + x;

							tmp = 0;
							int start_x = x - hmw;
							int start_y = y - hmh;
							int tmp_pos, mask_pos, tmp_x, tmp_y;

							for (int i = 0; i < mh; i++)
							{
									for (int j = 0; j < mw; j++)
									{
											tmp_x = start_x + j;
											tmp_y = start_y + i;
											if (tmp_x >= 0 && tmp_x < w && tmp_y >= 0 && tmp_y < h)
											{
													tmp_pos = tmp_y * w + tmp_x;
													mask_pos = i * mw + j;
													tmp += src[tmp_pos * 3] * mask[mask_pos];
											}
									}
							}
							dest[pos] = (unsigned char)tmp;
					}
			}
	}

	//TODO: Mora da ja refaktoriram funkcijava da e podobra
	void downscale_gaussian(unsigned char* src, int w, int h, unsigned char * dest, const float* mask, int mw, int mh){

			int hmh = mh >> 1;
			int hmw = mw >> 1;
			
			int pw = w << 1;
			int ph = h << 1;

			for (int y = 0; y < h; y++)
			{
					for (int x = 0; x < w; x++)
					{
							float tmp[3] = {0, 0, 0};
							int start_y = (y << 1) - hmh;
							int start_x = (x << 1) - hmw;
							for (int p = 0; p < mh; p++)
							{
									for (int q = 0; q < mw; q++)
									{
											int cx = start_x + q;
											int cy = start_y + p;
											if (cx >= 0 && cx < pw && cy >= 0 && cy < ph)
											{
													int mask_pos = p * mw + q;
													int img_pos = (cy * pw + cx) * 3;
													tmp[0] += mask[mask_pos] * src[img_pos];
													tmp[1] += mask[mask_pos] * src[img_pos + 1];
													tmp[2] += mask[mask_pos] * src[img_pos + 2];
											}
									}
							}
							dest[(y * w + x) * 3] = (unsigned char)tmp[0];
							dest[(y * w + x) * 3 + 1] = (unsigned char)tmp[1];
							dest[(y * w + x) * 3 + 2] = (unsigned char)tmp[2];
					}
			}
	}

	//TODO: mozhe da se napravi i kombinirana CPU/GPU verzija koja kje bide vo namespace cova mislam deka mozhe i da se ombined
	void gauss_pyramid(unsigned char **pyramid, int w, int h, int n, const float* mask, int mw, int mh)
	{

			for (int i = 1; i < n; i++)
			{
					int tw = w >> i;
					int th = h >> i;
					downscale_gaussian(pyramid[i - 1], tw, th, pyramid[i], mask, mw, mh);
			}
	}
	
	void srm_1ch(const unsigned char *arr1, const unsigned char *arr2, int w, int h, int ww, int wh, int *dest)
	{

			int hww = ww >> 1;
			int hwh = wh >> 1;

			int tmp_pos, tmp_x, tmp_y, pos, start_x, start_y;
			int tmp = 0;

			for (int i = 0; i < h; i++)
			{
					for (int j = 0; j < w; j++)
					{
							pos = i * w + j;
							start_x = j - hww;
							start_y = i - hwh;
							tmp = 0;
							for (int p = 0; p < wh; p++)
							{
									tmp_y = start_y + p;
									if (tmp_y < 0 || tmp_y >= h)
									{
											continue;
									}
									for (int q = 0; q < ww; q++)
									{
											tmp_x = start_x + q;
											if (tmp_x < 0 || tmp_x >= w)
											{
													continue;
											}
											tmp_pos = tmp_y * w + tmp_x;
											tmp += arr1[tmp_pos] * arr2[tmp_pos];
									}
							}
							dest[pos] = tmp;
					}
			}
	}

	void srm_3ch(unsigned char *arr1, unsigned char *arr2, int w, int h, int ww, int wh, int *dest)
	{

			int hkh = wh >> 1;
			int hkw = ww >> 1;

			for (int i = 0; i < h; i++)
			{
					for (int j = 0; j < w; j++)
					{
							int tmp[3] = {0, 0, 0};
							int start_x = j - hkw;
							int start_y = i - hkh;

							for (int y = 0; y < wh; y++)
							{
									for (int x = 0; x < ww; x++)
									{
											int cx = start_x + x;
											int cy = start_y + y;
											if (cx < 0 || cy < 0 || cx > w || cy > h)
											{
													continue;
											}
											int pos = cy * w + cx;
											tmp[0] += arr1[pos * 3] * arr2[pos * 3];
											tmp[1] += arr1[pos * 3 + 1] * arr2[pos * 3 + 1];
											tmp[2] += arr1[pos * 3 + 2] * arr2[pos * 3 + 2];
									}
							}
							int pos = i * w + j;
							dest[pos * 3] = tmp[0];
							dest[pos * 3 + 1] = tmp[1];
							dest[pos * 3 + 2] = tmp[2];
					}
			}
	}

	//TODO: mora da se refkatorira
	void shift_back_pyramid(const unsigned char *src, int w, int h, int level, int maxLevel, float **optFlowPyramid, unsigned char *dest)
	{
			int pos, tmp_pos;
			//? Ne sum siguren za ova
			//TODO: treba da go refaktoriram ova za da go izvadam #include <string.h>
			// zoshto e memcpy vo string, shto kur
			memcpy(dest, src, w * h * sizeof(unsigned char));
			for (int i = 0; i < h; i++)
			{
					for (int j = 0; j < w; j++)
					{
							// For every point of prev
							pos = i * w + j;
							// Find cumulative flow of all previous levels
							float u, v;
							u = v = 0;
							for (int k = maxLevel - 1; k > level; k--)
							{
									int offset = k - level;
									int tmp_i = i * (1 >> offset);
									int tmp_j = j * (1 >> offset);
									int tmp_pos = tmp_i * (w >> offset) + tmp_j;
									int multiplier = 1 << offset;
									u += (float) multiplier * optFlowPyramid[k][tmp_pos * 2];
									v += (float) multiplier * optFlowPyramid[k][tmp_pos * 2 + 1];
							}
							// calculate new_pos
							int new_pos_x = j + u;
							int new_pos_y = i + v;
							if (new_pos_x >= w || new_pos_x < 0 || new_pos_y >= h || new_pos_y < 0)
							{
									continue;
							}

							int new_pos = new_pos_y * w + new_pos_x;
							// Now put new_pos into pos of dest
							dest[pos * 3] = src[new_pos * 3];
							dest[pos * 3 + 1] = src[new_pos * 3 + 1];
							dest[pos * 3 + 2] = src[new_pos * 3 + 2];
					}
			}
	}

	//TODO: Ova ne raboteshe kako shto treba, mora da se smeni
	void inverse_matrix(int *sumIx2, int *sumIy2, int *sumIxIy, int *sumIxIt, int *sumIyIt, float **optFlowPyramid, int level, int w, int h)
	{
			for (int i = 0; i < h; i++)
			{
					for (int j = 0; j < w; j++)
					{
							// Calculate inverse matrix (AAT)^-1
							int pos = i * w + j;
							float a, b, c, d;
							a = (float)sumIx2[pos];
							b = c = (float)sumIxIy[pos];
							d = (float)sumIy2[pos];
							float prefix = 1 / (a * d - b * c);
							a *= prefix;
							b *= prefix;
							c *= prefix;
							d *= prefix;

							float u = -d * sumIxIt[pos] + b * sumIyIt[pos];
							float v = c * sumIxIt[pos] - a * sumIyIt[pos];
							optFlowPyramid[level][pos * 2] = u;
							optFlowPyramid[level][pos * 2 + 1] = v;
					}
			}
	}

	//TODO: Get this function working properly and refactor it, so that it does not manage its own memory
	void calc_optical_flow(const unsigned char *prev, unsigned char *next, int w, int h, float **optFlowPyramid, int level, int maxLevel)
	{
			// optFlowPyramid is the pyramid of all optical flows
			// optFlowPyramid[i] is the optical flow field, described by a vector (u, v) at each point

			// STEP 0
			// SHIFT NEXT IMAGE BACK BY PREVIOUSLY CALCULATED OPTICAL FLOW
			// Ova se pravi za celiot dosega presmetan optical flow
			unsigned char *shifted = (unsigned char *)malloc(w * h * 3 * sizeof(unsigned char));
			if (level != maxLevel - 1)
			{
					shift_back_pyramid(next, w, h, level, maxLevel, optFlowPyramid, shifted);
					next = shifted;
			}

			// STEP 1
			// calculate partial derivatives at all points using kernels for finite differences (Ix, Iy, It)
			unsigned char *Ix = (unsigned char *)malloc(w * h * sizeof(unsigned char));
			conv_3ch_to_1ch(prev, w, h, Ix, Dx_3x3, 3, 3);

			unsigned char *Iy = (unsigned char *)malloc(w * h * sizeof(unsigned char));
			conv_3ch_to_1ch(prev, w, h, Iy, Dy_3x3, 3, 3);

			unsigned char *It1 = (unsigned char *)malloc(w * h * sizeof(unsigned char));
			conv_3ch_to_1ch(prev, w, h, It1, GAUS_KERNEL_3x3, 3, 3);
			unsigned char *It2 = (unsigned char *)malloc(w * h * sizeof(unsigned char));
			conv_3ch_to_1ch(next, w, h, It2, GAUS_KERNEL_3x3, 3, 3);
			unsigned char *It = It1; // ova za da bide podobro optimizirano
			sub_arr(It2, It1, w * h, It);

			// STEP 2
			// Calculate sumIx2, sumIy2, sumIxIy, sumIxIt, sumIyIt
			int ww = 9;
			int wh = 9;
			int *sumIx2 = (int *)malloc(w * h * sizeof(int));
			srm_1ch(Ix, Ix, w, h, ww, wh, sumIx2);

			int *sumIy2 = (int *)malloc(w * h * sizeof(int));
			srm_1ch(Iy, Iy, w, h, ww, wh, sumIy2);

			int *sumIxIy = (int *)malloc(w * h * sizeof(int));
			srm_1ch(Ix, Iy, w, h, ww, wh, sumIxIy);

			int *sumIxIt = (int *)malloc(w * h * sizeof(int));
			srm_1ch(Ix, It, w, h, ww, wh, sumIxIt);
			int *sumIyIt = (int *)malloc(w * h * sizeof(int));
			srm_1ch(Iy, It, w, h, ww, wh, sumIyIt);

			// STEP 3
			// Calculate the optical flow vector at every point (i, j)
			//  inverse_matrix(sumIx2, sumIy2, sumIxIy, sumIxIt, sumIyIt, optFlowPyramid, level, w, h);
			for (int i = 0; i < h; i++)
			{
					for (int j = 0; j < w; j++)
					{
							// Calculate inverse matrix (AAT)^-1
							int pos = i * w + j;
							double a, b, c, d;
							a = (double)sumIx2[pos];
							b = c = (double)sumIxIy[pos];
							d = (double)sumIy2[pos];
							double prefix = 1 / (a * d - b * c);
							a *= prefix;
							b *= prefix;
							d *= prefix;

							float u = -d * sumIxIt[pos] + b * sumIyIt[pos];
							float v = c * sumIxIt[pos] - a * sumIyIt[pos];

							optFlowPyramid[level][pos * 2] = u;
							optFlowPyramid[level][pos * 2 + 1] = v;
					}
			}

			// Free all malloc memory
			free(Ix);
			free(Iy);
			free(It1);
			free(It2);

			free(sumIx2);
			free(sumIy2);
			free(sumIxIy);
			free(sumIxIt);
			free(sumIyIt);

			free(shifted);
	}

}
