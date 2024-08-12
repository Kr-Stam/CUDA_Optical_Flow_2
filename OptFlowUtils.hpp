#pragma once

#define _USE_MATH_DEFINES

#include <math.h>

namespace utils{

	/**
	 * \brief Cleanup outliers in the image, in order to get a cleaner image
	 *
	 * \param src  source image
	 * \param w    width
	 * \param h    height
	 *
	 * \details This function is used to clean up noisy images.
	 *          It is not the best solution, but it helps minimize noise.
	 */
	void cleanup_outliers(unsigned char* src, int w , int h);

	inline void arr_sub_float(float *arr1, float *arr2, int n, float *dest)
	{
			for (int i = 0; i < n; i++)
			{
					dest[i] = arr1[i] - arr2[i];
					//!Mozhno e da se tunira ova
					// if(dest[i] < 10 && dest[i] > -10){
					//     dest[i] = 0;
					// }
			}
	}

	// Nema da go napravam ova da se paralelizira so CUDA deka ova ne e del od algoritamot, samo go koristam za debagiranje
	/// @brief Upscales an src image with width w and height h, n times. This was mainly used for debugging
	/// @param src Source Image
	/// @param w Width of src
	/// @param h Height of src
	/// @param n Number of times you want to upscale
	/// @param dest Destination
	void upscale_3ch(unsigned char *src, int w, int h, int n, unsigned char *dest);

	// Nema da go napravam ova da se paralelizira so CUDA deka ova ne e del od algoritamot, samo go koristam za debagiranje
	/// @brief Upscales an src image with width w and height h, n times. This was mainly used for debugging
	/// @param src Source Image
	/// @param w Width of src
	/// @param h Height of src
	/// @param n Number of times you want to upscale
	/// @param dest Destination
	void upscale_1ch(unsigned char *src, int w, int h, int n, unsigned char *dest);

	void generate_gaussian_kernel(double sigmaS, int kernel_size, double *dest);
}
