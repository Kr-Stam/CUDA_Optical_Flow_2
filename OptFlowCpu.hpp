#pragma once

namespace cpu {

 /**
	* \brief Simple inline function to subtract two arrays
	* 
	* \param arr1  a pointer to the first array
	* \param arr2  a pointer to the second array
	* \param n     size of the array
	* \param dest  a pointer to the destination array
	*/
	void sub_arr(unsigned char *arr1, unsigned char *arr2, int n, unsigned char *dest);

 /** 
	* \brief Creates a grayscale image based on the average rgb value of each pixel
	* 
	* \param src   source image 
	* \param dest  destination image
	* \param w     width
	* \param h     height
	* 
	* \details This function expects a 3 color channel image with 8bit color depth.
	*          Because of this the real size of \p src and \p dest should be \p w * \p h * 3. 
	*/
	void grayscale_avg_cpu(const unsigned char *src, unsigned char *dest, int w, int h);

 /**
	* \brief An unoptimized sequential convolution for 3 channel images
	* 
	* \param src   source image
	* \param mask  mask
	* \param dest  destination image
	* \param w     image width
	* \param h     image height
	* \param mw    mask width
	* \param mh    mask height
 	*/
	void conv_3ch(const unsigned char *src, const float *mask, unsigned char *dest, int w, int h, int mw, int mh);

 /**
	* \brief An unoptimized sequential convolution for 3 channel images that reduces the channels from 3 to 1
	*
	*	\param src   source image
	*	\param w     image width
	*	\param h     image height
	*	\param dest  destination image
	*	\param mask  convolution kernel mask
	*	\param mw    mask width
	*	\param mh    mask height
	*
	*	\warning This function assums that the \p src is a grayscale image.
	*/
	void conv_3ch_to_1ch(const unsigned char *src, int w, int h, unsigned char *dest, const float *mask, int mw, int mh);

 /**
	* \brief Downscale the source image, resulting in an image with width \p w and height \p h
	*
	*	\param src   source image
	*	\param w     destination width
	*	\param h     destination height
	*	\param dest  destination image
	*	\param mask  convolution kernel mask
	*	\param mw    mask width
	*	\param mh    mask height
	*
	*	\warning This function assums that the previous width and height are 2 times the size of the target width \p w and the 
	*	         target height \p h
	*
	*	\details This function applies a convolution kernel to filter the source image.
	*	         The destination image is created by sampling the first pixel out of a 2x2 pixel square from the source image.
	*/
	void downscale_gaussian(unsigned char* src, int w, int h, unsigned char * dest, const float* mask, int mw, int mh);

 /**
	* \brief Sequential implementation of a gaussian pyramid, with an arbitrary kernel mask
	*
	*	\param pyramid  a source image at index 0 and an allocated array of progressively smaller images at the other indices
	*	\param w        source image width
	*	\param h        source image height
	*	\param n        number of levels of the pyramid
	*	\param mask     convolution kernel mask
	*	\param mw       mask width
	*	\param mh       mask height
	*
	*	\warning This function does not manage memory, if used all memory management is left to the user.
	*	         The user must allocate the appropriate memory for the pyramid array as well as for each level of the pyramid.
	*
	*	\details This function creates an array of progressively downscaled images with \p n levels.
	*	         Although this function is meant to be used with a gaussian kernel, an arbitrary kernel can be used if needed.
	*/
	void gauss_pyramid(unsigned char **pyramid, int w, int h, int n, const float* mask, int mw, int mh);

 /**
	* \brief Sum reduction and multiplication over window, using 1 color channel
	*
	*	\param pyramid  a source image at index 0 and an allocated array of progressively smaller images at the other indices
	*	\param arr1  matrix1
	*	\param arr2  matrix2
	*	\param w     width
	*	\param h     height
	*	\param ww    window width
	*	\param wh    window height
	*	\param dest  destination matrix
	*
	*	\details Multiplies arr1[i] * arr2[i] for all i belonging to a window around each point of the matrices and then
	*	         performs sum reduction on the results of the window
	*
	*/
	void srm_1ch(const unsigned char *arr1, const unsigned char *arr2, int w, int h, int ww, int wh, int *dest);

 /**
	* \brief Sum reduction and multiplication over window, using 3 color channels
	*
	*	\param pyramid  a source image at index 0 and an allocated array of progressively smaller images at the other indices
	*	\param arr1  matrix1
	*	\param arr2  matrix2
	*	\param w     width
	*	\param h     height
	*	\param ww    window width
	*	\param wh    window height
	*	\param dest  destination matrix
	*
	*	\details Multiplies arr1[i] * arr2[i] for all i belonging to a window around each point of the matrices and then
	*	         performs sum reduction on the results of the window
	*
	*/
	void srm_3ch(unsigned char *arr1, unsigned char *arr2, int w, int h, int ww, int wh, int *dest);

 /**
	* \brief Shifts the image back based on the optical flow so far
	*
	*	\param src             source image
	*	\param w               image width
	*	\param h               image height
	*	\param level           level of the gaussian pyramid
	*	\param maxLevel        maximum level of the gaussian pyramid
	*	\param optFlowPyramid  an array containing the calculated optical flow at each level of the pyramid
	*	\param dest            destination image
	*
	*	\details This function is used to shift each level of the gaussian pyramid back by the optical flow of the level below.
	*	         This enables the algorithm to detect dense optical flow at varying levels of granularity.
	*
	*/
	void shift_back_pyramid(const unsigned char *src, int w, int h, int level, int maxLevel, float **optFlowPyramid, unsigned char *dest);

 /**
	* \brief Solves the inverse matrix in the optical flow equation and calculates the opticalFlow
	*
	*	\param sumIx2
	*	\param sumIy2
	*	\param sumIxIy
	*	\param sumIxIt
	*	\param sumIyIt
	*	\param optFlowPyramid  optical flow pyramid
	*	\param level           current level of the pyramid
	*	\param w               current width
	*	\param h               current height
	*
	*	\details This function is used to shift each level of the gaussian pyramid back by the optical flow of the level below.
	*	         This enables the algorithm to detect dense optical flow at varying levels of granularity.
	*
	*/
	void inverse_matrix(int *sumIx2, int *sumIy2, int *sumIxIy, int *sumIxIt, int *sumIyIt, float **optFlowPyramid, int level, int w, int h);
	
 /**
	* \brief Calculates the optical flow between two succesive images and outputs an optical flow pyramid
	*
	*	\param prev            previous image
	*	\param next            next image
	*	\param w               image width
	*	\param h               image height
	*	\param optFlowPyramid  an allocated array used to output the optical flow field at every level of the pyramid
	*	\param level           level of the Gaussian pyramid
	*	\param maxLevel        maximum level of the Gaussian pyramid
	*
	*	\warning This function does not work properly, probably because of some downlevel function being broken, most likely srm_1ch
	*
	*/
	void calc_optical_flow(const unsigned char *prev, unsigned char *next, int w, int h, float **optFlowPyramid, int level, int maxLevel);

	//TODO: Add doxygen comment
	void bilinear_filter_3ch(unsigned char *src, unsigned char *gray, unsigned char *dest, int w, int h, int ww, int wh, double sigmaS, double sigmaB);
}
