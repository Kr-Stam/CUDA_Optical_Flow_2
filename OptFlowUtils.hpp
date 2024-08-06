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

}
