#include "OptFlowUtils.hpp"

namespace utils {

	void cleanup_outliers(unsigned char* src, int w , int h)
	{
			for (int i = 0; i < h; i++)
			{
					for (int j = 0; j < w; j++)
					{
							if (src[i * w + j] >= 240 || src[i * w + j] < 20)
							{
									src[i * w + j] = 0;
							}else{
									src[i * w + j] = 255;
							}
					}
			}
	}

}
