#include <cuda_runtime.h>
#include <device_launch_parameters.h>

//TODO: remove cpu later
#include "OptFlowCpu.hpp"
#include "OptFlowUtils.hpp"
#include "kernels.hpp"

namespace gpu 
{

	/**
	 * \brief CUDA kernel that creates a grayscale image using the average rgb values, each block is a line
	 *
	 * \param src Source Image
	 * \param dest Destination
	 * \param w Width
	 * \param h Height
	 *
	 * \details IMPORTANT: This kernel should be called with a 1D block, each block is one line of the image
	 */
	__global__ void g_grayscale_avg_1d(const unsigned char *src, unsigned char *dest, int w, int h)
	{
			int x = threadIdx.x;
			int y = blockIdx.x;

			if (x >= w || y >= h)
			{
					return;
			}

			int pos = (y * w + x) * 3;
			int avg = (src[pos] + src[pos + 1] + src[pos + 2]) / 3;
			dest[pos] = dest[pos + 1] = dest[pos + 2] = (unsigned char)avg;
	}

	/**
	 * \brief CUDA kernel that creates a grayscale image using the average rgb values, each block is a rectangle
	 *
	 * \param src Source Image
	 * \param dest Destination
	 * \param w Width
	 * \param h Height
	 *
	 * \details IMPORTANT: This kernel should be called with a 2D block, each block is a square of the image
	 */
	__global__ void g_grayscale_avg_2d(const unsigned char *src, unsigned char *dest, int w, int h)
	{
			int x = threadIdx.x + blockIdx.x * blockDim.x;
			int y = blockIdx.y * blockDim.y + threadIdx.y;

			if (x >= w || y >= h)
			{
					return;
			}

			int pos = (y * w + x) * 3;
			int avg = (src[pos] + src[pos + 1] + src[pos + 2]) / 3;
			dest[pos] = dest[pos + 1] = dest[pos + 2] = (unsigned char)avg;
	}

	/**
	 * \brief Launches a CUDA kernel to grayscale an image
	 *
	 * \param srch Source Image
	 * \param dest_h Destination
	 * \param w Width
	 * \param h Height
	 *
	 */
	void grayscale_avg(const unsigned char *src_h, unsigned char *dest_h, int h, int w)
	{

			unsigned char *src_d;
			unsigned char *dest_d;

			size_t size = h * w * 3 * sizeof(unsigned char);

			cudaMalloc((void **)&src_d, size);
			cudaMalloc((void **)&dest_d, size);

			cudaMemcpy(src_d, src_h, size, cudaMemcpyHostToDevice);

			int NUM_OF_THREADS = 32;
			dim3 block_size = dim3(NUM_OF_THREADS, NUM_OF_THREADS);
			int GRID_SIZE_X = (int)ceil((float)w / NUM_OF_THREADS);
			int GRID_SIZE_Y = (int)ceil((float)h / NUM_OF_THREADS);
			dim3 grid_size(GRID_SIZE_X, GRID_SIZE_Y);
			g_grayscale_avg_2d<<<grid_size, block_size>>>(src_d, dest_d, w, h);

			cudaMemcpy(dest_h, dest_d, size, cudaMemcpyDeviceToHost);

			cudaFree(dest_d);
			cudaFree(src_d);
	}

	/**
	 * \brief Unoptimized CUDA kernel for 2D convolution
	 *
	 * \param src Source Matrix
	 * \param mask Mask Matrix
	 * \param dest Destination Matrix
	 * \param w Width
	 * \param h Heigth
	 * \param mw Mask Width
	 * \param mh Mask Height
	 */
	 __global__ void g_conv_3ch_2d(const unsigned char *src, const float *mask, unsigned char *dest, int w, int h, int mw, int mh)
	{
			int x = threadIdx.x + blockDim.x * blockIdx.x;
			int y = threadIdx.y + blockDim.y * blockIdx.y;

			if (x >= w || y >= h)
			{
					return;
			}

			int pos = y * w + x;

			int tmp[3] = {0, 0, 0};

			int hmw = mw >> 1;
			int hmh = mh >> 1;
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

	/**
	 * \brief Launch a CUDA kernel to perform 2D convolution
	 *
	 * \param src Source Matrix
	 * \param dest Destination Matrix
	 * \param w Width
	 * \param h Height
	 * \param mask_t Mask Matrix
	 * \param mw Mask Width <=5
	 * \param mh Mask Height <=5
	 */
	void conv_3ch_2d(const unsigned char *src_h, unsigned char *dest_h, int w, int h, const float *mask_t, int mw, int mh)
	{

			size_t size = w * h * 3 * sizeof(unsigned char);

			unsigned char *src_d;
			unsigned char *dest_d;
			float *mask_d;

			cudaMalloc((void **)&src_d, size);
			cudaMalloc((void **)&dest_d, size);
			cudaMalloc((void **)&mask_d, mw * mh * sizeof(float));

			cudaMemcpy(src_d, src_h, size, cudaMemcpyHostToDevice);
			cudaMemcpy(mask_d, mask_t, mw * mh * sizeof(float), cudaMemcpyHostToDevice);

			int NUM_OF_THREADS = 32;
			dim3 blockSize(NUM_OF_THREADS, NUM_OF_THREADS);
			int GRID_SIZE_X = (int)ceil((float)w / NUM_OF_THREADS);
			int GRID_SIZE_Y = (int)ceil((float)h / NUM_OF_THREADS);
			dim3 gridSize(GRID_SIZE_X, GRID_SIZE_Y);
			g_conv_3ch_2d<<<blockSize, gridSize>>>(src_d, mask_d, dest_d, w, h, mw, mh);

			cudaMemcpy(dest_h, dest_d, size, cudaMemcpyDeviceToHost);

			cudaFree(src_d);
			cudaFree(dest_d);
			cudaFree(mask_d);
	}

	__constant__ float mask[25];

	/**
	 * \brief A more optimized 2D convolution where the mask is loaded into constant GPU memory before execution
	 *
	 * \param src Source Matrix
	 * \param dest Destination Matrix
	 * \param w Width
	 * \param h Height
	 * \param mw Mask Width
	 * \param mh Mask Height
	 */ 
	__global__ void g_conv_3ch_2d_constant(const unsigned char *src, unsigned char *dest, int w, int h, int mw, int mh)
	{

			int x = threadIdx.x + blockDim.x * blockIdx.x;
			int y = threadIdx.y + blockDim.y * blockIdx.y;

			if (x >= w || y >= h)
			{
					return;
			}

			int pos = y * w + x;

			int tmp[3] = {0, 0, 0};

			int hmw = mw >> 1;
			int hmh = mh >> 1;
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

	/**
	 * \brief Launch a CUDA kernel to perform a 2D convolution with constant memory
	 *
	 * \param src Source Matrix
	 * \param dest Destination Matrix
	 * \param w Width
	 * \param h Height
	 * \param mask_t Mask Matrix
	 * \param mw Mask Width <=5
	 * \param mh Mask Height <=5
	 */
	void conv_3ch_2d_constant(const unsigned char *src_h, unsigned char *dest_h, int w, int h, const float *mask_t, int mw, int mh)
	{

			size_t size = w * h * 3 * sizeof(unsigned char);

			unsigned char *src_d;
			unsigned char *dest_d;

			cudaMalloc((void **)&src_d, size);
			cudaMalloc((void **)&dest_d, size);

			cudaMemcpy(src_d, src_h, size, cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(mask, mask_t, mw * mh * sizeof(float));

			int NUM_OF_THREADS = 32;
			dim3 blockSize(NUM_OF_THREADS, NUM_OF_THREADS);
			int GRID_SIZE_X = (int)ceil((float)w / NUM_OF_THREADS);
			int GRID_SIZE_Y = (int)ceil((float)h / NUM_OF_THREADS);
			dim3 gridSize(GRID_SIZE_X, GRID_SIZE_Y);
			g_conv_3ch_2d_constant<<<blockSize, gridSize>>>(src_d, dest_d, w, h, mw, mh);

			cudaMemcpy(dest_h, dest_d, size, cudaMemcpyDeviceToHost);

			cudaFree(src_d);
			cudaFree(dest_d);
	}

	__global__ void g_conv_3ch_tiled(const unsigned char *src, unsigned char *dest, int w, int h, int mw, int mh, int TILE_SIZE_X, int TILE_SIZE_Y){
			//load all data
			//Objasnuvanje za kako raboti, povekje e ova za licna upotreba
			//Se upotrebuva maksimalniot mozhen blockSize shto e 32x32
			//Se loadiraat site vrednosti vnatre vo toj blockSize
			//Se koristi TILE_SIZE shto e 32-mw+1;
			//Za da se loadiraat vrednosti nadvor od src mora da se napravat input indeksi i output indeksi
			//Mapiranjeto na nivo na thread e out(0,0) e na TILE_SIZE, in(0,0) e na BLOCK_SIZE
			//Site threads loadiraat, ama ako threadot e nadvor od TILE_SIZE togash ne e output thread 

			extern __shared__ unsigned char tile[];    

			int hmh = mh >> 1;
			int hmw = mw >> 1;

			int x_o = threadIdx.x + blockIdx.x * TILE_SIZE_X;
			int y_o = threadIdx.y + blockIdx.y * TILE_SIZE_Y;
			int pos_o = x_o + y_o * w; 
			int x_i = x_o - hmw;
			int y_i = y_o - hmh;

			int tile_pos = threadIdx.x + threadIdx.y * blockDim.x;
			if(x_i < 0 || x_i >= w || y_i < 0 || y_i >= h){
					tile[tile_pos * 3] = tile[tile_pos * 3 + 1] = tile[tile_pos * 3 + 2] = 0;
			}else{
					int pos_i = x_i + y_i * w;
					tile[tile_pos * 3] = src[pos_i * 3];
					tile[tile_pos * 3 + 1] = src[pos_i * 3 + 1];
					tile[tile_pos * 3 + 2] = src[pos_i * 3 + 2];
			}

			__syncthreads();

			if(x_o >= w || y_o >= h){
					return;
			}
			if(threadIdx.x >= TILE_SIZE_X || threadIdx.y >= TILE_SIZE_Y){
					return;
			}

			int tmp_x, tmp_y, tmp_pos, mask_pos;
			float tmp[] = {0, 0, 0};
			for(int i = 0; i < mh; i++){
					tmp_y = threadIdx.y + i;
					for(int j = 0; j < mw; j++){
							tmp_x = threadIdx.x + j;
							tmp_pos = tmp_x + tmp_y * blockDim.x;
							mask_pos = j + i * mw;
							tmp[0] += tile[tmp_pos * 3] * mask[mask_pos];
							tmp[1] += tile[tmp_pos * 3 + 1] * mask[mask_pos];
							tmp[2] += tile[tmp_pos * 3 + 2] * mask[mask_pos];
					}
			}
			dest[pos_o * 3] = (unsigned char) tmp[0]; 
			dest[pos_o * 3 + 1] = (unsigned char) tmp[1]; 
			dest[pos_o * 3 + 2] = (unsigned char) tmp[2]; 

			//Tile e indeksiran na nivo na block
			//Odma gi isfrlame site outputs shto se out of bounds na src    
			//
	}

	void conv_3ch_tiled(const unsigned char *src_h, unsigned char *dest_h, int w, int h, const float *mask_t, int mw, int mh)
	{
			size_t size = w * h * 3 * sizeof(unsigned char);

			unsigned char *src_d;
			unsigned char *dest_d;

			cudaMalloc((void **)&src_d, size);
			cudaMalloc((void **)&dest_d, size);

			cudaMemcpy(src_d, src_h, size, cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(mask, mask_t, mw * mh * sizeof(float));

			int NUM_OF_THREADS = 32;
			int TILE_SIZE_X = NUM_OF_THREADS - mw + 1;
			int TILE_SIZE_Y = NUM_OF_THREADS - mh + 1;
			dim3 blockSize(NUM_OF_THREADS, NUM_OF_THREADS);
			//? Mozhe da se optimizira ova
			int GRID_SIZE_X = (int)ceil((float)w / TILE_SIZE_X);
			int GRID_SIZE_Y = (int)ceil((float)h / TILE_SIZE_Y);
			dim3 gridSize(GRID_SIZE_X, GRID_SIZE_Y);
			g_conv_3ch_tiled<<<gridSize, blockSize, blockSize.x * blockSize.y * sizeof(unsigned char) * 3>>>(src_d, dest_d, w, h, mw, mh, TILE_SIZE_X, TILE_SIZE_Y);

			cudaMemcpy(dest_h, dest_d, size, cudaMemcpyDeviceToHost);

			cudaFree(src_d);
			cudaFree(dest_d);
	}

	/// @brief This is CUDA kernel for 2D convolution, reducing the channels from 3 to 1
	/// @param src_h Source Image
	/// @param w Image Width
	/// @param h Image Height
	/// @param dest_h Destination Image
	/// @param mask_t Mask
	/// @param mw Mask Width (<=5)
	__global__ void g_conv_3ch_1ch_constant(unsigned char *src, int w, int h, unsigned char *dest, int mw, int mh)
	{

			int x = threadIdx.x + blockIdx.x * blockDim.x;
			int y = threadIdx.y + blockIdx.y * blockDim.y;

			if (x >= w || y >= h)
			{
					return;
			}

			int pos = y * w + x;

			int hmw = mw >> 1;
			int hmh = mh >> 1;

			int start_x = x - hmw;
			int start_y = y - hmh;

			int tmp_pos, tmp_x, tmp_y, mask_pos;
			int tmp = 0;
			for (int i = 0; i < mh; i++)
			{
					tmp_y = start_y + i;
					if (tmp_y < 0 || tmp_y >= h)
					{
							continue;
					}
					for (int j = 0; j < mw; j++)
					{
							tmp_x = start_x + j;
							if (tmp_x < 0 || tmp_x >= w)
							{
									continue;
							}
							tmp_pos = tmp_y * w + tmp_x;
							mask_pos = i * mw + j;
							if (mask[mask_pos] == 0)
							{
									continue;
							}
							tmp += src[tmp_pos * 3] * mask[mask_pos];
					}
			}
			dest[pos] = (unsigned char)tmp;
	}

	/// @brief This is a non-tiled implementation of a 2D convolution that loads the mask into constant memory
	/// @param src_h Source Image, size = w * h * 3
	/// @param w Image Width
	/// @param h Image Height
	/// @param dest_h Destination, size = w * h
	/// @param mask_t Mask
	/// @param mw Mask Width (<=5)
	/// @param mh Mask Height (<=5)
	void conv_3ch_1ch_constant(const unsigned char *src_h, int w, int h, unsigned char *dest_h, const float *mask_t, int mw, int mh)
	{
			unsigned char *src_d;
			unsigned char *dest_d;

			cudaMalloc((void **)&src_d, w * h * 3 * sizeof(unsigned char));
			cudaMalloc((void **)&dest_d, w * h * sizeof(unsigned char));

			cudaMemcpy(src_d, src_h, w * h * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(mask, mask_t, mw * mh * sizeof(float));

			int NUM_OF_THREADS = 32;
			dim3 blockSize(NUM_OF_THREADS, NUM_OF_THREADS);
			int GRID_SIZE_X = (int)ceil((float)w / (float)NUM_OF_THREADS);
			int GRID_SIZE_Y = (int)ceil((float)h / (float)NUM_OF_THREADS);
			dim3 gridSize(GRID_SIZE_X, GRID_SIZE_Y);
			g_conv_3ch_1ch_constant<<<gridSize, blockSize>>>(src_d, w, h, dest_d, mw, mh);

			cudaDeviceSynchronize();
			cudaMemcpy(dest_h, dest_d, w * h * sizeof(unsigned char), cudaMemcpyDeviceToHost);

			cudaFree(src_d);
			cudaFree(dest_d);
	}

	
	// TODO: OVOJ KOD IMA PROBLEMI, NE TREBA DA SE KORISTI
	// Ova e za kolku treba da se napravi padding na SHMEM za da se loadiraat vrednostite potrebni za konvolucija so maska od golemina
	// padding*2 + 1
#define SHMEM_PADDING 2
#define PRESUMED_NUM_OF_THREADS 32
	// #define TILE_SIZE 36;
	// naive implementation
	/// @brief This is a CUDA kernel for a tiled implementation of a 2D convolution where the mask is in constant memory
	/// @details IMPORTANT: This function is hardcoded to be run with a block size of 32x32, it may not work with other blockSizes
	/// @param src_h Source Image, size = w * h * 3
	/// @param w Image Width
	/// @param h Image Height
	/// @param dest_h Destination, size = w * h
	/// @param mask_t Mask
	/// @param mw Mask Width (<=5)
	/// @param mh Mask Height (<=5)
	__global__ void g_conv_3ch_1ch_tiled(unsigned char *src, int w, int h, unsigned char *dest, int mw, int mh)
	{

			//? ne znam dali da bide refaktorirano nadvor vo konstanta
			//? Mislam deka ke go napravi kodot samo pozbunuvachki
			// #define SHMEM_PADDING 2;
			// #define PRESUMED_NUM_OF_THREADS 32;
			// #define TILE_SIZE 36;

			__shared__ float tile[36 * 36]; // mnogu me nervira ova treba da razgledam ubavo kako rabotat konstanti vo c, samo ke gi zamenam site vrednosti direktno
			//TODO: Realno ova treba da bide so dinamichna extern shared memorija

			int global_x = threadIdx.x + blockIdx.x * blockDim.x;
			int global_y = threadIdx.y + blockIdx.y * blockDim.y;
			if (global_x >= w || global_y >= h)
			{
					return;
			}
			int global_pos = global_y * w + global_x;

			int local_x = threadIdx.x + 2;
			int local_y = threadIdx.y + 2;
			int local_pos = local_y * 36 + local_x;

			int hmw = mw >> 1;
			int hmh = mh >> 1;

			// Load data into tile

			tile[local_pos] = src[global_pos * 3];

			int tmp_global_x, tmp_global_y, tmp_local_x, tmp_local_y, tmp_global_pos, tmp_local_pos;
			// Left excess
			if (local_x == 2)
			{
					for (int i = 0; i < hmw; i++)
					{
							tmp_global_x = global_x - i;

							tmp_local_pos = local_pos - i;
							if (tmp_global_x < 0)
							{
									tile[tmp_local_pos] = 0;
									//? Ne znam dali e ova potrebno ama better safe than sorry
							}
							else
							{
									tmp_global_pos = global_pos - i;
									tile[tmp_local_pos] = src[tmp_global_pos * 3];
							}
					}
			}
			// Right excess
			if (local_x == 32 + 2 - 1)
			{
					for (int i = 0; i < hmw; i++)
					{
							tmp_global_x = global_x + i;
							tmp_local_pos = local_pos + i;
							if (tmp_global_x >= w)
							{
									tile[tmp_local_pos] = 0;
									//? Ne znam dali e ova potrebno ama better safe than sorry
							}
							else
							{
									tmp_global_pos = global_pos + i;
									tile[tmp_local_pos] = src[tmp_global_pos * 3];
							}
					}
			}

			// Top excess
			if (local_y == 2)
			{
					for (int i = 0; i < hmw; i++)
					{
							tmp_global_y = global_y - i;
							tmp_local_y = local_y - i;

							tmp_local_pos = tmp_local_y * 36 + local_x;

							if (tmp_global_y < 0)
							{
									tile[tmp_local_pos] = 0;
									//? Ne znam dali e ova potrebno ama better safe than sorry
							}
							else
							{
									tmp_global_pos = tmp_global_y * w + global_x;
									tile[tmp_local_pos] = src[tmp_global_pos * 3];
							}
					}
			}
			// Bottom excess
			if (local_y == 32 + 2 - 1)
			{
					for (int i = 0; i < hmw; i++)
					{
							tmp_global_y = global_y + i;
							tmp_local_y = local_y + i;

							tmp_local_pos = tmp_local_y * 36 + local_x;

							if (tmp_global_y >= h)
							{
									tile[tmp_local_pos] = 0;
									//? Ne znam dali e ova potrebno ama better safe than sorry
							}
							else
							{
									tmp_global_pos = tmp_global_y * w + global_x;
									tile[tmp_local_pos] = src[tmp_global_pos * 3];
							}
					}
			}

			// Corners
			// TL
			if (local_x == 2 && local_y == 2)
			{
					int local_start_y = local_y - 2;
					int global_start_y = global_y - 2;
					int local_start_x = local_x - 2;
					int global_start_x = global_x - 2;
					for (int i = 0; i < 2; i++)
					{
							tmp_local_y = local_start_y + i;
							tmp_global_y = global_start_y + i;
							for (int j = 0; j < 2; j++)
							{
									tmp_global_x = global_start_x + i;
									tmp_local_x = local_start_x + i;
									tmp_local_pos = tmp_local_y * 36 + tmp_local_x;
									if (tmp_global_y < 0 || tmp_global_x < 0)
									{
											tile[tmp_local_pos] = 0;
									}
									else
									{
											tmp_global_pos = tmp_global_y * w + global_x;
											tile[tmp_local_pos] = src[tmp_global_pos * 3];
									}
							}
					}
			}
			// TR
			if (local_x == 32 + 2 - 1 && local_y == 2)
			{
					int local_start_y = local_y - 2;
					int global_start_y = global_y - 2;
					int local_start_x = local_x;
					int global_start_x = global_x;
					for (int i = 0; i < 2; i++)
					{
							tmp_local_y = local_start_y + i;
							tmp_global_y = global_start_y + i;
							for (int j = 0; j < 2; j++)
							{
									tmp_global_x = global_start_x + i;
									tmp_local_x = local_start_x + i;
									tmp_local_pos = tmp_local_y * 36 + tmp_local_x;
									if (tmp_global_y < 0 || tmp_global_x < 0)
									{
											tile[tmp_local_pos] = 0;
									}
									else
									{
											tmp_global_pos = tmp_global_y * w + global_x;
											tile[tmp_local_pos] = src[tmp_global_pos * 3];
									}
							}
					}
			}
			// BL
			if (local_x == 2 && local_y == 32 - 2 + 1)
			{
					int local_start_y = local_y;
					int global_start_y = global_y;
					int local_start_x = local_x - 2;
					int global_start_x = global_x - 2;
					for (int i = 0; i < 2; i++)
					{
							tmp_local_y = local_start_y + i;
							tmp_global_y = global_start_y + i;
							for (int j = 0; j < 2; j++)
							{
									tmp_global_x = global_start_x + i;
									tmp_local_x = local_start_x + i;
									tmp_local_pos = tmp_local_y * 36 + tmp_local_x;
									if (tmp_global_y < 0 || tmp_global_x < 0)
									{
											tile[tmp_local_pos] = 0;
									}
									else
									{
											tmp_global_pos = tmp_global_y * w + global_x;
											tile[tmp_local_pos] = src[tmp_global_pos * 3];
									}
							}
					}
			}
			// BR
			if (local_x == 32 - 2 + 1 && local_y == 32 - 2 + 1)
			{
					int local_start_y = local_y;
					int global_start_y = global_y;
					int local_start_x = local_x;
					int global_start_x = global_x;
					for (int i = 0; i < 2; i++)
					{
							tmp_local_y = local_start_y + i;
							tmp_global_y = global_start_y + i;
							for (int j = 0; j < 2; j++)
							{
									tmp_global_x = global_start_x + i;
									tmp_local_x = local_start_x + i;
									tmp_local_pos = tmp_local_y * 36 + tmp_local_x;
									if (tmp_global_y < 0 || tmp_global_x < 0)
									{
											tile[tmp_local_pos] = 0;
									}
									else
									{
											tmp_global_pos = tmp_global_y * w + global_x;
											tile[tmp_local_pos] = src[tmp_global_pos * 3];
									}
							}
					}
			}
			// Loading finished

			__syncthreads();

			// Now the convolution code
			int local_start_x = local_x - hmw;
			int local_start_y = local_y - hmh;
			int tmp = 0;
			int mask_pos;
			for (int i = 0; i < mh; i++)
			{
					tmp_local_y = local_start_y + i;

					for (int j = 0; j < mw; j++)
					{
							tmp_local_x = local_start_x + j;

							tmp_local_pos = tmp_local_y * 36 + tmp_local_x;
							mask_pos = i * mw + j;
							tmp += tile[tmp_local_pos] * mask[mask_pos];
					}
			}
			dest[global_pos] = (unsigned char)tmp;
	}


	/// @brief This is a tiled implementation of a 2D convolution that loads the mask into constant memory
	/// @param src_h Source Image, size = w * h * 3
	/// @param w Image Width
	/// @param h Image Height
	/// @param dest_h Destination, size = w * h
	/// @param mask_t Mask
	/// @param mw Mask Width (<=5)
	/// @param mh Mask Height (<=5)
	void conv_3ch_1ch_tiled(const unsigned char *src_h, int w, int h, unsigned char *dest_h, const float *mask_t, int mw, int mh)
	{
			unsigned char *src_d;
			unsigned char *dest_d;

			cudaMalloc((void **)&src_d, w * h * 3 * sizeof(unsigned char));
			cudaMalloc((void **)&dest_d, w * h * sizeof(unsigned char));

			cudaMemcpy(src_d, src_h, w * h * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(mask, mask_t, mw * mh * sizeof(float));

			// Mora ovaa funkcija da se povika so ovaa golemina na blokovi poradi nachinot na koj e napravena shared memorija
			int NUM_OF_THREADS = 32;
			dim3 blockSize(NUM_OF_THREADS, NUM_OF_THREADS);
			int GRID_SIZE_X = (int)ceil((float)w / (float)NUM_OF_THREADS);
			int GRID_SIZE_Y = (int)ceil((float)h / (float)NUM_OF_THREADS);
			dim3 gridSize(GRID_SIZE_X, GRID_SIZE_Y);
			// convolutionGPU2D_3CH_to_1CH_Tiled<<<gridSize, blockSize>>>(src_d, w, h, dest_d, mw, mh);
			g_conv_3ch_1ch_constant<<<gridSize, blockSize>>>(src_d, w, h, dest_d, mw, mh);

			cudaDeviceSynchronize();
			cudaMemcpy(dest_h, dest_d, w * h * sizeof(unsigned char), cudaMemcpyDeviceToHost);

			cudaFree(src_d);
			cudaFree(dest_d);
	}

	//TODO: OVOJ KOD IMA PROBLEM, NE SMEE DA SE KORISTI
	/// @brief This is a CUDA kernel for a tiled implementation of a 2D convolution where the mask is in constant memory
	/// @details IMPORTANT: This function is hardcoded to be run with a block size of 32x32, it may not work with other blockSizes
	/// @param src_h Source Image, size = w * h * 3
	/// @param w Image Width
	/// @param h Image Height
	/// @param dest_h Destination, size = w * h
	/// @param mask_t Mask
	/// @param mw Mask Width (<=5)
	/// @param mh Mask Height (<=5)
	__global__ void g_conv_3ch_1ch_tiled_uchar_float(unsigned char *src, int w, int h, float *dest, int mw, int mh)
	{

		//? ne znam dali da bide refaktorirano nadvor vo konstanta
		//? Mislam deka ke go napravi kodot samo pozbunuvachki
		// #define SHMEM_PADDING 2;
		// #define PRESUMED_NUM_OF_THREADS 32;
		// #define TILE_SIZE 36;

		__shared__ float tile[36 * 36]; // mnogu me nervira ova treba da razgledam ubavo kako rabotat konstanti vo c, samo ke gi zamenam site vrednosti direktno

		int global_x = threadIdx.x + blockIdx.x * blockDim.x;
		int global_y = threadIdx.y + blockIdx.y * blockDim.y;
		if (global_x >= w || global_y >= h)
		{
				return;
		}
		int global_pos = global_y * w + global_x;

		int local_x = threadIdx.x + 2;
		int local_y = threadIdx.y + 2;
		int local_pos = local_y * 36 + local_x;

		int hmw = mw >> 1;
		int hmh = mh >> 1;

		// Load data into tile

		tile[local_pos] = src[global_pos * 3];

		int tmp_global_x, tmp_global_y, tmp_local_x, tmp_local_y, tmp_global_pos, tmp_local_pos;
		// Left excess
		if (local_x == 2)
		{
				for (int i = 0; i < hmw; i++)
				{
						tmp_global_x = global_x - i;

						tmp_local_pos = local_pos - i;
						if (tmp_global_x < 0)
						{
								tile[tmp_local_pos] = 0;
								//? Ne znam dali e ova potrebno ama better safe than sorry
						}
						else
						{
								tmp_global_pos = global_pos - i;
								tile[tmp_local_pos] = src[tmp_global_pos * 3];
						}
				}
		}
		// Right excess
		if (local_x == 32 + 2 - 1)
		{
				for (int i = 0; i < hmw; i++)
				{
						tmp_global_x = global_x + i;
						tmp_local_pos = local_pos + i;
						if (tmp_global_x >= w)
						{
								tile[tmp_local_pos] = 0;
								//? Ne znam dali e ova potrebno ama better safe than sorry
						}
						else
						{
								tmp_global_pos = global_pos + i;
								tile[tmp_local_pos] = src[tmp_global_pos * 3];
						}
				}
		}

		// Top excess
		if (local_y == 2)
		{
				for (int i = 0; i < hmw; i++)
				{
						tmp_global_y = global_y - i;
						tmp_local_y = local_y - i;

						tmp_local_pos = tmp_local_y * 36 + local_x;

						if (tmp_global_y < 0)
						{
								tile[tmp_local_pos] = 0;
								//? Ne znam dali e ova potrebno ama better safe than sorry
						}
						else
						{
								tmp_global_pos = tmp_global_y * w + global_x;
								tile[tmp_local_pos] = src[tmp_global_pos * 3];
						}
				}
		}
		// Bottom excess
		if (local_y == 32 + 2 - 1)
		{
				for (int i = 0; i < hmw; i++)
				{
						tmp_global_y = global_y + i;
						tmp_local_y = local_y + i;

						tmp_local_pos = tmp_local_y * 36 + local_x;

						if (tmp_global_y >= h)
						{
								tile[tmp_local_pos] = 0;
								//? Ne znam dali e ova potrebno ama better safe than sorry
						}
						else
						{
								tmp_global_pos = tmp_global_y * w + global_x;
								tile[tmp_local_pos] = src[tmp_global_pos * 3];
						}
				}
		}

		// Corners
		// TL
		if (local_x == 2 && local_y == 2)
		{
				int local_start_y = local_y - 2;
				int global_start_y = global_y - 2;
				int local_start_x = local_x - 2;
				int global_start_x = global_x - 2;
				for (int i = 0; i < 2; i++)
				{
						tmp_local_y = local_start_y + i;
						tmp_global_y = global_start_y + i;
						for (int j = 0; j < 2; j++)
						{
								tmp_global_x = global_start_x + i;
								tmp_local_x = local_start_x + i;
								tmp_local_pos = tmp_local_y * 36 + tmp_local_x;
								if (tmp_global_y < 0 || tmp_global_x < 0)
								{
										tile[tmp_local_pos] = 0;
								}
								else
								{
										tmp_global_pos = tmp_global_y * w + global_x;
										tile[tmp_local_pos] = src[tmp_global_pos * 3];
								}
						}
				}
		}
		// TR
		if (local_x == 32 + 2 - 1 && local_y == 2)
		{
				int local_start_y = local_y - 2;
				int global_start_y = global_y - 2;
				int local_start_x = local_x;
				int global_start_x = global_x;
				for (int i = 0; i < 2; i++)
				{
						tmp_local_y = local_start_y + i;
						tmp_global_y = global_start_y + i;
						for (int j = 0; j < 2; j++)
						{
								tmp_global_x = global_start_x + i;
								tmp_local_x = local_start_x + i;
								tmp_local_pos = tmp_local_y * 36 + tmp_local_x;
								if (tmp_global_y < 0 || tmp_global_x < 0)
								{
										tile[tmp_local_pos] = 0;
								}
								else
								{
										tmp_global_pos = tmp_global_y * w + global_x;
										tile[tmp_local_pos] = src[tmp_global_pos * 3];
								}
						}
				}
		}
		// BL
		if (local_x == 2 && local_y == 32 - 2 + 1)
		{
				int local_start_y = local_y;
				int global_start_y = global_y;
				int local_start_x = local_x - 2;
				int global_start_x = global_x - 2;
				for (int i = 0; i < 2; i++)
				{
						tmp_local_y = local_start_y + i;
						tmp_global_y = global_start_y + i;
						for (int j = 0; j < 2; j++)
						{
								tmp_global_x = global_start_x + i;
								tmp_local_x = local_start_x + i;
								tmp_local_pos = tmp_local_y * 36 + tmp_local_x;
								if (tmp_global_y < 0 || tmp_global_x < 0)
								{
										tile[tmp_local_pos] = 0;
								}
								else
								{
										tmp_global_pos = tmp_global_y * w + global_x;
										tile[tmp_local_pos] = src[tmp_global_pos * 3];
								}
						}
				}
		}
		// BR
		if (local_x == 32 - 2 + 1 && local_y == 32 - 2 + 1)
		{
				int local_start_y = local_y;
				int global_start_y = global_y;
				int local_start_x = local_x;
				int global_start_x = global_x;
				for (int i = 0; i < 2; i++)
				{
						tmp_local_y = local_start_y + i;
						tmp_global_y = global_start_y + i;
						for (int j = 0; j < 2; j++)
						{
								tmp_global_x = global_start_x + i;
								tmp_local_x = local_start_x + i;
								tmp_local_pos = tmp_local_y * 36 + tmp_local_x;
								if (tmp_global_y < 0 || tmp_global_x < 0)
								{
										tile[tmp_local_pos] = 0;
								}
								else
								{
										tmp_global_pos = tmp_global_y * w + global_x;
										tile[tmp_local_pos] = src[tmp_global_pos * 3];
								}
						}
				}
		}
		// Loading finished

		__syncthreads();

		// Now the convolution code
		int local_start_x = local_x - hmw;
		int local_start_y = local_y - hmh;
		float tmp = 0;
		int mask_pos;
		for (int i = 0; i < mh; i++)
		{
				tmp_local_y = local_start_y + i;

				for (int j = 0; j < mw; j++)
				{
						tmp_local_x = local_start_x + j;

						tmp_local_pos = tmp_local_y * 36 + tmp_local_x;
						mask_pos = i * mw + j;
						tmp += (float) tile[tmp_local_pos] * mask[mask_pos];
				}
		}
		dest[global_pos] = tmp;
	}


	/// @brief This is CUDA kernel for 2D convolution, reducing the channels from 3 to 1
	/// @param src_h Source Image
	/// @param w Image Width
	/// @param h Image Height
	/// @param dest_h Destination Image
	/// @param mask_t Mask
	/// @param mw Mask Width (<=5)
	__global__ void g_conv_3ch_1ch_constant_uchar_float(unsigned char *src, int w, int h, float *dest, int mw, int mh)
	{

			int x = threadIdx.x + blockIdx.x * blockDim.x;
			int y = threadIdx.y + blockIdx.y * blockDim.y;

			if (x >= w || y >= h)
			{
					return;
			}

			int pos = y * w + x;

			int hmw = mw >> 1;
			int hmh = mh >> 1;

			int start_x = x - hmw;
			int start_y = y - hmh;

			int tmp_pos, tmp_x, tmp_y, mask_pos;
			float tmp = 0;
			for (int i = 0; i < mh; i++)
			{
					tmp_y = start_y + i;
					if (tmp_y < 0 || tmp_y >= h)
					{
							continue;
					}
					for (int j = 0; j < mw; j++)
					{
							tmp_x = start_x + j;
							if (tmp_x < 0 || tmp_x >= w)
							{
									continue;
							}
							tmp_pos = tmp_y * w + tmp_x;
							mask_pos = i * mw + j;
							if (mask[mask_pos] == 0)
							{
									continue;
							}
							tmp += (float) src[tmp_pos * 3] * mask[mask_pos];
					}
			}

			//!MOzhno e ova da se tunira
			// if(tmp > -10 && tmp < 10){
			//     tmp = 0;
			// }
			dest[pos] = tmp;
	}

	/// @brief This is a tiled implementation of a 2D convolution that loads the mask into constant memory
	/// @param src_h Source Image, size = w * h * 3
	/// @param w Image Width
	/// @param h Image Height
	/// @param dest_h Destination, size = w * h
	/// @param mask_t Mask
	/// @param mw Mask Width (<=5)
	/// @param mh Mask Height (<=5)
	void conv_3ch_1ch_tiled_uchar_float(const unsigned char *src_h, int w, int h, float *dest_h, const float *mask_t, int mw, int mh)
	{
			unsigned char *src_d;
			float *dest_d;

			cudaMalloc((void **)&src_d, w * h * 3 * sizeof(unsigned char));
			cudaMalloc((void **)&dest_d, w * h * sizeof(float));

			cudaMemcpy(src_d, src_h, w * h * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(mask, mask_t, mw * mh * sizeof(float));

			// Mora ovaa funkcija da se povika so ovaa golemina na blokovi poradi nachinot na koj e napravena shared memorija
			int NUM_OF_THREADS = 32;
			dim3 blockSize(NUM_OF_THREADS, NUM_OF_THREADS);
			int GRID_SIZE_X = (int)ceil((float)w / (float)NUM_OF_THREADS);
			int GRID_SIZE_Y = (int)ceil((float)h / (float)NUM_OF_THREADS);
			dim3 gridSize(GRID_SIZE_X, GRID_SIZE_Y);
			// convolutionGPU2D_3CH_to_1CH_Tiled<<<gridSize, blockSize>>>(src_d, w, h, dest_d, mw, mh);
			g_conv_3ch_1ch_constant_uchar_float<<<gridSize, blockSize>>>(src_d, w, h, dest_d, mw, mh);

			cudaDeviceSynchronize();
			cudaMemcpy(dest_h, dest_d, w * h * sizeof(float), cudaMemcpyDeviceToHost);

			cudaFree(src_d);
			cudaFree(dest_d);
	}

	/// @brief An unoptimized CUDA kernel for 1D convolutions
	/// @param src Source Array
	/// @param mask Mask Array
	/// @param dest Destination Array
	/// @param m Array Size
	/// @param n Mask Size
	/// @return void
	__global__ void g_conv_1d_3ch(const unsigned char *src, const float *mask_t, unsigned char *dest, int m, int n)
	{
			int x = threadIdx.x + blockIdx.x * blockDim.x;

			if (x >= m)
			{
					return;
			}

			int r = n >> 1;
			int start = x - r;

			int temp[3] = {0, 0, 0};
			for (int i = 0; i < n; i++)
			{
					if (start + i >= 0 && start + i <= m)
					{
							temp[0] += (float)src[(start + i) * 3] * mask_t[i];
							temp[1] += (float)src[(start + i) * 3 + 1] * mask_t[i];
							temp[2] += (float)src[(start + i) * 3 + 2] * mask_t[i];
					}
			}
			dest[x * 3] = (unsigned char)temp[0];
			dest[x * 3 + 1] = (unsigned char)temp[1];
			dest[x * 3 + 2] = (unsigned char)temp[2];
	}

	/// @brief Launches a CUDA kernel to perform 1D convolution, not used just made for practice
	/// @param src Sourc Array
	/// @param dest Destination Array
	void conv_1d_3ch(unsigned char* src_h, int w, int h, unsigned char* dest_h)
	{

			float test[9] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1};
			int n = 9;

			size_t size = w * h * 3 * sizeof(unsigned char);

			unsigned char *src_d;
			unsigned char *dest_d;
			float *mask_d;
			cudaMalloc((void **)&src_d, size);
			cudaMalloc((void **)&dest_d, size);
			cudaMalloc((void **)&mask_d, n * sizeof(float));

			cudaMemcpy(src_d, src_h, size, cudaMemcpyHostToDevice);
			cudaMemcpy(mask_d, test, n * sizeof(float), cudaMemcpyHostToDevice);

			int NUM_OF_THREADS = 1024;
			int NUM_OF_BLOCKS = (int)ceil((float)(w * h) / 1024.0);
			g_conv_1d_3ch<<<NUM_OF_BLOCKS, NUM_OF_THREADS>>>(src_d, mask_d, dest_d, size, n);

			cudaMemcpy(dest_h, dest_d, size, cudaMemcpyDeviceToHost);

			cudaFree(src_d);
			cudaFree(dest_d);
			cudaFree(mask_d);
	}

	__constant__ float GAUS_KERNEL_3x3_d[9] = {
			0.0625, 0.125, 0.0625,
			0.125, 0.25, 0.125,
			0.0625, 0.125, 0.0625};

	__global__ void g_gauss_pyramid(const unsigned char *src, int w, int h, unsigned char *dest)
	{

			int x = threadIdx.x + blockIdx.x * blockDim.x;
			int y = threadIdx.y + blockIdx.y * blockDim.y;

			if (x >= w || y >= h)
			{
					return;
			}

			float tmp[3] = {0, 0, 0};
			int start_y = (y << 1) - 1;
			int start_x = (x << 1) - 1;
			for (int p = 0; p < 3; p++)
			{
					for (int q = 0; q < 3; q++)
					{
							int cx = start_x + q;
							int cy = start_y + p;
							if (cx >= 0 && cx < w * 2 && cy >= 0 && cy < h * 2)
							{
									int mask_pos = p * 3 + q;
									int img_pos = (cy * w * 2 + cx) * 3;
									tmp[0] += GAUS_KERNEL_3x3_d[mask_pos] * src[img_pos];
									tmp[1] += GAUS_KERNEL_3x3_d[mask_pos] * src[img_pos + 1];
									tmp[2] += GAUS_KERNEL_3x3_d[mask_pos] * src[img_pos + 2];
							}
					}
			}
			int pos = y * w + x;
			dest[pos * 3] = (unsigned char)tmp[0];
			dest[pos * 3 + 1] = (unsigned char)tmp[1];
			dest[pos * 3 + 2] = (unsigned char)tmp[2];
	}

	//TODO: Refactor properly
	void gauss_pyramid_level(const unsigned char *src_h, int w, int h, unsigned char *dest_h, const float* mask, int mw, int mh)
	{
			unsigned char *src_d;
			unsigned char *dest_d;

			int dw = w << 1;
			int dh = h << 1;
			cudaMalloc((void **)&src_d, dw * dh * 3 * sizeof(unsigned char));
			cudaMalloc((void **)&dest_d, w * h * 3 * sizeof(unsigned char));

			cudaMemcpy(src_d, src_h, dw * dh * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

			int NUM_OF_THREADS = 32;
			dim3 blockSize(NUM_OF_THREADS, NUM_OF_THREADS);
			int GRID_SIZE_X = (int)ceil((float)w / (float)NUM_OF_THREADS);
			int GRID_SIZE_Y = (int)ceil((float)h / (float)NUM_OF_THREADS);
			dim3 gridSize(GRID_SIZE_X, GRID_SIZE_Y);

			gpu::g_gauss_pyramid<<<blockSize, gridSize>>>(src_d, w, h, dest_d);

			cudaMemcpy(dest_h, dest_d, w * h * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

			cudaFree(src_d);
			cudaFree(dest_d);
	}

	//? Mozhebi ke dodadam proverka za dali e dovolno golema slikata vo nivoto za da se koristi GPU ili CPU?
	void gauss_pyramid(unsigned char ** pyramid, int w, int h, int levels, const float* mask, int mw, int mh)
	{

			for (int k = 1; k < levels; k++)
			{
					w = w >> 1;
					h = h >> 1;
					gauss_pyramid_level(pyramid[k - 1], w, h, pyramid[k], mask, mw, mh);
			}
	}

	__global__ void g_srm_3ch_1ch_tiled(unsigned char *arr1, unsigned char *arr2, int *dest, int w, int h, int ww, int wh)
	{

			extern __shared__ unsigned char tile1[];
			extern __shared__ unsigned char tile2[];

			int hwh = wh >> 1;
			int hww = ww >> 1;

			const int TILE_SIZE_X = blockDim.x + (hww << 1);
			const int TILE_SIZE_Y = blockDim.y + (hwh << 1);
			const int TILE_SIZE = TILE_SIZE_X * TILE_SIZE_Y;

			int global_x = threadIdx.x + blockIdx.x * blockDim.x;
			int global_y = threadIdx.y + blockIdx.y * blockDim.y;
			int global_pos = global_y * w + global_x;

			int local_x = threadIdx.x + hww;
			int local_y = threadIdx.y + hwh;
			int local_pos = local_y * TILE_SIZE + local_x;

			// Load all values

			tile1[local_pos] = arr1[global_pos * 3];
			tile2[local_pos] = arr2[global_pos * 3];

			// Load excess
			int tmp_global_x, tmp_global_y, tmp_local_x, tmp_local_y, tmp_global_pos, tmp_local_pos;
			// Left excess
			if (local_x == 2)
			{
					for (int i = 0; i < hww; i++)
					{
							tmp_global_x = global_x - i;

							tmp_local_pos = local_pos - i;
							if (tmp_global_x < 0)
							{
									tile1[tmp_local_pos] = tile2[tmp_local_pos] = 0;
									//? Ne znam dali e ova potrebno ama better safe than sorry
							}
							else
							{
									tmp_global_pos = global_pos - i;
									tile1[tmp_local_pos - i] = arr1[tmp_global_pos * 3];
									tile2[tmp_local_pos - i] = arr2[tmp_global_pos * 3];
							}
					}
			}
			// Right excess
			if (local_x == 32 + 2 - 1)
			{
					for (int i = 0; i < hww; i++)
					{
							tmp_global_x = global_x + i;
							tmp_local_pos = local_pos + i;
							if (tmp_global_x >= w)
							{
									tile1[tmp_local_pos] = tile2[tmp_local_pos] = 0;
									//? Ne znam dali e ova potrebno ama better safe than sorry
							}
							else
							{
									tmp_global_pos = global_pos + i;
									tile1[tmp_local_pos - i] = arr1[tmp_global_pos * 3];
									tile2[tmp_local_pos - i] = arr2[tmp_global_pos * 3];
							}
					}
			}

			// Top excess
			if (local_y == 2)
			{
					for (int i = 0; i < hwh; i++)
					{
							tmp_global_y = global_y - i;
							tmp_local_y = local_y - i;

							tmp_local_pos = tmp_local_y * TILE_SIZE + local_x;

							if (tmp_global_y < 0)
							{
									tile1[tmp_local_pos] = tile2[tmp_local_pos] = 0;
									//? Ne znam dali e ova potrebno ama better safe than sorry
							}
							else
							{
									tmp_global_pos = tmp_global_y * w + global_x;
									tile1[tmp_local_pos - i] = arr1[tmp_global_pos * 3];
									tile2[tmp_local_pos - i] = arr2[tmp_global_pos * 3];
							}
					}
			}
			// Bottom excess
			if (local_y == 32 + 2 - 1)
			{
					for (int i = 0; i < hwh; i++)
					{
							tmp_global_y = global_y + i;
							tmp_local_y = local_y + i;

							tmp_local_pos = tmp_local_y * TILE_SIZE + local_x;

							if (tmp_global_y >= h)
							{
									tile1[tmp_local_pos] = tile2[tmp_local_pos] = 0;
									//? Ne znam dali e ova potrebno ama better safe than sorry
							}
							else
							{
									tmp_global_pos = tmp_global_y * w + global_x;
									tile1[tmp_local_pos - i] = arr1[tmp_global_pos * 3];
									tile2[tmp_local_pos - i] = arr2[tmp_global_pos * 3];
							}
					}
			}
			// Loading finished

			__syncthreads();

			int start_local_x = local_x - hwh;
			int start_local_y = local_y - hwh;

			int tmp = 0;
			for (int i = 0; i < wh; i++)
			{
					tmp_local_y = start_local_y + i;
					if (tmp_local_y < 0 || tmp_local_y >= TILE_SIZE_Y)
					{
							continue;
					}
					for (int j = 0; j < ww; j++)
					{
							tmp_local_x = start_local_x + j;
							if (tmp_local_x < 0 || tmp_local_x >= TILE_SIZE_X)
							{
									continue;
							}
							tmp_local_pos = tmp_local_y * TILE_SIZE_X + tmp_local_x;
							tmp += arr1[tmp_local_pos] * arr2[tmp_local_pos];
					}
			}

			dest[global_pos] = tmp;
	}

	__global__ void g_srm_3ch_1ch(unsigned char *arr1, unsigned char *arr2, int *dest, int w, int h, int ww, int wh)
	{

			int x = threadIdx.x + blockIdx.x * blockDim.x;
			int y = threadIdx.y + blockIdx.y * blockDim.y;

			int pos = y * w + x;

			int start_x = x - (ww >> 1);
			int start_y = y - (wh >> 1);

			int tmp_pos, tmp_x, tmp_y;
			int tmp = 0;

			for (int i = 0; i < wh; i++)
			{
					tmp_y = start_y + i;
					if (tmp_y < 0 || tmp_y >= h)
					{
							continue;
					}
					for (int j = 0; j < ww; j++)
					{
							tmp_x = start_x + j;
							if (tmp_x < 0 || tmp_x >= w)
							{
									continue;
							}
							tmp_pos = tmp_y * w + tmp_x;
							tmp += arr1[tmp_pos * 3] * arr2[tmp_pos * 3];
					}
			}

			dest[pos] = tmp;
	}

	/// @brief CUDA kernel to multiply arr1[i] * arr2[i] for all i belonging to a window around each point of the matrices
	/// @param arr1 Matrix1
	/// @param arr2 Matrix2
	/// @param w Width
	/// @param h Height
	/// @param ww Window Width
	/// @param wh Window Height
	/// @param dest Destination Matrix
	__global__ void g_srm_1ch(const unsigned char *arr1, const unsigned char *arr2, int w, int h, int ww, int wh, int *dest)
	{

			int x = threadIdx.x + blockIdx.x * blockDim.x;
			int y = threadIdx.y + blockIdx.y * blockDim.y;

			if (x >= w || y >= h)
			{
					return;
			}

			int tmp_pos, tmp_x, tmp_y, pos, start_x, start_y;
			int tmp = 0;

			int hww = ww >> 1;
			int hwh = wh >> 1;
			pos = y * w + x;
			start_x = x - hww;
			start_y = y - hwh;

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

	/// @brief Launches a CUDA kernel to multiply arr1[i] * arr2[i] for all i belonging to a window around each point of the matrices
	/// @param arr1_h Matrix1
	/// @param arr2_h Matrix2
	/// @param w Width
	/// @param h Height
	/// @param ww Window Width
	/// @param wh Window Height
	/// @param dest_h Destination Matrix
	void srm_1ch(const unsigned char *arr1_h, const unsigned char *arr2_h, int w, int h, int ww, int wh, int *dest_h)
	{

			unsigned char *arr1_d;
			unsigned char *arr2_d;
			int *dest_d;

			cudaMalloc((void **)&arr1_d, w * h * sizeof(unsigned char));
			cudaMalloc((void **)&arr2_d, w * h * sizeof(unsigned char));
			cudaMalloc((void **)&dest_d, w * h * sizeof(int));

			cudaMemcpy(arr1_d, arr1_h, w * h * sizeof(unsigned char), cudaMemcpyHostToDevice);
			cudaMemcpy(arr2_d, arr2_h, w * h * sizeof(unsigned char), cudaMemcpyHostToDevice);

			int NUM_OF_THREADS = 32;
			dim3 blockSize(NUM_OF_THREADS, NUM_OF_THREADS);
			int GRID_SIZE_X = (int)ceil((float)w / (float)NUM_OF_THREADS);
			int GRID_SIZE_Y = (int)ceil((float)h / (float)NUM_OF_THREADS);
			dim3 gridSize(GRID_SIZE_X, GRID_SIZE_Y);

			g_srm_1ch<<<blockSize, gridSize>>>(arr1_d, arr2_d, w, h, ww, wh, dest_d);

			cudaMemcpy(dest_h, dest_d, w * h * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(arr1_d);
			cudaFree(arr2_d);
			cudaFree(dest_d);
	}

	/// @brief CUDA kernel to multiply arr1[i] * arr2[i] for all i belonging to a window around each point of the matrices
	/// @param arr1 Matrix1
	/// @param arr2 Matrix2
	/// @param w Width
	/// @param h Height
	/// @param ww Window Width
	/// @param wh Window Height
	/// @param dest Destination Matrix
	__global__ void g_srm_1ch_float(const float *arr1, const float *arr2, int w, int h, int ww, int wh, float *dest)
	{

			int x = threadIdx.x + blockIdx.x * blockDim.x;
			int y = threadIdx.y + blockIdx.y * blockDim.y;

			if (x >= w || y >= h)
			{
					return;
			}

			int tmp_pos, tmp_x, tmp_y, pos, start_x, start_y;
			float tmp = 0;

			int hww = ww >> 1;
			int hwh = wh >> 1;
			pos = y * w + x;
			start_x = x - hww;
			start_y = y - hwh;

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

	/// @brief Launches a CUDA kernel to multiply arr1[i] * arr2[i] for all i belonging to a window around each point of the matrices
	/// @param arr1_h Matrix1
	/// @param arr2_h Matrix2
	/// @param w Width
	/// @param h Height
	/// @param ww Window Width
	/// @param wh Window Height
	void srm_1ch_float(const float *arr1_h, const float *arr2_h, int w, int h, int ww, int wh, float *dest_h)
	{

		float *arr1_d;
		float *arr2_d;
		float *dest_d;

		cudaMalloc((void **)&arr1_d, w * h * sizeof(float));
		cudaMalloc((void **)&arr2_d, w * h * sizeof(float));
		cudaMalloc((void **)&dest_d, w * h * sizeof(float));

		cudaMemcpy(arr1_d, arr1_h, w * h * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(arr2_d, arr2_h, w * h * sizeof(float), cudaMemcpyHostToDevice);

		int NUM_OF_THREADS = 32;
		dim3 blockSize(NUM_OF_THREADS, NUM_OF_THREADS);
		int GRID_SIZE_X = (int)ceil((float)w / (float)NUM_OF_THREADS);
		int GRID_SIZE_Y = (int)ceil((float)h / (float)NUM_OF_THREADS);
		dim3 gridSize(GRID_SIZE_X, GRID_SIZE_Y);

		// sumReductionAndMultOverWindowGPU1CH_Tiled<<<blockSize, gridSize>>>(arr1_d, arr2_d, w, h, ww, wh, dest_d);
		g_srm_1ch_float<<<blockSize, gridSize>>>(arr1_d, arr2_d, w, h, ww, wh, dest_d);

		cudaMemcpy(dest_h, dest_d, w * h * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(arr1_d);
		cudaFree(arr2_d);
		cudaFree(dest_d);
	}

	//? Optimization Notes:
	__global__ void g_srm_1ch_tiled(const unsigned char *arr1, const unsigned char *arr2, int w, int h, int ww, int wh, int *dest, int TILE_SIZE_X, int TILE_SIZE_Y)
	{

			extern __shared__ unsigned char shmem[];
			unsigned char* tile1 = shmem;
			unsigned char* tile2 = shmem + w * h;

			int hwh = wh >> 1;
			int hww = ww >> 1;

			int x_o = threadIdx.x + blockIdx.x * TILE_SIZE_X;
			int y_o = threadIdx.y + blockIdx.y * TILE_SIZE_Y;
			int pos_o = x_o + y_o * w;

			int x_i = x_o - hww;
			int y_i = y_o - hwh;
			int tile_pos = threadIdx.x + threadIdx.y * blockDim.x;
			if(x_i < 0 || x_i >= w || y_i < 0 || y_i >= h){
					tile1[tile_pos] = tile2[tile_pos] = 0;
			}else{
					int pos_i = x_i + y_i * w;
					tile1[tile_pos] = arr1[pos_i];
					tile2[tile_pos] = arr2[pos_i];
			}
			// Loading finished

			__syncthreads();

			if(x_o >= w || y_o >= h){
					return;
			}
			if(threadIdx.x >= TILE_SIZE_X || threadIdx.y >= TILE_SIZE_Y){
					return; 
			}

			int tmp = 0;
			int tmp_x, tmp_y, tmp_pos;
			for (int i = 0; i < wh; i++)
			{
					tmp_y = threadIdx.y + i;
					for (int j = 0; j < ww; j++)
					{
							tmp_x = threadIdx.x + j;
							tmp_pos = tmp_x + tmp_y * blockDim.x; 
							tmp += tile1[tmp_pos] * tile2[tmp_pos];
					}
			}

			dest[pos_o] = tmp;
	}

	/// @brief Launches a CUDA kernel to multiply arr1[i] * arr2[i] for all i belonging to a window around each point of the matrices
	/// @param arr1_h Matrix1
	/// @param arr2_h Matrix2
	/// @param w Width
	/// @param h Height
	/// @param ww Window Width
	/// @param wh Window Height
	/// @param dest_h Destination Matrix
	void srm_1ch_tiled(const unsigned char *arr1_h, const unsigned char *arr2_h, int w, int h, int ww, int wh, int *dest_h)
	{

			unsigned char *arr1_d;
			unsigned char *arr2_d;
			int *dest_d;

			cudaMalloc((void **)&arr1_d, w * h * sizeof(unsigned char));
			cudaMalloc((void **)&arr2_d, w * h * sizeof(unsigned char));
			cudaMalloc((void **)&dest_d, w * h * sizeof(int));

			cudaMemcpy(arr1_d, arr1_h, w * h * sizeof(unsigned char), cudaMemcpyHostToDevice);
			cudaMemcpy(arr2_d, arr2_h, w * h * sizeof(unsigned char), cudaMemcpyHostToDevice);

			int NUM_OF_THREADS = 32;
			int TILE_SIZE_X = NUM_OF_THREADS - ww + 1;
			int TILE_SIZE_Y = NUM_OF_THREADS - wh + 1;
			dim3 blockSize(NUM_OF_THREADS, NUM_OF_THREADS);
			int GRID_SIZE_X = (int)ceil((float)w / (float)TILE_SIZE_X);
			int GRID_SIZE_Y = (int)ceil((float)h / (float)TILE_SIZE_Y);
			dim3 gridSize(GRID_SIZE_X, GRID_SIZE_Y);

			g_srm_1ch_tiled<<<blockSize, gridSize, w * h * sizeof(unsigned char) * 2>>>(arr1_d, arr2_d, w, h, ww, wh, dest_d, TILE_SIZE_X, TILE_SIZE_Y);

			cudaMemcpy(dest_h, dest_d, w * h * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(arr1_d);
			cudaFree(arr2_d);
			cudaFree(dest_d);
	}

	/// @brief CUDA kernel for solving the inverse matrix and calculating optical flow
	/// @param sumIx2
	/// @param sumIy2
	/// @param sumIxIy
	/// @param sumIxIt
	/// @param sumIyIt
	/// @param optFlow float* of the destination of the optical flow calculation
	/// @param w Width
	/// @param h Height
	__global__ void g_inv_matrix(int *sumIx2, int *sumIy2, int *sumIxIy, int *sumIxIt, int *sumIyIt, float *optFlow, int w, int h)
	{

			int x = threadIdx.x + blockDim.x * blockIdx.x;
			int y = threadIdx.y + blockDim.y * blockIdx.y;

			if (x >= w || y >= h)
			{
					return;
			}

			int pos = y * w + x;

			double a, b, c, d;
			a = (double)sumIx2[pos];
			b = c = (double)sumIxIy[pos];
			d = (double)sumIy2[pos];
			double prefix = 1 / (a * d - b * c);
			a *= prefix;
			b *= prefix;
			c *= prefix;
			d *= prefix;

			float u = -d * sumIxIt[pos] + b * sumIyIt[pos];
			float v = c * sumIxIt[pos] - a * sumIyIt[pos];

			optFlow[pos * 2] = u;
			optFlow[pos * 2 + 1] = v;
	}

	/// @brief Solves the inverse matrix in the optical flow equation and calculates the opticalFlow
	/// @param sumIx2
	/// @param sumIy2
	/// @param sumIxIy
	/// @param sumIxIt
	/// @param sumIyIt
	/// @param optFlowPyramid Optical Flow Pyramid
	/// @param level Current Level of the pyramid
	/// @param w Current Width
	/// @param h Current Height
	void inverse_matrix(int *sumIx2, int *sumIy2, int *sumIxIy, int *sumIxIt, int *sumIyIt, float **optFlowPyramid, int level, int w, int h)
	{
			int *sumIx2_d;
			int *sumIy2_d;
			int *sumIxIy_d;
			int *sumIxIt_d;
			int *sumIyIt_d;
			float *optFlow_d;

			size_t size = w * h * sizeof(int);
			cudaMalloc((void **)&sumIx2_d, size);
			cudaMalloc((void **)&sumIy2_d, size);
			cudaMalloc((void **)&sumIxIy_d, size);
			cudaMalloc((void **)&sumIxIt_d, size);
			cudaMalloc((void **)&sumIyIt_d, size);

			cudaMemcpy(sumIx2_d, sumIx2, size, cudaMemcpyHostToDevice);
			cudaMemcpy(sumIy2_d, sumIy2, size, cudaMemcpyHostToDevice);
			cudaMemcpy(sumIxIy_d, sumIxIy, size, cudaMemcpyHostToDevice);
			cudaMemcpy(sumIxIt_d, sumIxIt, size, cudaMemcpyHostToDevice);
			cudaMemcpy(sumIyIt_d, sumIyIt, size, cudaMemcpyHostToDevice);

			size_t flowSize = w * h * 2 * sizeof(float);
			cudaMalloc((void **)&optFlow_d, flowSize);

			int NUM_OF_THREADS = 32;
			dim3 blockSize(NUM_OF_THREADS, NUM_OF_THREADS);
			int GRID_SIZE_X = (int)ceil(w / NUM_OF_THREADS);
			int GRID_SIZE_Y = (int)ceil(h / NUM_OF_THREADS);
			dim3 gridSize(GRID_SIZE_X, GRID_SIZE_Y);
			g_inv_matrix<<<blockSize, gridSize>>>(sumIx2_d, sumIy2_d, sumIxIy_d, sumIxIt_d, sumIyIt_d, optFlow_d, w, h);

			cudaMemcpy(optFlowPyramid[level], optFlow_d, flowSize, cudaMemcpyDeviceToHost);

			cudaFree(sumIx2_d);
			cudaFree(sumIy2_d);
			cudaFree(sumIxIy_d);
			cudaFree(sumIxIt_d);
			cudaFree(sumIyIt_d);

			cudaFree(optFlow_d);
	}

	/// @brief CUDA kernel for solving the inverse matrix and calculating optical flow
	/// @param sumIx2
	/// @param sumIy2
	/// @param sumIxIy
	/// @param sumIxIt
	/// @param sumIyIt
	/// @param optFlow float* of the destination of the optical flow calculation
	/// @param w Width
	/// @param h Height
	__global__ void g_inv_matrix_float(float *sumIx2, float *sumIy2, float *sumIxIy, float *sumIxIt, float *sumIyIt, float *optFlow, int w, int h)
	{
			int x = threadIdx.x + blockDim.x * blockIdx.x;
			int y = threadIdx.y + blockDim.y * blockIdx.y;

			if (x >= w || y >= h)
			{
					return;
			}

			int pos = y * w + x;

			double a, b, c, d;
			a = (double)sumIx2[pos];
			b = c = (double)sumIxIy[pos];
			d = (double)sumIy2[pos];
			double prefix = 1 / (a * d - b * c);
			a *= prefix;
			b *= prefix;
			c *= prefix;
			d *= prefix;

			float u = -d * sumIxIt[pos] + b * sumIyIt[pos];
			float v = c * sumIxIt[pos] - a * sumIyIt[pos];

			optFlow[pos * 2] = u;
			optFlow[pos * 2 + 1] = v;
	}

	/// @brief Solves the inverse matrix in the optical flow equation and calculates the opticalFlow
	/// @param sumIx2
	/// @param sumIy2
	/// @param sumIxIy
	/// @param sumIxIt
	/// @param sumIyIt
	/// @param optFlowPyramid Optical Flow Pyramid
	/// @param level Current Level of the pyramid
	/// @param w Current Width
	/// @param h Current Height
	void inverse_matrix_float(float *sumIx2, float *sumIy2, float *sumIxIy, float *sumIxIt, float *sumIyIt, float **optFlowPyramid, int level, int w, int h)
	{
			float *sumIx2_d;
			float *sumIy2_d;
			float *sumIxIy_d;
			float *sumIxIt_d;
			float *sumIyIt_d;
			float *optFlow_d;

			size_t size = w * h * sizeof(float);
			cudaMalloc((void **)&sumIx2_d, size);
			cudaMalloc((void **)&sumIy2_d, size);
			cudaMalloc((void **)&sumIxIy_d, size);
			cudaMalloc((void **)&sumIxIt_d, size);
			cudaMalloc((void **)&sumIyIt_d, size);

			cudaMemcpy(sumIx2_d, sumIx2, size, cudaMemcpyHostToDevice);
			cudaMemcpy(sumIy2_d, sumIy2, size, cudaMemcpyHostToDevice);
			cudaMemcpy(sumIxIy_d, sumIxIy, size, cudaMemcpyHostToDevice);
			cudaMemcpy(sumIxIt_d, sumIxIt, size, cudaMemcpyHostToDevice);
			cudaMemcpy(sumIyIt_d, sumIyIt, size, cudaMemcpyHostToDevice);

			size_t flowSize = w * h * 2 * sizeof(float);
			cudaMalloc((void **)&optFlow_d, flowSize);

			int NUM_OF_THREADS = 32;
			dim3 blockSize(NUM_OF_THREADS, NUM_OF_THREADS);
			int GRID_SIZE_X = (int)ceil(w / NUM_OF_THREADS);
			int GRID_SIZE_Y = (int)ceil(h / NUM_OF_THREADS);
			dim3 gridSize(GRID_SIZE_X, GRID_SIZE_Y);
			g_inv_matrix_float<<<blockSize, gridSize>>>(sumIx2_d, sumIy2_d, sumIxIy_d, sumIxIt_d, sumIyIt_d, optFlow_d, w, h);

			cudaMemcpy(optFlowPyramid[level], optFlow_d, flowSize, cudaMemcpyDeviceToHost);

			cudaFree(sumIx2_d);
			cudaFree(sumIy2_d);
			cudaFree(sumIxIy_d);
			cudaFree(sumIxIt_d);
			cudaFree(sumIyIt_d);

			cudaFree(optFlow_d);
	}

	/// @brief A function that calculates optical flow for a single level of the Gaussian Pyramid using GPU functions
	/// @param prev Previous Image
	/// @param next Next Image
	/// @param w Image Width at this level
	/// @param h Image Height at this level
	/// @param optFlowPyramid An array containing the optical flow field at every level of the pyramid
	/// @param level Level of the Gaussian pyramid
	/// @param maxLevel MaxLevel of the Gaussian pyramid
	void calc_opt_flow(const unsigned char *prev, unsigned char *next, int w, int h, float **optFlowPyramid, int level, int maxLevel)
	{
			// optFlowPyramid is the pyramid of all optical flows
			// optFlowPyramid[i] is the optical flow field, described by a vector (u, v) at each point

			// STEP 0
			// SHIFT NEXT IMAGE BACK BY PREVIOUSLY CALCULATED OPTICAL FLOW
			// Ova se pravi za celiot dosega presmetan optical flow
			unsigned char *shifted = (unsigned char *)malloc(w * h * 3 * sizeof(unsigned char));
			if (level != maxLevel - 1)
			{
					cpu::shift_back_pyramid(next, w, h, level, maxLevel, optFlowPyramid, shifted);
					next = shifted;
			}

			//!Ke vidime dali ova ke raboti
			//!RABOTI!!!!!!!!!
			// STEP 1
			// calculate partial derivatives at all points using kernels for finite differences (Ix, Iy, It)

			float *Ix = (float *)malloc(w * h * sizeof(float));
			gpu::conv_3ch_1ch_tiled_uchar_float(prev, w, h, Ix, Dx_3x3, 3, 3);

			float *Iy = (float *)malloc(w * h * sizeof(float));
			gpu::conv_3ch_1ch_tiled_uchar_float(prev, w, h, Iy, Dy_3x3, 3, 3);

			float *It1 = (float *)malloc(w * h * sizeof(float));
			gpu::conv_3ch_1ch_tiled_uchar_float(prev, w, h, It1, Dt_3x3, 3, 3);
			float *It2 = (float *)malloc(w * h * sizeof(float));
			gpu::conv_3ch_1ch_tiled_uchar_float(next, w, h, It2, Dt_3x3, 3, 3);
			float *It = It1; // ova za da bide podobro optimizirano
			utils::arr_sub_float(It2, It1, w * h, It);

			// STEP 2
			// Calculate sumIx2, sumIy2, sumIxIy, sumIxIt, sumIyIt
			int ww = 9;
			int wh = 9;
			
			float *sumIx2 = (float *)malloc(w * h * sizeof(float));
			gpu::srm_1ch_float(Ix, Ix, w, h, ww, wh, sumIx2);

			float *sumIy2 = (float *)malloc(w * h * sizeof(float));
			gpu::srm_1ch_float(Iy, Iy, w, h, ww, wh, sumIy2);

			float *sumIxIy = (float *)malloc(w * h * sizeof(float));
			gpu::srm_1ch_float(Ix, Iy, w, h, ww, wh, sumIxIy);

			float *sumIxIt = (float *)malloc(w * h * sizeof(float));
			gpu::srm_1ch_float(Ix, It, w, h, ww, wh, sumIxIt);

			float *sumIyIt = (float *)malloc(w * h * sizeof(int));
			gpu::srm_1ch_float(Iy, It, w, h, ww, wh, sumIyIt);

			// STEP 3
			// Calculate the optical flow vector at every point (i, j)
			gpu::inverse_matrix_float(sumIx2, sumIy2, sumIxIy, sumIxIt, sumIyIt, optFlowPyramid, level, w, h);

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

	//ova ima prostor do 10x10
	__constant__ double gaus_kernel_10x10_gpu[100];

	__global__ void g_bilinear_filter(unsigned char *src, unsigned char *gray, unsigned char *dest, int w, int h, int ww, int wh, double sigmaB)
	{
			int x = threadIdx.x + blockIdx.x * blockDim.x;
			int y = threadIdx.y + blockIdx.y * blockDim.y;
			
			if(x < 0 || y < 0 || x >= w || y >= h){
					return;
			}

			int hwh = wh >> 1;
			int hww = ww >> 1;

			const 
			double* gaus_mask = gaus_kernel_10x10_gpu;

			int pos = y * w + x;
			double wsb = 0;

			int start_y = y - hwh;
			int start_x = x - hww;

			double f_ij = gray[pos * 3];

			double tmp[3] = {0, 0, 0};
			for (int m = 0; m < wh; m++)
			{
					int c_y = start_y + m;
					if (c_y < 0 || c_y >= h)
					{
							continue;
					}
					for (int n = 0; n < ww; n++)
					{
							double sigmaB2 = sigmaB * sigmaB;

							int c_x = start_x + n;

							if (c_x < 0 || c_x >= w)
							{
									continue;
							}

							int c_pos = c_y * w + c_x;

							double f_mn = gray[c_pos * 3];
							double k = f_mn - f_ij;
							double k2 = k * k;

							double n_b = 1.0 / (2.0 * M_PI * sigmaB2) * pow(M_E, -0.5 * (k2) / sigmaB2);
							double n_s = gaus_mask[m * ww + n];

							wsb += n_b * n_s;
							tmp[0] += src[c_pos * 3] * n_b * n_s;
							tmp[1] += src[c_pos * 3 + 1] * n_b * n_s;
							tmp[2] += src[c_pos * 3 + 2] * n_b * n_s;
					}
			}
			tmp[0] /= wsb;
			tmp[1] /= wsb;
			tmp[2] /= wsb;

			dest[pos * 3] = (unsigned char)tmp[0];
			dest[pos * 3 + 1] = (unsigned char)tmp[1];
			dest[pos * 3 + 2] = (unsigned char)tmp[2];
	}

	void bilinear_filter(unsigned char *src, unsigned char *gray, unsigned char *dest, int w, int h, int ww, int wh, double sigmaS, double sigmaB)
	{
			double* gaus_mask = (double*) malloc(ww * wh * sizeof(double)); 
			utils::generate_gaussian_kernel(sigmaS, ww, gaus_mask);

			unsigned char* src_d;
			unsigned char* gray_d;
			unsigned char* dest_d;
			
			cudaMalloc((void**) &src_d, w * h * 3 * sizeof(unsigned char));
			cudaMalloc((void**) &gray_d, w * h * 3 * sizeof(unsigned char));
			cudaMalloc((void**) &dest_d, w * h * 3 * sizeof(unsigned char));

			cudaMemcpyToSymbol(gaus_kernel_10x10_gpu, gaus_mask, ww * wh * sizeof(double));

			cudaMemcpy(src_d, src, w * h * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);
			cudaMemcpy(gray_d, gray, w * h * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice);

			int NUM_OF_THREADS = 32;
			dim3 blockSize(NUM_OF_THREADS, NUM_OF_THREADS);
			int GRID_SIZE_X = (int) ceil((float) w / (float) NUM_OF_THREADS);
			int GRID_SIZE_Y = (int) ceil((float) h / (float) NUM_OF_THREADS);
			dim3 gridSize(GRID_SIZE_X, GRID_SIZE_Y);

			g_bilinear_filter<<<blockSize, gridSize>>>(src_d, gray_d, dest_d, w, h, ww, wh, sigmaB);

			cudaMemcpy(dest, dest_d, w * h * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

			cudaFree(src_d);
			cudaFree(gray_d);
			cudaFree(dest_d);

			free(gaus_mask);
	}

}
