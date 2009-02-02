#ifdef __CUDACC__
extern "C"{
#endif

#include "common/common.h"

#include <cutil_inline.h>
#include <limits.h>

#include "cuda-me.h"

#define SEARCHWINDOW 8
#define LAMBDA 4
#define TILE 16

// declare texture reference for 2D float texture
texture<uint8_t, 2, cudaReadModeElementType> currentTex;
texture<uint8_t, 2, cudaReadModeElementType> lastTex;

__global__ void minSAD(unsigned, int* mvXs,int* mvYs, unsigned *, unsigned *);
__global__ void minSAD2(unsigned bw, int* mvXs,int* mvYs, unsigned* minSAD, unsigned* bestLoc);

//These are the new and better written versions.
__global__ void minSAD2a(unsigned bw, unsigned bwUp, int2* inMV, int2* outMV);
__global__ void minSADa(unsigned bw, unsigned bwUp, int2* inMV, int2* outMV);
uint8_t* last;
unsigned frameNumber=0;

void runCuda(uint8_t *current,  uint8_t *ref, int strideCur, int strideRef, int width, int height,  int** mvX, int** mvY, int A,int  *mvsXD, int* mvsY);

//This is the new and more efficient implementation. It saves device/host copies.
void runCudaA(uint8_t *current,  uint8_t *ref, int strideCur, int strideRef, int width, int height,  int2** mvOut, int A,int2  *mvsX);

void cuda_me(x264_t *h, int** mvX, int** mvY){
	int *mvsXD, *mvsYD;
    int bw = (h->fenc->i_width[0] + 15)/16;
    int bh = (h->fenc->i_lines[0] + 15)/16;

	cudaMalloc((void**)&mvsXD,bh*bw* sizeof(int ));
	cudaMalloc((void**)&mvsYD,bh*bw* sizeof(int ));
	cudaMemset(mvsXD,0,bh*bw* sizeof(int ));
	cudaMemset(mvsYD,0,bh*bw* sizeof(int ));

	runCuda(h->fenc->plane[0], h->fref0[0]->plane[0], h->fenc->i_stride[0], h->fref0[0]->i_stride[0], h->fenc->i_width[0], h->fenc->i_lines[0], mvX, mvY, 0,mvsXD,mvsYD);

	cutilSafeCall(cudaFree(mvsXD));
	cutilSafeCall(cudaFree(mvsYD));
}

void cuda_me2(x264_t *h, int** mvX, int** mvY){
	currentTex.addressMode[0] = cudaAddressModeClamp;
	lastTex.addressMode[0] = cudaAddressModeClamp;

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint8_t>();
	cudaArray* currentArray, *lastArray;

	cutilSafeCall(cudaMallocArray(&currentArray, 	&channelDesc, h->fenc->i_width[0], 		h->fenc->i_lines[0]));
	cutilSafeCall(cudaMallocArray(&lastArray, 		&channelDesc, h->fref0[0]->i_width[0], 	h->fref0[0]->i_lines[0]));

	cutilSafeCall(cudaMemcpy2DToArray(currentArray, 0, 0,
						h->fenc->plane[0], h->fenc->i_stride[0]* sizeof(uint8_t),
						h->fenc->i_width[0] * sizeof(uint8_t),  h->fenc->i_lines[0],
						cudaMemcpyHostToDevice));

	cutilSafeCall(cudaMemcpy2DToArray(lastArray, 0, 0,
						h->fref0[0]->plane[0],  h->fref0[0]->i_stride[0]* sizeof(uint8_t),
						h->fref0[0]->i_width[0] * sizeof(uint8_t), h->fref0[0]->i_lines[0],
						cudaMemcpyHostToDevice));

	cutilSafeCall(cudaBindTextureToArray(currentTex, 	currentArray, 	channelDesc));
	cutilSafeCall(cudaBindTextureToArray(lastTex, 		lastArray, 		channelDesc));

    int bw2 = (h->fenc->i_width[0] + 63)/64;
    int bh2 = (h->fenc->i_lines[0] + 63)/64;

	int bw = (h->fenc->i_width[0] + 15)/16;
	int bh = (h->fenc->i_lines[0] + 15)/16;

	int2* MVFinal=NULL;

	{
		int2 *MVFinalD =NULL;

		{
			int2* mv1D = NULL;
			{
				int2 *mvsD;
				cudaMalloc((void**)&mvsD,bh2*bw2* sizeof(int2));
				cudaMemset(mvsD,0,bh2*bw2* sizeof(int2));
				runCudaA(h->fenc->plane[0], h->fref0[0]->plane[0], h->fenc->i_stride[0], h->fref0[0]->i_stride[0], h->fenc->i_width[0], h->fenc->i_lines[0], &mv1D, 1,mvsD);

				cutilSafeCall(cudaFree(mvsD));
			}

			runCudaA(h->fenc->plane[0], h->fref0[0]->plane[0], h->fenc->i_stride[0], h->fref0[0]->i_stride[0], h->fenc->i_width[0], h->fenc->i_lines[0], &MVFinalD, 0,mv1D);
			cutilSafeCall(cudaFree(mv1D));
		}

		MVFinal=(int2 *)malloc(bh*bw*sizeof(int2 ));
		cudaMemcpy(MVFinal, MVFinalD, bh*bw*sizeof(int2 ), cudaMemcpyDeviceToHost);

		cutilSafeCall(cudaFree(MVFinalD));
	}

	*mvX = (int*)malloc(bh*bw*sizeof(int));
	*mvY = (int*)malloc(bh*bw*sizeof(int));
	int y,x;
	for(y = 0 ; y < bh; y++){
		for(x = 0 ; x < bw ; x++){
			(*mvX)[x+y*bw] = MVFinal[x+y*bw].x;
			(*mvY)[x+y*bw] = MVFinal[x+y*bw].y;

		}
	}
	free(MVFinal);

	cutilSafeCall(cudaFreeArray(currentArray));
	cutilSafeCall(cudaFreeArray(lastArray));

	cudaUnbindTexture(currentTex);
	cudaUnbindTexture(lastTex);
}

void runCuda(uint8_t *current,  uint8_t *ref, int strideCur, int strideRef, int width, int height,  int** mvX, int** mvY, int A,int  *mvsXD, int* mvsYD){
	currentTex.addressMode[0] = cudaAddressModeClamp;
	lastTex.addressMode[0] = cudaAddressModeClamp;


    int bw = (width + (TILE<<A)-1)/((TILE<<A));
    int bh = (height + (TILE<<A)-1)/((TILE<<A));

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint8_t>();
	cudaArray *currentArray, *lastArray;

	cutilSafeCall(cudaMallocArray(&currentArray, 	&channelDesc, width, height));
	cutilSafeCall(cudaMallocArray(&lastArray,		&channelDesc, width, height));

	cutilSafeCall(cudaMemcpy2DToArray(currentArray, 0, 0, 	current, 	strideCur * sizeof(uint8_t), width * sizeof(uint8_t), height,	cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy2DToArray(lastArray, 	0, 0,  	ref, 		strideRef * sizeof(uint8_t), width * sizeof(uint8_t), height, 	cudaMemcpyHostToDevice));

	unsigned * retD,*minSADD;
	cudaMalloc((void**)&retD,bh*bw* sizeof(unsigned ));
	cudaMalloc((void**)&minSADD,bh*bw* sizeof(unsigned ));

	dim3 blocks(bw, bh);
	dim3 threads(2*SEARCHWINDOW+1,2*SEARCHWINDOW+1);

	unsigned * ret=(unsigned *)malloc(bh*bw*sizeof(unsigned ));
	unsigned * retLoc=(unsigned *)malloc(bh*bw*sizeof(unsigned ));

	cutilSafeCall( cudaBindTextureToArray( currentTex, currentArray, channelDesc));
	cutilSafeCall( cudaBindTextureToArray( lastTex, lastArray, channelDesc));

	printf("kernel call...\n");
	switch(A){
		case 0:
			minSAD<<<blocks, threads>>>(bw,mvsXD,mvsYD, retD, minSADD);
			break;
		case 1:
			minSAD2<<<blocks, threads>>>(bw,mvsXD,mvsYD, retD, minSADD);
			break;
	}
	cudaMemcpy(ret, retD, bh*bw*sizeof(unsigned ), cudaMemcpyDeviceToHost);
	cudaMemcpy(retLoc, minSADD, bh*bw*sizeof(unsigned ), cudaMemcpyDeviceToHost);
	printf("results back.\n");

	*mvX = (int *)malloc(bh*bw*sizeof(int ));
	*mvY = (int *)malloc(bh*bw*sizeof(int ));
	int k;
	for(k=0;k<bh*bw;k++){
		(*mvX)[k] = ((int)retLoc[k]%(2*SEARCHWINDOW+1))-(int)SEARCHWINDOW;
		(*mvY)[k]= ((int)retLoc[k]/(2*SEARCHWINDOW+1))-(int)SEARCHWINDOW;
	}

	free(ret);
	free(retLoc);

	cutilSafeCall(cudaFreeArray(currentArray));
	cutilSafeCall(cudaFreeArray(lastArray));

	cudaUnbindTexture(currentTex);
	cudaUnbindTexture(lastTex);

	cutilSafeCall(cudaFree((void*)retD));
	cutilSafeCall(cudaFree((void*)minSADD));
}

void runCudaA(uint8_t *current,  uint8_t *ref, int strideCur, int strideRef, int width, int height,  int2** retD, int A,int2  *inMVD){

    int bw = (width + (TILE<<A)-1)/((TILE<<A));
    int bh = (height + (TILE<<A)-1)/((TILE<<A));

    int bwOld = (width + (TILE<<(A+1))-1)/((TILE<<(A+1)));

	cudaMalloc((void**)retD,bh*bw* sizeof(int2 ));

	dim3 blocks(bw, bh);
	dim3 threads(2*SEARCHWINDOW+1,2*SEARCHWINDOW+1);

	printf("kernel call...\n");
	switch(A){
		case 0:
			minSADa<<<blocks, threads>>>(bw,bwOld,inMVD, *retD);
			break;
		case 1:
			minSAD2a<<<blocks, threads>>>(bw,bwOld,inMVD, *retD);
			break;
	}
	printf("results back.\n");
}

__device__ unsigned mvCost(int dx, int dy){
	dx = (abs(dx))<<2;
	dy = (abs(dy))<<2;
	int xCost = round((log2f(dx+1)*2 + 0.718f + !!dx) + .5f);
	int yCost = round((log2f(dy+1)*2 + 0.718f + !!dy) + .5f);
	return (LAMBDA * (xCost+yCost));
}

__device__ unsigned SAD(ImageTexture a, ImageTexture b, unsigned xStart, unsigned yStart, unsigned height, unsigned width, unsigned dx, unsigned dy){
	unsigned ret = 0;
	for(unsigned x = xStart; x< xStart + width; x++){
		for(unsigned y = yStart; y< yStart + height; y++){
			ret = __usad(tex2D(a, x,y),tex2D(b, x+dx,y+dy),ret);
		}
	}
	return ret;
}

#define FILTER(a,b,c,d) ((((a+b+1)>>1)+((c+d+1)>>1)+1)>>1)

__device__ uint8_t val2(ImageTexture t, unsigned  x, unsigned  y){
	return FILTER(tex2D(t, x, y),tex2D(t, x+1, y),tex2D(t, x, y+1),tex2D(t, x+1, y+1));
}

__device__ uint8_t val4(unsigned  x, unsigned  y, ImageTexture t){
	return (val2(t, x, y)+val2(t, x+2, y)+val2(t, x, y+2)+val2(t, x+2, y+2))>>2;
}

__device__ uint8_t val16(unsigned  x, unsigned  y, ImageTexture t){
	return (val4(x, y, t)+val4(x+4, y, t)+val4(x, y+4, t)+val4(x+4, y+4, t))>>2;
}

__device__ unsigned SAD2(ImageTexture a, ImageTexture b, unsigned xStart, unsigned yStart, unsigned height, unsigned width, unsigned dx, unsigned dy){
	unsigned ret = 0;
	for(unsigned x = xStart; x< xStart + 2*width; x+=2){
		for(unsigned y = yStart; y< yStart + 2*height; y+=2){
			ret = __usad(val2(a, x,y),val2(b, x+dx,y+dy),ret);
		}
	}
	return ret;
}


__global__ void minSAD(unsigned bw, int* mvXs,int* mvYs, unsigned* minSAD, unsigned* bestLoc){
	const unsigned int bx = blockIdx.x;
	const unsigned int by = blockIdx.y;

	const unsigned int tx = threadIdx.x;
	const unsigned int ty = threadIdx.y;

	const int blockTop = TILE* by;
	const int blockLeft = TILE* bx;

	const int dx = tx-SEARCHWINDOW;
	const int dy = ty-SEARCHWINDOW;


	__shared__  unsigned mins[(2*SEARCHWINDOW+1)*(2*SEARCHWINDOW+1)];
	__shared__  unsigned smallestLoc[(2*SEARCHWINDOW+1)*(2*SEARCHWINDOW+1)];
	int oldMVX=mvXs[(bx>>0)+ bw*(by>>0)]<<0;
	int oldMVY=mvYs[(bx>>0)+ bw*(by>>0)]<<0;
	unsigned myLoc =tx+ty *(2*SEARCHWINDOW+1);
	unsigned sum = SAD(currentTex, lastTex, blockLeft,blockTop, TILE,TILE,dx+oldMVX,dy+oldMVY);

	mins[myLoc] = sum+ mvCost(dx,dy);
	smallestLoc[myLoc] = tx+oldMVX+(ty+oldMVY) *(2*SEARCHWINDOW+1);


	__syncthreads();
	unsigned m=0;
	for(unsigned k = ((2*SEARCHWINDOW+1)*(2*SEARCHWINDOW+1));k>1;k=m){
		m = (k+1)>>1;
		if(myLoc<m && (m+myLoc<k))
		{
			if(mins[myLoc] > mins[myLoc+m]){
				smallestLoc[myLoc] = smallestLoc[myLoc+m];
				mins[myLoc] = mins[myLoc+m];
			}
		}

		__syncthreads();
	}

	if(myLoc==0){
		minSAD[bx+by*bw] = mins[0];
		bestLoc[bx+by*bw] = smallestLoc[0];
	}
}

__global__ void minSAD2(unsigned bw, int* mvXs,int* mvYs, unsigned* minSAD, unsigned* bestLoc){
	const unsigned int bx = blockIdx.x;
	const unsigned int by = blockIdx.y;

	const unsigned int tx = threadIdx.x;
	const unsigned int ty = threadIdx.y;

	const int blockTop = 2*TILE* by;
	const int blockLeft = 2*TILE* bx;

	const int dx = (tx-SEARCHWINDOW)<<1;
	const int dy = (ty-SEARCHWINDOW)<<1;

	__shared__  unsigned mins[(2*SEARCHWINDOW+1)*(2*SEARCHWINDOW+1)];
	__shared__  unsigned smallestLoc[(2*SEARCHWINDOW+1)*(2*SEARCHWINDOW+1)];

	unsigned myLoc =tx+ty *(2*SEARCHWINDOW+1);
	unsigned sum = SAD2(currentTex, lastTex, blockLeft,blockTop, TILE,TILE,dx+mvXs[bx+ bw*by],dy+mvYs[bx+ bw*by]);


	mins[myLoc] = sum+ mvCost(dx,dy);
	smallestLoc[myLoc] = myLoc;


	__syncthreads();
	unsigned m=0;
	for(unsigned k = ((2*SEARCHWINDOW+1)*(2*SEARCHWINDOW+1));k>1;k=m){
		m = (k+1)>>1;
		if(myLoc<m && (m+myLoc<k))
		{
			if(mins[myLoc] > mins[myLoc+m]){
				smallestLoc[myLoc] = smallestLoc[myLoc+m];
				mins[myLoc] = mins[myLoc+m];
			}
		}

		__syncthreads();
	}

	if(myLoc==0){
		minSAD[bx+by*bw] = mins[0];
		bestLoc[bx+by*bw] = smallestLoc[0];
	}
}

__global__ void minSAD2a(unsigned bw, unsigned bwUp, int2* inMV, int2* outMV){
	const unsigned int bx = blockIdx.x;
	const unsigned int by = blockIdx.y;

	const unsigned int tx = threadIdx.x;
	const unsigned int ty = threadIdx.y;

	const int blockTop = 2*TILE* by;//
	const int blockLeft = 2*TILE* bx;

	const int dx = (tx-SEARCHWINDOW)*2;//
	const int dy = (ty-SEARCHWINDOW)*2;

	__shared__  unsigned mins[(2*SEARCHWINDOW+1)*(2*SEARCHWINDOW+1)];
	__shared__  int2 smallestLoc[(2*SEARCHWINDOW+1)*(2*SEARCHWINDOW+1)];

	unsigned myLoc =tx+ty *(2*SEARCHWINDOW+1);
	int2 oldMV = inMV[(bx>>1)+ (bwUp*(by>>1))];
	oldMV.x *= 2;
	oldMV.y *= 2;

	unsigned sum = SAD2(currentTex, lastTex, blockLeft,blockTop, TILE,TILE,dx+oldMV.x,dy+oldMV.y);//


	mins[myLoc] = sum+ mvCost(dx,dy);

	smallestLoc[myLoc].x = dx+oldMV.x;
	smallestLoc[myLoc].y = dy+oldMV.y;

	__syncthreads();
	unsigned m=0;
	for(unsigned k = ((2*SEARCHWINDOW+1)*(2*SEARCHWINDOW+1));k>1;k=m){
		m = (k+1)>>1;
		if(myLoc<m && (m+myLoc<k))
		{
			if(mins[myLoc] > mins[myLoc+m]){
				smallestLoc[myLoc] = smallestLoc[myLoc+m];
				mins[myLoc] = mins[myLoc+m];
			}
		}

		__syncthreads();
	}

	if(myLoc==0){
		outMV[bx+by*bw] = smallestLoc[0];
	}
}

__global__ void minSADa(unsigned bw, unsigned bwUp, int2* inMV, int2* outMV){
	const unsigned int bx = blockIdx.x;
	const unsigned int by = blockIdx.y;

	const unsigned int tx = threadIdx.x;
	const unsigned int ty = threadIdx.y;

	const int blockTop = TILE* by;//
	const int blockLeft = TILE* bx;

	const int dx = (tx-SEARCHWINDOW);//
	const int dy = (ty-SEARCHWINDOW);

	__shared__  unsigned mins[(2*SEARCHWINDOW+1)*(2*SEARCHWINDOW+1)];
	__shared__  int2 smallestLoc[(2*SEARCHWINDOW+1)*(2*SEARCHWINDOW+1)];

	unsigned myLoc =tx+ty *(2*SEARCHWINDOW+1);
	int2 oldMV = inMV[(bx>>1)+ (bwUp*(by>>1))];
	oldMV.x *= 2;
	oldMV.y *= 2;

	unsigned sum = SAD(currentTex, lastTex, blockLeft,blockTop, TILE,TILE,dx+oldMV.x,dy+oldMV.y);//

	mins[myLoc] = sum+ mvCost(dx,dy);

	smallestLoc[myLoc].x = dx+oldMV.x;
	smallestLoc[myLoc].y = dy+oldMV.y;

	__syncthreads();
	unsigned m=0;
	for(unsigned k = ((2*SEARCHWINDOW+1)*(2*SEARCHWINDOW+1));k>1;k=m){
		m = (k+1)>>1;
		if(myLoc<m && (m+myLoc<k))
		{
			if(mins[myLoc] > mins[myLoc+m]){
				smallestLoc[myLoc] = smallestLoc[myLoc+m];
				mins[myLoc] = mins[myLoc+m];
			}
		}

		__syncthreads();
	}

	if(myLoc==0){
		outMV[bx+by*bw] = smallestLoc[0];
	}
}



#ifdef __CUDACC__
}
#endif
