/*
 * main.c
 *
 *  Created on: Jan 15, 2009
 *      Author: Alex Rothberg (rothberg@mit.edu)
 */

#include "gold.h"
#include <assert.h>
#include <stdio.h>
#include <math.h>

#define LAMBDA 4

/**
 * Gets the pixel value at a given x,y. It clamps the x,y wo be between 0 and width/ height
 * This simulates the behavior of a clamped CUDA texture.
 * It also represents the expected behavior in H.264 (where the border it propagated.
 */
inline u_int8_t getImg(u_int8_t* img, int x, int y, int stride, int width, int height){
	if(x<0) x=0;
	if(y<0) y=0;

	if(x>=width) x=width-1;
	if(y>=height) y=height-1;

	return img[stride * y + x];
}

#define FILTER(a,b,c,d) ((((a+b+1)>>1)+((c+d+1)>>1)+1)>>1)

#define GETNSCALEUP(NAME, SUBSCALE, N)\
	int NAME(u_int8_t* img, int x, int y, int stride, int width, int height){\
	return(			FILTER((int)SUBSCALE(img, x, 	y, 		stride, width, height),\
				 	(int)SUBSCALE(img, x+N, 	y, 		stride,	width, height),\
				 	(int)SUBSCALE(img, x, 	y+N, 	stride,	width, height),\
				 	(int)SUBSCALE(img, x+N, 	y+N, 	stride,	width, height)));\
	}

#define GETNSCALE(NAME, SUBSCALE, N)\
	int NAME(u_int8_t* img, int x, int y, int stride, int width, int height){\
	return(			(int)SUBSCALE(img, x, 	y, 		stride, width, height)\
				+ 	(int)SUBSCALE(img, x+N, 	y, 		stride,	width, height)\
				+ 	(int)SUBSCALE(img, x, 	y+N, 	stride,	width, height)\
				+ 	(int)SUBSCALE(img, x+N, 	y+N, 	stride,	width, height))>>2;\
	}

/**
 * Gets the value of a pixel at x,y where the image is down sampled @ x(2*N).
 * N should be a power of 2
 */
#define GETNSCALE2(NAME, SUBSUM, N, K)\
	int NAME(u_int8_t* img, int x, int y,unsigned stride, unsigned width, unsigned height){\
		return (	SUBSUM(img, x, 		y,		stride, width, height)\
				+ 	SUBSUM(img, x+N, 	y,		stride, width, height)\
				+ 	SUBSUM(img, x, 		y+N, 	stride, width, height)\
				+ 	SUBSUM(img, x+N, 	y+N, 	stride, width, height))>>(2*(K));\
	}

/**
 * Gets the sum of all the pixels in an NxN region. This method + GETNSCALE2 is used to get the downsampled image.
 * The reason for having a separate sum function rather than calling GETNSCALE2 on increasingly smaller Ns is the rounding problem.
 */
#define GETSUM(NAME, SUBSUM, N)\
	unsigned NAME(u_int8_t* img, int x, int y,unsigned stride, unsigned width, unsigned height){\
		return 	((int)SUBSUM(img, 	x, 		y, 		stride, width, height)\
				+ (int)SUBSUM(img, 	x+N, 	y,		stride, width, height)\
				+ (int)SUBSUM(img, 	x, 		y+N,	stride, width, height)\
				+ (int)SUBSUM(img, 	x+N, 	y+N,	stride, width, height));\
	}

GETSUM(get2Sum, getImg, 1)
GETSUM(get4Sum, get2Sum, 2)
GETSUM(get8Sum, get4Sum, 4)

#if ROUNDING == NO_ROUNDING
	GETNSCALE2(get2Scale, 	getImg,		1, 	1)
	GETNSCALE2(get4Scale, 	get2Sum, 	2,	2)
	GETNSCALE2(get8Scale, 	get4Sum, 	4,	3)
	GETNSCALE2(get16Scale, get8Sum, 	8,	4)
#elif ROUNDING == ROUND_DOWN
	GETNSCALE(get2Scale, getImg, 1)
	GETNSCALE(get4Scale, get2Scale, 2)
	GETNSCALE(get8Scale, get4Scale, 4)
	GETNSCALE(get16Scale, get8Scale, 8)
#elif ROUNDING == ROUND_UP
	GETNSCALEUP(get2Scale, getImg, 1)
	GETNSCALEUP(get4Scale, get2Scale, 2)
	GETNSCALEUP(get8Scale, get4Scale, 4)
	GETNSCALEUP(get16Scale, get8Scale, 8)
#endif



/**
 * Gets the SAD for a TILE x TILE square, downsampled at xN.
 *
 * The function used to get the pixel values is SCALE.
 *
 * bx, by represent the block number.
 * For example if bx = 3, and we are downsampled x16, we would be looking at a block that starts at x = TILE * 16 * 3 in the original image.
 * dx,dy represent by how much we have shifted the down sampled image over when looking at the reference frame.
 */
#define SADN(NAME, SCALE, N)\
	unsigned NAME(u_int8_t* current, u_int8_t* last, unsigned stride, unsigned width, unsigned height, int bx, int by, int dx, int dy){\
		assert(by>=0);\
		assert(by>=0);\
		unsigned sum =0;\
		int x,y;\
		for(x = (N)*bx * TILE;x<(N)*(bx*TILE+TILE);x+=(N)){\
			for(y= (N)*by*TILE;y<(N)*(by*TILE+TILE);y+=(N)){\
				sum += abs(SCALE(current, x,y,stride, width, height) - SCALE(last, x+dx*(N),y+dy*(N),stride, width, height));\
			}\
		}\
		return sum;\
	}



SADN(SAD16, get16Scale, 16)
SADN(SAD8, get8Scale, 8)
SADN(SAD4, get4Scale, 4)
SADN(SAD2, get2Scale, 2)
SADN(SAD, getImg, 1)

inline unsigned mvCost(int dx, int dy){
	dx = (abs(dx))<<2;
	dy = (abs(dy))<<2;
	int xCost = round((log2f(dx+1)*2 + 0.718f + !!dx) + .5f);
	int yCost = round((log2f(dy+1)*2 + 0.718f + !!dy) + .5f);
	return (LAMBDA * (xCost+yCost));
}

/**
 * Performs a full search of a given block (bx, by) in the [-s,+s] window. ARound the MVp
 * Down sampling as determined by SADN
 * MVP = (mvX, mvX)
 * Returns the SAD.
 * Returns the MV as minX, minY. Note this is the absolute MV and NOT the delta from the MVp.
 */
#define fullSearchMBN(NAME, SADN)\
	unsigned NAME(u_int8_t* current, u_int8_t* last,unsigned stride, unsigned width, unsigned height, int bx, int by, int s, int mvX, int mvY, int* minX, int* minY){\
		unsigned min = UINT_MAX;\
		int dx,dy;\
		unsigned sad=0;\
		\
		for(dx=-s;dx<=s;dx++){\
			for(dy=-s;dy<=s;dy++){\
				unsigned curSAD = SADN(current, last, stride, width, height, bx, by, dx+mvX, dy+mvY);\
				unsigned cost = curSAD + mvCost(dx, dy);\
	\
				if(cost < min){\
					min = cost;\
					sad = curSAD;\
					*minX = dx+mvX;\
					*minY = dy+mvY;\
				}\
			}\
		}\
	\
		return sad;\
	}

fullSearchMBN(fullSearchMB, SAD)
fullSearchMBN(fullSearchMB2, SAD2)
fullSearchMBN(fullSearchMB4, SAD4)
fullSearchMBN(fullSearchMB8, SAD8)
fullSearchMBN(fullSearchMB16, SAD16)

/**
 * Creates the helper functions for performing a pyramidal search.
 */
#define pyramidSearchHelperN(NAME, NEXT, SEARCH, A)\
	void NAME(u_int8_t* current, u_int8_t* last, unsigned bx, unsigned by, unsigned stride, unsigned width, unsigned height, int s, int* sads[A+1], int* minXs[A+1], int* minYs[A+1], int mvX, int mvY){\
		/*if(mvX != 0 || mvY != 0)*/\
			/*printf("a %i %i\n", mvX, mvY);*/\
		int minX=0, minY=0;\
		unsigned bWidth = (width+((TILE<<A)-1))/(TILE<<A);\
		unsigned bHeight = (height+((TILE<<A)-1))/(TILE<<A);\
		if(bx >=bWidth || by >= bHeight) return;\
		unsigned sad = SEARCH(current, last, stride, width, height, bx, by, s, mvX,mvY,&minX, &minY);\
		sads[0][bx + by * bWidth] = sad;\
		minXs[0][bx + by * bWidth] = minX;\
		minYs[0][bx + by * bWidth] = minY;\
		NEXT(current, last, bx<<1,		by<<1, 		stride, width, height, s, &sads[1], &minXs[1], &minYs[1],minX<<1,minY<<1);\
		NEXT(current, last, (bx<<1)+1,	by<<1,		stride, width, height, s, &sads[1], &minXs[1], &minYs[1],minX<<1,minY<<1);\
		NEXT(current, last, bx<<1,		(by<<1)+1,	stride, width, height, s, &sads[1], &minXs[1], &minYs[1],minX<<1,minY<<1);\
		NEXT(current, last, (bx<<1)+1,	(by<<1)+1,	stride, width, height, s, &sads[1], &minXs[1], &minYs[1],minX<<1,minY<<1);\
	}

/**
 * Performs a search for a given macroblock and then stores the results in sads, minYs, minXs.
 * Uses MVp = (mvX, mvY) in window s
 */
#define SEARCH(NAME, MB_SEARCH)\
	void NAME(u_int8_t* current, u_int8_t* last, unsigned bx, unsigned by,unsigned stride, unsigned width, unsigned height, int s, int* sads[1], int* minXs[1], int* minYs[1], int mvX, int mvY){\
		/*if(mvX !=0 || mvY !=0){*/\
			/*printf("%i %i\n", mvX,mvY);}*/\
		int minX=0, minY=0;\
		unsigned bWidth = (width+(TILE-1))/(TILE);\
		unsigned bHeight = (height+(TILE-1))/(TILE);\
		if(bx >=bWidth || by >= bHeight) return;\
		unsigned sad = MB_SEARCH(current, last, stride, width, height, bx, by, s, mvX,mvY,&minX, &minY);\
		sads[0][by * bWidth + bx] = sad;\
		minXs[0][by * bWidth + bx] = minX;\
		minYs[0][by * bWidth + bx] = minY;\
	}

SEARCH(FullSearch, fullSearchMB)

/*
		if(by * bWidth + bx==20)printf("og20: %i\n", sad);\
		if(bx +by*width==20)printf("og20: %i %i\n", mvX,mvY);\
 */

pyramidSearchHelperN(pyramidSearchHelper2, 	FullSearch,				fullSearchMB2,	1)
pyramidSearchHelperN(pyramidSearchHelper4, 	pyramidSearchHelper2,	fullSearchMB4,	2)
pyramidSearchHelperN(pyramidSearchHelper8, 	pyramidSearchHelper4,	fullSearchMB8,	3)
pyramidSearchHelperN(pyramidSearchHelper16, pyramidSearchHelper8,	fullSearchMB16,	4)

#define pyramidSearchN(NAME, A, HELPER)\
	void NAME(u_int8_t* current, u_int8_t* last, unsigned stride,  unsigned width, unsigned height, int s, int* sads[A+1], int* minXs[A+1], int* minYs[A+1]){\
		unsigned x,y;\
		for(x=0;x<(width+((TILE<<A)-1))/(TILE<<A);x++){\
			for(y=0;y<(height+((TILE<<A)-1))/(TILE<<A);y++){\
				/*printf("%i %i\n", x , y);*/\
				HELPER(current, last, x,y, stride,width, height, s, sads, minXs, minYs,0,0);\
			}\
		}\
	}

pyramidSearchN(pyramidSearch16, 4, pyramidSearchHelper16)
pyramidSearchN(pyramidSearch8, 	3, pyramidSearchHelper8)
pyramidSearchN(pyramidSearch4, 	2, pyramidSearchHelper4)
pyramidSearchN(pyramidSearch2, 	1, pyramidSearchHelper2)
pyramidSearchN(SearchAllMBs, 	0, FullSearch)
