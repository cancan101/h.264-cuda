/*
 * main.h
 *
 *  Created on: Jan 15, 2009
 *      Author: alex
 */

#ifndef GOLD_H_
#define GOLD_H_


#include <math.h>
#include <stdlib.h>

#include <limits.h>


#define TILE 16

#define NO_ROUNDING 	0
#define ROUND_UP 		1
#define ROUND_DOWN 		2
#define ROUNDING 		ROUND_UP

//BROKEN
#define CLAMP2(X, MIN, MAX)\
	(X) > (MIN) ? ((X) < (MAX) ? (X) : (MAX)) : (MIN)

#define MAX(A,B)\
	if((A) > (B))\
		(A)\
	else (B)

#define MIN(A,B)\
	if((A) < (B))\
		(A)\
	else (B)


typedef unsigned char u_int8_t;
//typedef unsigned short u_int16_t;

unsigned SAD(u_int8_t* current, u_int8_t* last, unsigned stride, unsigned width, unsigned height, int bx, int by, int dx, int dy);
unsigned SAD2(u_int8_t* current, u_int8_t* last, unsigned stride, unsigned width, unsigned height, int bx, int by, int dx, int dy);
unsigned SAD4(u_int8_t* current, u_int8_t* last, unsigned stride, unsigned width, unsigned height, int bx, int by, int dx, int dy);
unsigned SAD8(u_int8_t* current, u_int8_t* last, unsigned stride, unsigned width, unsigned height, int bx, int by, int dx, int dy);
unsigned SAD16(u_int8_t* current, u_int8_t* last, unsigned stride, unsigned width, unsigned height, int bx, int by, int dx, int dy);

unsigned fullSearchMB(u_int8_t* current, u_int8_t* last, unsigned stride, unsigned width, unsigned height, int bx, int by, int s, int mvX, int mvY, int* minX, int* minY);

unsigned mvCost(int dx, int dy);

unsigned fullSearchMB2(u_int8_t* current, u_int8_t* last, unsigned stride, unsigned width, unsigned height, int bx, int by, int s, int mvX, int mvY, int* minX, int* minY);
void SearchAllMBs(u_int8_t* current, u_int8_t* last, unsigned stride, unsigned width, unsigned height, int s, int* sads[1], int* minXs[1], int* minYs[1]);
void pyramidSearch2(u_int8_t* current, u_int8_t* last, unsigned stride, unsigned width, unsigned height, int s, int* sads[2], int* minXs[2], int* minYs[2]);
void pyramidSearch16(u_int8_t* current, u_int8_t* last, unsigned stride, unsigned width, unsigned height, int s, int* sads[5], int* minXs[5], int* minYs[5]);
void pyramidSearch8(u_int8_t* current, u_int8_t* last, unsigned stride, unsigned width, unsigned height, int s, int* sads[4], int* minXs[4], int* minYs[4]);
void pyramidSearch4(u_int8_t* current, u_int8_t* last, unsigned stride, unsigned width, unsigned height, int s, int* sads[3], int* minXs[3], int* minYs[3]);

void pyramidSearchHelper2(u_int8_t* current, u_int8_t* last, unsigned stride, unsigned bx, unsigned by, unsigned width, unsigned height, int s, int* sads[2], int* minXs[2], int* minYs[2], int mvX, int mvY);

#endif /* GOLD_H_ */
