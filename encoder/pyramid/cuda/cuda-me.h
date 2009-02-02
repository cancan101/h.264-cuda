/*
 * cuda-me.h
 *
 *  Created on: Jan 12, 2009
 *      Author: rothberg
 */

#ifndef CUDAME_H_
#define CUDAME_H_

#ifdef __CUDACC__ //not sure if this is even needed
extern "C"{
#include "hier.h"
#endif

	void cuda_me(x264_t *h, int** mvX, int** mvY);
	void cuda_me2(x264_t *h, int** mvX, int** mvY);


#ifdef __CUDACC__
}
#endif



#endif /* CUDAME_H_ */
