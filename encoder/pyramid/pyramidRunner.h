/*
 * pyramidRunner.h
 *
 *  Created on: Jan 26, 2009
 *      Author: alex
 */

#ifndef PYRAMIDRUNNER_H_
#define PYRAMIDRUNNER_H_

#include "common/common.h"
#include "gold/gold.h"
#include "gold/gold2.h"

void goldMVs(x264_t *h, int** mvX, int** mvY, int s, int depth);
void goldMVs2(x264_t *h, int** mvX, int** mvY, int s, int depth);

#endif /* PYRAMIDRUNNER_H_ */
