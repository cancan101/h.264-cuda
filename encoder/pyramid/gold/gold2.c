/*
 * gold2.c
 *
 *  Created on: Jan 27, 2009
 *      Author: alex
 */

#include "gold2.h"

unsigned GetSAD(x264_t *h,u_int8_t* currentBlock, u_int8_t* last, unsigned currentStride,unsigned lastStride, unsigned width, unsigned height, int bx, int by, int dx, int dy){
	return h->pixf.fpelcmp[PIXEL_16x16](
			currentBlock,
			currentStride,
			&last[lastStride * ((by*TILE)+dy) + ((bx * TILE)+dx)],
			lastStride );
}

inline unsigned mvCost2(int dx, int dy){
	int xCost = round((log2f(dx+1)*2 + 0.718f + !!dx) + .5f);
	int yCost = round((log2f(dy+1)*2 + 0.718f + !!dy) + .5f);
	return (4 * (xCost+yCost));
}



unsigned static fullSearch2(x264_t *h, x264_frame_t* current, x264_frame_t* last, int bx, int by, int s, int mvX, int mvY, int* minX, int* minY){
	unsigned min = UINT_MAX;
	int dx,dy;
	unsigned sad=0;

	uint8_t *currBlock = &current->plane[0][(current->i_stride[0] * (by*TILE)) + (bx * TILE)];

	for(dx=-s;dx<=s;dx++){
		for(dy=-s;dy<=s;dy++){
			unsigned curSAD =SAD(current->plane[0], last->plane[0], current->i_stride[0],current->i_width[0], current->i_lines[0], bx, by, dx+mvX, dy+mvY);//   GetSAD(h, currBlock, last->plane[0], current->i_stride[0],last->i_stride[0],current->i_width[0], current->i_lines[0], bx, by, dx+mvX, dy+mvY);
			unsigned cost = curSAD + mvCost2((abs(dx))<<2, (abs(dy))<<2);
			//if(bx +by*(current->i_width[0]/16)==232 && dx==-1 && dy==1)printf("g: %i\n", curSAD);

			if(cost < min){
				min = cost;
				sad = curSAD;
				*minX = dx+mvX;
				*minY = dy+mvY;
			}
		}
	}

	return sad;
}

x264_frame_t* downSample(x264_frame_t *frame, x264_t *h){
	x264_frame_init_lowres(h, frame );

	x264_frame_t *newFrame = x264_frame_new(h);

    newFrame->plane[0] = frame->lowres[0];
    newFrame->i_stride[0] = frame->i_stride_lowres;
    newFrame->i_lines[0] = frame->i_lines_lowres;
    newFrame->i_width[0] = frame ->i_width_lowres;

    return newFrame;
}

void pyramidSearchHelper(x264_t *h, x264_frame_t* currents[], x264_frame_t* lasts[], unsigned bx, unsigned by, int A, int s,int* sads[A+1], int* minXs[A+1], int* minYs[A+1], int mvX, int mvY){
	unsigned sad=0;
	int oldX = mvX, oldY = mvY;
	if(0 && A>0){
		sad = fullSearch2(h, currents[A-1],lasts[A-1], bx, by,s,mvX,mvY,&mvX,&mvY);
	}else{
		sad = fullSearch2(h, currents[A],lasts[A], bx, by,s,mvX,mvY,&mvX,&mvY);
	}

	unsigned width = (currents[A]->i_width[0]+((TILE)-1))/(TILE);
	sads[A][bx +by*width] = sad;
	minXs[A][bx +by*width] = mvX;
	minYs[A][bx +by*width] = mvY;
	//if(bx +by*width==232)printf("g: %i\n", sad+mvCost(-8,0));

	if(A>0){
		//printf("ng(%i): %i(%i,%i) %i %i\n",width, bx +by*width,bx,by, mvX,mvY);
		pyramidSearchHelper(h, currents, lasts, bx<<1,by<<1,  A-1, s, sads, minXs, minYs,mvX<<1,mvY<<1);
		pyramidSearchHelper(h, currents, lasts, (bx<<1)+1,by<<1,  A-1, s, sads, minXs, minYs,mvX<<1,mvY<<1);
		pyramidSearchHelper(h, currents, lasts, bx<<1,(by<<1)+1,  A-1, s, sads, minXs, minYs,mvX<<1,mvY<<1);
		pyramidSearchHelper(h, currents, lasts, (bx<<1)+1,(by<<1)+1, A-1, s, sads, minXs, minYs,mvX<<1,mvY<<1);
	}else{


	}

}

void pyramidSearch(x264_t *h, int A, int s,int* sads[], int* minXs[], int* minYs[]){
	unsigned width = h->fenc->i_width[0];
	unsigned height = h->fenc->i_lines[0];

	x264_frame_t* currents[A+1];
	x264_frame_t* lasts[A+1];

	x264_frame_t* current =  h->fenc;
	x264_frame_t* last =  h->fref0[0];

	currents[0]=current;
	lasts[0]=last;

	int step;
	for(step =0;step<A;step++){
		current = downSample(current, h);
		last = downSample(last, h);

		currents[step+1]=current;
		lasts[step+1]=last;

		//printf("%i %i\n", current)
	}
		printf("g2 size: %i %i\n", (width+((TILE<<A)-1))/(TILE<<A), (height+((TILE<<A)-1))/(TILE<<A));
	unsigned x,y;
	for(y=0;y<(height+((TILE<<A)-1))/(TILE<<A);y++){
		for(x=0;x<(width+((TILE<<A)-1))/(TILE<<A);x++){
			pyramidSearchHelper(h, currents, lasts, x,y, A, s, sads, minXs, minYs,0,0);
		}
	}

	for(step =0;step<A;step++){
		x264_frame_delete(currents[step+1]);
		x264_frame_delete(lasts[step+1]);
	}

}
