#include "pyramidRunner.h"

void goldMVs(x264_t *h, int** mvX, int** mvY, int s, int depth){

    int* minXs[depth];
    int* minYs[depth];
    int* sads[depth];

    int width = h->fenc->i_width[0];
    int height = h->fenc->i_lines[0];

    int k;
    for(k=depth-1;k>=0;k--){
    	int shift = (depth-k-1);
    	int w = (width+((TILE<<shift)-1))/(TILE<<shift);
    	int h = (height+((TILE<<shift)-1))/(TILE<<shift);

        minXs[k] = (int*)malloc(w*h*sizeof(int));
        minYs[k]= (int*)malloc(w*h*sizeof(int));
        sads[k]= (int*)malloc(w*h*sizeof(int));
    }

    //int bw = ((width+15)/16);

    assert(h->fenc->i_stride[0]==h->fref0[0]->i_stride[0]);

    switch (depth) {
		case 1:
			 SearchAllMBs(h->fenc->plane[0],  h->fref0[0]->plane[0], h->fenc->i_stride[0],  h->fenc->i_width[0], h->fenc->i_lines[0],  s,sads,minXs,minYs);
			break;
		case 2:
			pyramidSearch2(h->fenc->plane[0],  h->fref0[0]->plane[0], h->fenc->i_stride[0],  h->fenc->i_width[0], h->fenc->i_lines[0],  s,sads,minXs,minYs);
			break;
		case 3:
			pyramidSearch4(h->fenc->plane[0],  h->fref0[0]->plane[0], h->fenc->i_stride[0],  h->fenc->i_width[0], h->fenc->i_lines[0],  s,sads,minXs,minYs);
			break;
		default:
			break;
	}


    /*int q;
    for(q =0;q< h->sh.i_last_mb;q++){
    	if(minXs[depth-1][q]!=0 || minYs[depth-1][q] !=0){
    		//printf("%i(%i %i): %i %i = %i\n",q,q%bw,q/bw, minXs[depth-1][q], minYs[depth-1][q], sads[depth-1][q]);
    		printf("%i: %i %i\n",q, minXs[depth-1][q], minYs[depth-1][q]);
    	}
    }*/

    for(k=0;k<depth-1;k++){
        free( minXs[k]);
        free( minYs[k]);
        free( sads[k]);
    }

    *mvX= (minXs[depth-1]);
    *mvY= (minYs[depth-1]);

    free( sads[depth-1]);
}

void goldMVs2(x264_t *h, int** mvX, int** mvY, int s, int depth){

    int* minXs[depth];
    int* minYs[depth];
    int* sads[depth];

    int width = h->fenc->i_width[0];
    int height = h->fenc->i_lines[0];

    int k;
    for(k=depth-1;k>=0;k--){
    	int shift = (depth-k-1);
    	int w = (width+((TILE<<shift)-1))/(TILE<<shift);
    	int h = (height+((TILE<<shift)-1))/(TILE<<shift);

        minXs[shift] = (int*)malloc(w*h*sizeof(int));
        minYs[shift]= (int*)malloc(w*h*sizeof(int));
        sads[shift]= (int*)malloc(w*h*sizeof(int));
    }
    //assert(h->fenc->i_stride[0]==h->fref0[0]->i_stride[0]);
    pyramidSearch(h,depth-1,s,sads,minXs,minYs);

    for(k=1;k<depth;k++){
        free( minXs[k]);
        free( minYs[k]);
        free( sads[k]);
    }

    *mvX= (minXs[0]);
    *mvY= (minYs[0]);

    free( sads[0]);
}
