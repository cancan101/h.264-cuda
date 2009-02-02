/*
 * tests.c
 *
 *  Created on: Jan 16, 2009
 *      Author: alex
 */

#include "tests.h"
//#define CUDA_TEST

unsigned height = 1024;
unsigned width = 1280;

void testx264DownSampling(x264_t *h);

void mainTests(x264_t *h ){
	//testx264DownSampling(h);
	testSAD(h);
	testSearch();
	testPyramidSearch();
	printf("DONE!\n");

}

void testSearch(){
	testSearch1();
	testSearch1a();
	testSearch2();
	testSearch3();
	testSearch4();
	testSearch5();
	testPad1();
	testPad2();
	testPad3();
	testPad4();
}

void testPyramidSearch(){
	testTotalSearch1();
	testTotalSearch2();
	testTotalSearch3a();
	testTotalSearch3();
	testTotalSearch4();
#ifdef CUDA_TEST
	testTotalSearch4cuda();
#endif
}

void testSAD(x264_t *h){
	test1();
	test2();
	test3();
	test3a(h);
	test4();
	test5();
	test6();
	test7();
	test8();
	test9();
	test10();
	test10b();
	test11();
	test12();
	test13();
	test14();
	test15();
	test16();
}

void GetAlter(u_int8_t* current, u_int8_t* last, int width, int height)
{
    unsigned x;
    for(x = 0;x < width;x++){
        unsigned y;
    for(y = 0;y < height;y++){
        	int k = (x % 2 + y % 2) %2;
        	if(k==0){
				current[x + y * width] = 2;
				last[x + y * width] = 0;
        	}else{
				current[x + y * width] = 0;
				last[x + y * width] = 2;
        	}
        }
    }
}

#define GetAlterN(NAME, N,Z, Q)\
	void NAME(u_int8_t* current, u_int8_t* last, int width, int height)\
	{\
		unsigned x;\
		for(x = 0;x < width;x++){\
			unsigned y;\
			for(y = 0;y < height;y++){\
				if(x%(N)==0 && y%(N)==0){\
					current[x + y * width] = (Z)*(Z);\
					last[x + y * width] = 1;\
				}else if(x%(N)==((N)-1) && y%(N)==((N)-1)){\
					current[x + y * width] = (Q);\
					last[x + y * width] = 1;\
				}else{\
					current[x + y * width] = 0;\
					last[x + y * width] = 1;\
				}\
			}\
		}\
	}

GetAlterN(GetAlter4, 4,4, 0)
GetAlterN(GetAlter8, 8,8, 0)
GetAlterN(GetAlter16,16, 15, 31)

void GetAlter2o(u_int8_t* current, u_int8_t* last, int width, int height)
{
    unsigned x;
    for(x = 0;x < width;x++){
        unsigned y;
        for(y = 0;y < height;y++){

        	if(x % 4 == 0 &&  y % 4 ==0){
				current[x + y * width] = 10;
				last[x + y * width] = 1;
        	}else if(x % 4 == 2 &&  y % 4 ==2){
				current[x + y * width] = 22;
				last[x + y * width] = 1;
        	}else{
				current[x + y * width] = 0;
				last[x + y * width] = 1;
        	}
        }
    }
}

void Get10(u_int8_t* current, u_int8_t* last, int width, int height)
{
    unsigned x;
    for(x = 0;x < width;x++){
        unsigned y;
        for(y = 0;y < height;y++){
            current[x + y * width] = 1; //(rand()*255 / (float)RAND_MAX);
            last[x + y * width] = 0; //(rand()*255 / (float)RAND_MAX);
        }
    }
}

void Get11(u_int8_t* current, u_int8_t* last, int width, int height)
{
    unsigned x;
    for(x = 0;x < width;x++){
        unsigned y;
    for(y = 0;y < height;y++){
            current[x + y * width] = 1;
            last[x + y * width] = 1;
        }
    }
}

#define BASIC_TEST(NAME,GEN, SADF, ASSERT, SIZE)\
void NAME(){\
	assert(height % (SIZE) ==0);\
	assert(width % (SIZE) ==0);\
\
	u_int8_t* current = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);\
	u_int8_t* last = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);\
    GEN(current, last, width, height);\
\
    unsigned ret = SADF(current, last, width, width,height, 0,0, 0,0);\
	assert((ASSERT)== ret);\
\
	free(current);\
	free(last);\
}

x264_frame_t* downSample(x264_frame_t *frame, x264_t *h){
	x264_frame_t *newFrame = x264_frame_new(h);

    newFrame->plane[0] = frame->lowres[0];
    newFrame->i_stride[0] = frame->i_stride_lowres;
    newFrame->i_lines[0] = frame->i_lines_lowres;
    newFrame->i_width[0] = frame ->i_width_lowres;

    return newFrame;
}

void testx264DownSampling(x264_t *h){
	x264_frame_t* orig = h->fenc;

	x264_frame_init_lowres(h, orig );
	x264_frame_t* half = downSample(orig,h);

	x264_frame_init_lowres(h, half );
	x264_frame_t* q = downSample(half,h);

	unsigned x,y;
	for(y=0;y<16;y++){
		for(x=0;x<16;x++){
			printf("%i ", orig->plane[0][x + y * orig->i_stride[0]]);
		}
		printf("\n");
	}
	printf("\n");

	for(y=0;y<8;y++){
		for(x=0;x<8;x++){
			printf("%i ", half->plane[0][x + y * half->i_stride[0]]);
		}
		printf("\n");
	}

	for(y=0;y<4;y++){
		for(x=0;x<4;x++){
			printf("%i ", q->plane[0][x + y * q->i_stride[0]]);
		}
		printf("\n");
	}

	//printf("%i %i\n", h->fenc->plane[0][3 + 1 * h->fenc->i_stride[0]], h->fenc->lowres[0][1 + h->fenc->i_stride_lowres*0]);
}



BASIC_TEST(test1,Get10, SAD,TILE*TILE,TILE)
BASIC_TEST(test2,Get11, SAD,0,TILE)
BASIC_TEST(test3,Get10, SAD2,(TILE * TILE),TILE*2)
BASIC_TEST(test4,GetAlter, SAD2,0,TILE*2)
BASIC_TEST(test5,GetAlter, SAD,2*TILE*TILE,TILE)
BASIC_TEST(test6,GetAlter, SAD4,0,TILE*4)
BASIC_TEST(test7,GetAlter4, SAD4,0,TILE*4)
BASIC_TEST(test8,GetAlter4, SAD,TILE/4 * TILE/4 * 15 + TILE*TILE-TILE/4 * TILE/4,TILE*4)

//tests to make sure that we are causing rounding errors by adding dividing early
#if ROUNDING == NO_ROUNDING
	BASIC_TEST(test9,GetAlter2o, SAD4, TILE * TILE ,TILE*4)
#elif ROUNDING == ROUND_DOWN
	BASIC_TEST(test9,GetAlter2o, SAD4, 0 ,TILE*4)
#elif ROUNDING == ROUND_UP
	BASIC_TEST(test9,GetAlter2o, SAD4, 2* TILE * TILE ,TILE*4)
#endif

void Getx1(u_int8_t* current, u_int8_t* last, int width, int height)
{
    unsigned x;
    for(x = 0;x < width;x++){
        unsigned y;
    for(y = 0;y < height;y++){
            current[x + y * width] = (x+1);
            last[x + y * width] = 1;
        }
    }
}

x264_frame_t* frameFromImg(x264_t *h, u_int8_t *img, int width, int height, int stride){
	x264_frame_t *newFrame = x264_frame_new(h);

    newFrame->plane[0] = img;
    newFrame->i_stride[0] =stride;
    newFrame->i_lines[0] = height;
    newFrame->i_width[0] = width;

    return newFrame;
}

void test3a(x264_t *h){
return;
	x264_frame_t *currentFrame = x264_frame_new(h);
	x264_frame_t *lastFrame = x264_frame_new(h);

	int width = currentFrame->i_width[0];
	int height = currentFrame->i_lines[0];

	assert(height % (16*2) ==0);
	assert(width % (16*2) ==0);

	//u_int8_t* current = (u_int8_t*)malloc(sizeof(u_int8_t) * currentFrame->i_stride[0] * height);
	//u_int8_t* last = (u_int8_t*)malloc(sizeof(u_int8_t) * currentFrame->i_stride[0] * height);
	Get10(currentFrame->plane[0], lastFrame->plane[0], currentFrame->i_stride[0], height);

   // x264_frame_t* currentFrame = frameFromImg(h, current, width, height, width);
    //x264_frame_t* lastFrame = frameFromImg(h, last, width, height, width);

	x264_frame_init_lowres(h, currentFrame );
	x264_frame_init_lowres(h, lastFrame );


    unsigned ret = SAD(currentFrame->lowres[0], lastFrame->lowres[0], currentFrame->i_stride_lowres,currentFrame->i_width_lowres, currentFrame->i_lines_lowres, 0,0, 0,0);
	assert(((16 * 16))== ret);

	free(currentFrame);
	free(lastFrame);
}

//remove simple pattern
BASIC_TEST(test10,Getx1, SAD,TILE * (TILE-1) * (1+TILE-1)/2,TILE)

void test10b(){
	assert(height % 16 ==0);
	assert(width % 16 ==0);

	u_int8_t* current = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
	u_int8_t* last = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
	Getx1(current, last, width, height);

    unsigned ret = SAD(current, last, width, width,height, 1,0, 0,0);
    //printf("%i %i\n", ret, TILE * (TILE-1) * TILE/2);
    assert(TILE * (TILE)/2 * (TILE+2*TILE-1) == ret);//16...31
    assert(TILE * (TILE-1) * (1+TILE-1)/2 == SAD(current, last, width, width,height, 0,1, 0,0));


	free(current);
	free(last);
}

void GetModX(u_int8_t* current, u_int8_t* last, int width, int height)
{
    unsigned x;
    for(x = 0;x < width;x++){
        unsigned y;
    for(y = 0;y < height;y++){
            current[x + y * width] = x % 2;
            last[x + y * width] = x % 2;
        }
    }
}

//Check shifting
void test11(){
	assert(height % 16 ==0);
	assert(width % 16 ==0);

	u_int8_t* current = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
	u_int8_t* last = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
	GetModX(current, last, width, height);

    unsigned ret = SAD(current, last, width, width,height, 0,0, 0,0);
    assert(0 == ret);
    assert(TILE *TILE == SAD(current, last, width, width,height, 0,0, 1,0));

	free(current);
	free(last);
}

//Check shifting (and border repeats)
void test12(){
	assert(height % 16 ==0);
	assert(width % 16 ==0);

	u_int8_t* current = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
	u_int8_t* last = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
	GetModX(current, last, width, height);

    unsigned ret = SAD(current, last, width, width,height, 0,0, 0,0);
    assert(0 == ret);
    assert((TILE-1) *TILE == SAD(current, last, width, width,height, 0,0, -1,0));
    assert((TILE) *TILE == SAD(current, last, width, width,height, 1,0, -1,0));
    assert(0 == SAD(current, last, width, width,height, 1,0, -2,0));
    assert(0 == SAD(current, last, width, width,height, 1,0, 0,3));
    assert(0 == SAD(current, last, width, width,height, 0,0, 0,3));
    assert(0 == SAD(current, last, width, width,height, 0,0, 0,-3));
    assert(0 == SAD(current, last, width, width,height, 1,0, 0,-3));

	free(current);
	free(last);
}

void test13(){
	assert(height % 256 ==0);
	assert(width % 256 ==0);

	u_int8_t* current = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
	u_int8_t* last = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
    GetAlter8(current, last, width, height);

    unsigned ret = SAD8(current, last, width, width, height, 0,0, 0,0);
	assert(0== ret);
    //printf("%i %i\n", SAD(current, last, width, width, height, 0,0, 0,0), 4*(15*15-1+(31-1)+(TILE*TILE-2)));

	assert(504== SAD(current, last, width, width, height, 0,0, 0,0));//4*(15*15-1+(31-1)+(TILE*TILE-2))

	free(current);
	free(last);
}

void test14(){
	assert(height % 256 ==0);
	assert(width % 256 ==0);

	u_int8_t* current = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
	u_int8_t* last = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
    GetAlter16(current, last, width, height);

    unsigned ret = SAD16(current, last, width, width, height, 0,0, 0,0);

#if ROUNDING == NO_ROUNDING
	assert(0== ret);
#elif ROUNDING == ROUND_DOWN
	assert(TILE*TILE== ret);
#elif ROUNDING == ROUND_UP
	assert(TILE*TILE== ret);
#endif

	assert(225-1+31-1 + (TILE * TILE-2)== SAD(current, last, width, width, height, 0,0, 0,0));

	free(current);
	free(last);
}

void test15(){
	int height = TILE * 4;
	int width = TILE * 4;

	u_int8_t* current = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
	u_int8_t* last = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
	Get10(current, last, width, height);

    unsigned ret = SAD(current, last, width, width, height, 3,3, 0,0);
	assert(TILE * TILE == ret);
	int minX=0,minY=0;
	assert(TILE*TILE==fullSearchMB(current, last, width, width, height, 3,3,4,0,0,&minX,&minY));


    last[3*TILE+2 +(3*TILE-1)*width] = 1;
    last[3*TILE+2 +(3*TILE-2)*width] = 100;
    last[3*TILE-1 +(3*TILE-1)*width] = 100;
    ret = SAD(current, last, width, width, height, 3,3, 0,-1);
   //printf("%i\n", ret);
    assert(TILE * TILE-1 == ret);
    //printf("%i\n", fullSearchMB(current, last, width, width, height, 3,3,4,0,0,&minX,&minY));
    assert(TILE * TILE-1==fullSearchMB(current, last, width, width, height, 3,3,4,0,0,&minX,&minY));
    assert(minY==-1);
    assert(minX==0);

	assert((TILE * TILE)== SAD(current, last, width, width, height, 0,0, 0,0));
    assert(TILE * TILE==fullSearchMB(current, last, width, width, height, 0,0,4,0,0,&minX,&minY));


	free(current);
	free(last);
}

void test16(){

	int height = TILE * 4;
	int width = TILE * 4;

	u_int8_t* current = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
	u_int8_t* last = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
	Get10(current, last, width, height);

	int minX=0,minY=0;

    last[(3*TILE+2)*width +(3*TILE-1)] = 1;
    last[(3*TILE+2)*width +(3*TILE-2)] = 100;
    last[(3*TILE-1)*width +(3*TILE-1)] = 100;
    unsigned ret = SAD(current, last, width, width, height, 3,3, -1,0);
   //printf("%i\n", ret);
    assert(TILE * TILE-1 == ret);
    assert(TILE * TILE-1==fullSearchMB(current, last, width, width, height, 3,3,4,0,0,&minX,&minY));
    assert(minY==0);
    assert(minX==-1);

	assert((TILE * TILE)== SAD(current, last, width, width, height, 0,0, 0,0));
    assert(TILE * TILE==fullSearchMB(current, last, width, width, height, 0,0,4,0,0,&minX,&minY));


	free(current);
	free(last);
}

void testSearch1(){
	assert(height % 16 ==0);
	assert(width % 16 ==0);

	u_int8_t* current = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
	u_int8_t* last = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);

    Get10(current, last, width, height);
    last[TILE + 7 + width * 7] = 1;

    int minX,minY;

    unsigned ret = fullSearchMB(current, last, width, width, height, 0,0, 8,0,0,&minX,&minY);
	assert(TILE *TILE-1== ret);
	assert(minX==8);

	free(current);
	free(last);
}

void testSearch1a(){
	assert(height % 16 ==0);
	assert(width % 16 ==0);

	u_int8_t* current = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
	u_int8_t* last = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);

    Get10(current, last, width, height);
    last[1 + width * (8+(TILE-1))] = 1;

    int minX,minY;

    unsigned ret = fullSearchMB(current, last, width, width, height, 0,0, 8,0,0,&minX,&minY);
    //printf("%i\n", ret);
	assert(TILE *TILE-1== ret);
	assert( minY==8);

	free(current);
	free(last);
}

void testSearch2(){
	assert(height % 16 ==0);
	assert(width % 16 ==0);

	u_int8_t* current = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
	u_int8_t* last = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);

    Get10(current, last, width, height);
    last[TILE + 7 + width * 7] = 1;
    last[TILE + 6 + width * 7] = 1;

    int minX,minY;

    unsigned ret = fullSearchMB(current, last, width, width, height, 0,0, 8,0,0,&minX,&minY);
	assert((TILE *TILE)-2== ret);
	assert(minX==8);

	free(current);
	free(last);
}

void testSearch3(){
	assert(height % 16 ==0);
	assert(width % 16 ==0);

	u_int8_t* current = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
	u_int8_t* last = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);

    Get10(current, last, width, height);
    last[TILE + 6 + width * 7] = 1;
    last[TILE + 7 + width * 7] = 1;
    last[TILE + 8 + width * 7] = 1;

    int minX,minY;

    unsigned ret = fullSearchMB(current, last, width, width, height, 0,0, 8,0,0,&minX,&minY);
	assert(TILE *TILE-2== ret);
	assert(minX==8);

	free(current);
	free(last);
}

void testSearch4(){
	assert(height % 16 ==0);
	assert(width % 16 ==0);

	u_int8_t* current = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
	u_int8_t* last = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);

    Get10(current, last, width, height);
    last[(TILE-1) + 8 + width * 7] = 1; //right 8
    last[1 + width * (8+(TILE-1))] = 1;

    int minX,minY;

    unsigned ret = fullSearchMB(current, last, width, width, height, 0,0, 8,0,0,&minX,&minY);
    //printf("%i\n", ret);
	assert(TILE *TILE-1== ret);
	assert(minX==8 || minY==8);

	free(current);
	free(last);
}

void testSearch5(){
	assert(height % 16 ==0);
	assert(width % 16 ==0);

	u_int8_t* current = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
	u_int8_t* last = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);

    Get10(current, last, width, height);
    last[0] = 1;

    int minX,minY;

    unsigned ret = fullSearchMB(current, last, width, width, height, 0,0, 8,0,0,&minX,&minY);
    //printf("%i\n", ret);
	assert(TILE *TILE-81== ret);
	assert(minX==-8 && minY==-8);

	free(current);
	free(last);
}

void testPad1(){
	unsigned width = 17;
	unsigned height = 16;

	u_int8_t* current = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
	u_int8_t* last = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);

	Get10(current, last, width, height);
    last[16 + width*1] = 1;
    last[16 + width*3] = 1;

    int minX,minY;

    unsigned ret = fullSearchMB(current, last, width, width, height, 0,0, 16,0,0,&minX,&minY);
    //printf("%i\n", minX);
	assert(TILE * (TILE-2)== ret);
	assert(minX==16);

	free(current);
	free(last);
}

void testPad2(){
	unsigned width = 17;
	unsigned height = 16;

	u_int8_t* current = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
	u_int8_t* last = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);

	Get10(current, last, width, height);
    last[0 + width*1] = 1;
    last[0 + width*3] = 1;

    int minX,minY;

    unsigned ret = fullSearchMB(current, last, width, width, height, 0,0, 16,0,0,&minX,&minY);
    //printf("%i\n", minX);
	assert(TILE * (TILE-2)== ret);
	assert(minX==-16);

	free(current);
	free(last);
}

void testPad3(){
	unsigned width = 16;
	unsigned height = 17;

	u_int8_t* current = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
	u_int8_t* last = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);

	Get10(current, last, width, height);
    last[1 + width*16] = 1;
    last[3 + width*16] = 1;

    int minX,minY;

    unsigned ret = fullSearchMB(current, last, width, width, height, 0,0, 16,0,0,&minX,&minY);
    //printf("%i\n", minX);
	assert(TILE * (TILE-2)== ret);
	assert(minY==16);

	free(current);
	free(last);
}

void testPad4(){
	unsigned width = 16;
	unsigned height = 17;

	u_int8_t* current = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
	u_int8_t* last = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);

	Get10(current, last, width, height);
    last[1 + width*0] = 1;
    last[3 + width*0] = 1;

    int minX,minY;

    unsigned ret = fullSearchMB(current, last, width, width, height, 0,0, 16,0,0,&minX,&minY);
    //printf("%i\n", minX);
	assert(TILE * (TILE-2)== ret);
	assert(minY==-16);

	free(current);
	free(last);
}

void GetBlockCB(u_int8_t* current, u_int8_t* last, int width, int height)
{
    unsigned x;
    for(x = 0;x < width;x++){
        unsigned y;
    for(y = 0;y < height;y++){
            current[x + y * width] = (x/TILE + y/TILE) % 2;
            last[x + y * width] = !((x/TILE + y/TILE) % 2);
        }
    }
}

void testTotalSearch1(){
	unsigned width = TILE*2;
	unsigned height = TILE*2;

	u_int8_t* current = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
	u_int8_t* last = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);

	GetBlockCB(current, last, width, height);

	assert(SAD(current, last, width, width, height, 0,0,0,0)==TILE*TILE);
	assert(SAD(current, last, width, width, height, 1,0,0,0)==TILE*TILE);
	assert(SAD(current, last, width, width, height, 0,1,0,0)==TILE*TILE);
	assert(SAD(current, last, width, width, height, 1,1,0,0)==TILE*TILE);

    int* minX[1];
    int* minY[1];
    int* sad[1];

    minX[0] = (int*)malloc(2*2*sizeof(int));
    minY[0]= (int*)malloc(2*2*sizeof(int));
    sad[0]= (int*)malloc(2*2*sizeof(int));

    SearchAllMBs(current, last, width, width, height,  16,sad,minX,minY);
    assert(sad[0][0] == 0);
    assert(sad[0][1] == 0);
    assert(sad[0][2] == 0);
    assert(sad[0][3] == 0);

    assert(minX[0][0] == 16 || minY[0][0]==16);
    assert(minX[0][1] == -16 || minY[0][1]==16);
    assert(minX[0][2] == 16 || minY[0][2]==-16);
    assert(minX[0][3] == -16 || minY[0][3]==-16);

    free(minX[0]);
    free(minY[0]);
    free(sad[0]);

	free(current);
	free(last);
}

void GetAlterPyr1(u_int8_t* current, u_int8_t* last, int width, int height)
{
    unsigned x;
    for(x = 0;x < width;x++){
        unsigned y;
    for(y = 0;y < height;y++){
        	int k = (x % 2 + y % 2) %2;
        	if(k==0){
				current[x + y * width] = 2;
				last[x + y * width] = 1;
        	}else{
				current[x + y * width] = 0;
				last[x + y * width] = 1;
        	}
        }
    }
}

void testTotalSearch2(){
	unsigned width = TILE*2*2;
	unsigned height = TILE*2*2;

	u_int8_t* current = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
	u_int8_t* last = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);

	GetAlterPyr1(current, last, width, height);

	last[3 + 3*width] += 10;
	current[3 + 3*width] += 10;

	int mvX =0, mvY=0;

	unsigned ret = fullSearchMB2(current, last, width, width, height, 0,0,3,0,0,&mvX, &mvY);
	assert(ret == 0);
	assert(mvX == 0);
	assert(mvY == 0);

	last[3+2*TILE + (3+2*TILE)*width] += 10;
	current[3+2*TILE + (3+2*TILE)*width] += 10;

	//assert(last[3+TILE + (2+TILE)*width]==1);
	//assert(last[2+TILE + (2+TILE)*width]==1);

	//assert(current[3+TILE + (2+TILE)*width]==2 * !(((3+TILE)%2 + (2+TILE)%2)%2));
	//assert(current[3+TILE + (1+TILE)*width]==2 * !(((3+TILE)%2 + (1+TILE)%2)%2));

	assert(SAD2(current, last, width, width, height, 1, 1, 0, 0)==0);

	ret = fullSearchMB2(current, last, width, width, height, 1,1,3,0,0,&mvX, &mvY);
	//printf("%i\n", ret);
	assert(ret == 0);
	assert(mvX == 0);
	assert(mvY == 0);

	free(current);
	free(last);
}

void GetBlockCB2(u_int8_t* current, u_int8_t* last, int width, int height)
{
    unsigned x;
    for(x = 0;x < width;x++){
        unsigned y;
        for(y = 0;y < height;y++){
        	if((x/(2*TILE) + y/(2*TILE)) % 2){
        		if((x % 2 + y % 2) %2){
        			last[x + y * width] = 2;
        		}else{
        			last[x + y * width] = 0;
        		}
    			current[x + y * width] = 1;
        	}else{
    			last[x + y * width] = 100;
    			current[x + y * width] = 1;
        	}
        }
    }
    unsigned y;
    for(y=2*TILE;y<4*TILE;y++){
    	last[(6*TILE) + y * width] = 1;
    }
}

void testTotalSearch3a(){
	unsigned width = TILE*2*4;
	unsigned height = TILE*2*4;

	u_int8_t* current = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
	u_int8_t* last = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);

	GetBlockCB2(current, last, width, height);

    int mvX =0, mvY=0;

    assert(0==SAD2(current, last, width, width, height,1,1,-16,0));
    assert(0==SAD2(current, last, width, width, height,1,1,0,-16));
    assert(TILE*TILE*99==SAD2(current, last, width, width, height,1,1,0,0));
	//printf("%i\n", SAD2(current, last, width, width, height,1,1,-16,-16));
    //assert(0==SAD2(current, last, width, width, height,1,1,16,16));



    //last[(2*TILE/2) * width + (2*TILE/2)] = 255; //LU
    last[2*TILE/2 + (2*TILE/2 + 2*TILE) * width] = 255; //L
    last[(2*TILE/2) * width + (2*TILE/2 + 2*TILE)] = 255; //U
    //last[(2*2*TILE + 2*TILE/2) * width + (2*TILE/2)] = 255; //RU
    //last[(2*TILE/2) * width + (2*TILE/2 + 2* 2*TILE)] = 255; //lD

    last[(2*TILE + 2*TILE/2) * width + (2*TILE/2 + 2*2*TILE)] = 255; //R
    //last[(TILE + TILE/2) * width + (TILE/2 + 2*TILE+1)] = 255;

    //last[(2*TILE/2+2*2*TILE) * width + (2*TILE/2 + 2*TILE)] = 255; //D
    //last[(TILE/2+2*TILE) * width + (TILE/2 + TILE+1)] = 255; //D

    unsigned sad =  fullSearchMB2(current, last, width, width, height, 1,1,16,0,0,&mvX,&mvY);

	//printf("%i %i\n", mvX, mvY);
	assert(sad == 0);
    assert(mvX == 0 && mvY == 16);



	free(current);
	free(last);
}

void testTotalSearch3(){
	unsigned width = TILE*2*4;
	unsigned height = TILE*2*4;

	u_int8_t* current = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
	u_int8_t* last = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);

	GetBlockCB2(current, last, width, height);

    int mvX =0, mvY=0;

    assert(0==SAD2(current, last, width, width, height,1,1,-16,0));
    assert(0==SAD2(current, last, width, width, height,1,1,0,-16));
    assert(TILE*TILE*99==SAD2(current, last, width, width, height,1,1,0,0));
	//printf("%i\n", SAD2(current, last, width, width, height,1,1,-16,-16));
    //assert(0==SAD2(current, last, width, width, height,1,1,16,16));



    //last[(2*TILE/2) * width + (2*TILE/2)] = 255; //LU
    last[2*TILE/2 + (2*TILE/2 + 2*TILE) * width] = 255; //L
    last[(2*TILE/2) * width + (2*TILE/2 + 2*TILE)] = 255; //U
    //last[(2*2*TILE + 2*TILE/2) * width + (2*TILE/2)] = 255; //RU
    //last[(2*TILE/2) * width + (2*TILE/2 + 2* 2*TILE)] = 255; //lD

    //last[(2*TILE + 2*TILE/2) * width + (2*TILE/2 + 2*2*TILE)] = 255; //R
    //last[(TILE + TILE/2) * width + (TILE/2 + 2*TILE+1)] = 255;

    last[(2*TILE/2+2*2*TILE) * width + (2*TILE/2 + 2*TILE)] = 255; //D
    //last[(TILE/2+2*TILE) * width + (TILE/2 + TILE+1)] = 255; //D

    unsigned sad =  fullSearchMB2(current, last, width, width, height, 1,1,16,0,0,&mvX,&mvY);

	//printf("%i %i\n", mvX, mvY);
	assert(sad == 0);
    assert(mvX == 16 && mvY == 0);

    int* minXs[2];
    int* minYs[2];
    int* sads[2];


    minXs[1] = (int*)malloc(8*8*sizeof(int));
    minYs[1]= (int*)malloc(8*8*sizeof(int));
    sads[1]= (int*)malloc(8*8*sizeof(int));

    minXs[0] = (int*)malloc(4*4*sizeof(int));
    minYs[0]= (int*)malloc(4*4*sizeof(int));
    sads[0]= (int*)malloc(4*4*sizeof(int));


    pyramidSearchHelper2(current, last, 1,1,width, width,height, 16, sads,minXs, minYs,0,0);
    assert(sads[0][5]==0);
	//printf("%i %i\n",minXs[1][18],minYs[1][18]);
    assert((minXs[1][18] == 2*TILE) );
    assert((sads[1][18] == TILE*TILE) );

	//printf("%i %i %i %i\n",last[6*TILE+1 + width * (2*TILE+1)], minXs[1][19], sads[1][19], SAD(current, last, width, width, height,3,2,33,0));

    assert((minXs[1][19] == 2*TILE+1) );
    assert((sads[1][19] == TILE*TILE-TILE) );

		//pyramidSearch2h(current, last,0,0,width, height, 16, sad,minX,minY,0,0);
		//pyramidSearch2(current, last, width, width, height,  16,sad,minX,minY);

    free(minXs[0]);
    free(minYs[0]);
    free(sads[0]);
    free(minXs[1]);
    free(minYs[1]);
    free(sads[1]);

	free(current);
	free(last);
}

void testTotalSearch4(){

	unsigned width = TILE*2*4;
	unsigned height = TILE*2*4;

	u_int8_t* current = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);
	u_int8_t* last = (u_int8_t*)malloc(sizeof(u_int8_t) * width * height);

	//For some reason I can't get memcpy to work:
	//char* z= (char*) malloc(sizeof(char));
	//*z = 0;
	//memcpy(current,z, width * height*sizeof(u_int8_t));
	//memcpy(last,z, width * height*sizeof(u_int8_t));

	int x;
    for(x =0;x< width;x++){
		int y;
		for(y = 0;y<height;y++){
			current[x + y * width]=0;
			last[x + y * width]=0;
		}
	}

    for(x = TILE *2;x< TILE*4;x++){
		unsigned y;
		for(y = TILE *2;y< TILE*4;y++){
			current[x+y*width] = 2;
			last[(x+2*TILE)+y*width] = 1 + 2*((x % 2 + y % 2) %2);
			last[(x+4*TILE)+y*width]=2;
		}
	}

    int mvX =0, mvY=0;

    unsigned sad =  fullSearchMB2(current, last, width, width, height, 1,1,16,0,0,&mvX,&mvY);
    assert(sad ==0);
    assert(mvX ==16);
    assert(mvY ==0);

    int* minXs[2];
    int* minYs[2];
    int* sads[2];

    minXs[1] = (int*)malloc(8*8*sizeof(int));
    minYs[1]= (int*)malloc(8*8*sizeof(int));
    sads[1]= (int*)malloc(8*8*sizeof(int));

    minXs[0] = (int*)malloc(4*4*sizeof(int));
    minYs[0]= (int*)malloc(4*4*sizeof(int));
    sads[0]= (int*)malloc(4*4*sizeof(int));

   // printf("dsa\n");
    pyramidSearchHelper2(current, last,1,1,width, width, height, 16, sads,minXs,minYs,0,0);
    assert(sads[0][1 + 1 * 4]==0);
    assert(sads[0][6]==0);
    //printf("%i\n", minXs[0][5]);
    //assert(minXs[0][6]==16);

    pyramidSearch2(current, last, width, width, height,  16,sads,minXs,minYs);
    assert(TILE * TILE == sads[1][2 * 8 + 2]);
    //assert(minYs[1][2 * 8 + 2] ==0||minYs[1][3 * 8 + 2] ==16);
    assert(minXs[1][2 * 8 + 2]==32);
    assert(minXs[0][1 * 4 + 1]==16);
    assert(sads[0][1 + 1 * 4]==0);
    assert(sads[0][6]==0);


    assert(minYs[1][3 * 8 + 2] ==-16 || minYs[1][3 * 8 + 2] ==0);
    assert(minXs[1][3 * 8 + 2]==32);

    //printf("%i %i %i\n",sads[1][3 * 8 + 3], minYs[1][3 * 8 + 3], minXs[1][3 * 8 + 3]);
    assert(0 == sads[1][2 * 8 + 4]);
    assert(0 == sads[1][3 * 8 + 4]);
    assert(TILE * TILE == sads[1][2 * 8 + 2]);
    assert(TILE * TILE == sads[1][3 * 8 + 2]);
    assert(0== sads[1][2 * 8 + 3]);
    assert(0 == sads[1][3 * 8 + 3]);

    free( minXs[0]);
    free( minYs[0]);
    free( sads[0]);

    free( minXs[1]);
    free( minYs[1]);
    free( sads[1]);

	free(current);
	free(last);
}
