/*
 * tests.h
 *
 *  Created on: Jan 16, 2009
 *      Author: alex
 */

#ifndef TESTS_H_
#define TESTS_H_

#include "gold.h"
#include <stdarg.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include "common/common.h"

void mainTests(x264_t *h );
void testSearch();
void testSAD(x264_t *h);
void testPyramidSearch();

void test1();
void test2();
void test3();
void test3a(x264_t *h);
void test4();
void test5();
void test6();
void test7();
void test8();
void test9();
void test10();
void test10b();
void test11();
void test12();
void test13();
void test14();
void test15();
void test16();

void testSearch1();
void testSearch1a();
void testSearch2();
void testSearch3();
void testSearch4();
void testSearch5();
void testPad1();
void testPad2();
void testPad3();
void testPad4();

void testTotalSearch1();
void testTotalSearch2();
void testTotalSearch3();
void testTotalSearch3a();
void testTotalSearch4();
void testTotalSearch4cuda();

#endif /* TESTS_H_ */
