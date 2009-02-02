#include <Python.h>
#include <arrayobject.h>

#define max(a, b)  (((a) > (b)) ? (a) : (b))
#define min(a, b)  (((a) < (b)) ? (a) : (b))

void init_me_gold();
static PyObject *me_gold(PyObject *self, PyObject *args);
void me(const uint8_t *frame1, const uint8_t *frame2, const size_t width, const size_t height, const int s, int* mvx, int* mvy, int* res);

typedef unsigned char uint8_t; static PyMethodDef _me_goldMethods[] = { {"me_gold", me_gold, METH_VARARGS},
	{NULL, NULL}
};

void init_me_gold() {
	(void) Py_InitModule("_me_gold", _me_goldMethods);
	import_array();
}

static PyObject *me_gold(PyObject *self, PyObject *args) {
	PyArrayObject *current_frame;
	PyArrayObject *reference_frame;
	if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &current_frame, &PyArray_Type, &reference_frame)) {
		return NULL;
	}
	if (current_frame == NULL || reference_frame == NULL) {
		return NULL;
	}
	if (current_frame->nd != 2 || reference_frame->nd != 2) {
		PyErr_SetString(PyExc_ValueError, "Frames have too many dimensions");
		return NULL;
	}
	if (current_frame->descr->type_num != PyArray_UINT8 || reference_frame->descr->type_num != PyArray_UINT8) {
		PyErr_SetString(PyExc_ValueError, "Inputs are of the wrong type");
		return NULL;
	}

	int dims[2], mbdims[2], width, height, numElements, mbNumElements;
	height = dims[1] = current_frame->dimensions[1];
	width = dims[0] = current_frame->dimensions[0];
	mbdims[1] = height >> 4;
	mbdims[0] = width >> 4;
	numElements = width * height;
	mbNumElements = mbdims[0] * mbdims[1];

	PyArrayObject *residual = (PyArrayObject *) PyArray_FromDims(2, dims, NPY_INT32); //TODO: npy_int16?
	PyArrayObject *mvx = (PyArrayObject *) PyArray_FromDims(2, mbdims, NPY_INT32); //TODO: npy_int16?
	PyArrayObject *mvy = (PyArrayObject *) PyArray_FromDims(2, mbdims, NPY_INT32); //TODO: npy_int16?

	uint8_t *current_frame_data = (uint8_t *)malloc(sizeof(uint8_t) * numElements);
	uint8_t *reference_frame_data = (uint8_t *)malloc(sizeof(uint8_t) * numElements);

	int *residual_data = (int *)malloc(sizeof(int) * numElements);
	int *mvx_data = (int *)malloc(sizeof(int) * mbNumElements);
	int *mvy_data = (int *)malloc(sizeof(int) * mbNumElements);

	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			current_frame_data[j * width + i] = *(uint8_t *)(current_frame->data + i * current_frame->strides[0] + j * current_frame->strides[1]);
			reference_frame_data[j * width + i] = *(uint8_t *)(reference_frame->data + i * reference_frame->strides[0] + j * reference_frame->strides[1]);
		}	
	}

	me(current_frame_data, reference_frame_data, width, height, 16, mvx_data, mvy_data, residual_data);

	for (int j = 0; j < height; j++) {
		for (int i = 0; i < width; i++) {
			*(int *)(residual->data + i * residual->strides[0] + j * residual->strides[1]) = residual_data[j * width + i];
		}	
	}
	for (int j = 0; j < height >> 4; j++) {
		for (int i = 0; i < width >> 4; i++) {
			*(int *)(mvx->data + i * mvx->strides[0] + j * mvx->strides[1]) = mvx_data[j * (width>>4) + i];
			*(int *)(mvy->data + i * mvy->strides[0] + j * mvy->strides[1]) = mvy_data[j * (width>>4) + i];
		}	
	}

	PyObject *output = PyTuple_New(3);
	PyTuple_SET_ITEM(output, 0, (PyObject *)mvx);
	PyTuple_SET_ITEM(output, 1, (PyObject *)mvy);
	PyTuple_SET_ITEM(output, 2, (PyObject *)residual);

	free(current_frame_data);
	free(reference_frame_data);
	free(mvx_data);
	free(mvy_data);
	free(residual_data);

	return output;
}


/* #define SEARCH_WINDOW_SIZE 16 */
void me(const uint8_t *frame1, const uint8_t *frame2, const size_t width, const size_t height, const int s, int* mvx, int* mvy, int* res) {
	unsigned int currSad, bestSad;
	int bestDx, bestDy;

	// Iterate through macroblocks
	for (int mbx = 0; mbx < (width+15) >> 4; mbx++) {
		for (int mby = 0; mby < (height+15) >> 4; mby++) {
			bestSad = UINT_MAX;
			bestDx = 0;
			bestDy = 0;

			//Iterate through search positions
			for (int dx = -s; dx < s; dx++) {
				for (int dy = -s; dy < s; dy++) {

					// Iterate through pixels and do SAD
					currSad = 0;
					for (int x = 0; x < 16; x++) {
						for (int y = 0; y < 16; y++) {
							currSad += (unsigned int)abs((int)frame2[(mbx * 16 + x) + (mby * 16 + y) * width] - (int)frame1[min(max(0, mbx * 16 + x + dx), width) + min(max(0, mby * 16 + y + dy), height) * width]);
						}
					}

					// See if this is a better position
					if (currSad < bestSad) {
						bestSad = currSad;
						bestDx = dx;
						bestDy = dy;
					}
				}
			}

			/* mexPrintf("%d\n", bestDx); */

			mvx[mby * ((width+15) >> 4) + mbx] = bestDx;
			mvy[mby * ((width+15) >> 4) + mbx] = bestDy;
			/* sad[mby * ((width+15) >> 4) + mbx] = bestSad; */

			int curx, cury;
			for (int x = 0; x < 16; x++) {
				for (int y = 0; y < 16; y++) {
					curx = mbx * 16 + x;
					cury = mby * 16 + y;
					res[curx + cury * width] = frame2[curx + cury * width] - frame1[min(max(0, curx + bestDx), width) + min(max(0, cury + bestDy), height) * width];
				}
			}
		}
	}
}
