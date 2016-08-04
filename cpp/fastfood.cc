 /** @internal
 ** @file     FisherExtractor.cxx
 ** @brief    JNI Wrapper for enceval GMM and Fisher Vector
 **/

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include <ctime>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <stdlib.h>
#include <iostream>

#include "fht_header_only.h"

extern "C" {
void fastfood(
    float* gaussian,
    float* radamacher,
    float* chiSquared,
    float* patchMatrix,
    float* output,
    int outSize,
    int inSize,
    int numPatches);
}

using namespace Eigen;
using namespace std;

void fastfood(
    float* gaussian,
    float* radamacher,
    float* chiSquared,
    float* patchMatrix,
    float* output,
    int outSize,
    int inSize,
    int numPatches)
{
  float* out;

  int ret = posix_memalign((void**) &out, 32, outSize*numPatches*sizeof(float));
  if (ret != 0) {
    throw std::runtime_error("posix_memalign failed\n");
  }

  /* (outSize x numPatches) matrix */
  Map< Array<float, Dynamic, Dynamic, RowMajor> > outM(out, outSize, numPatches);
  Map< Array<float, Dynamic, Dynamic, RowMajor> > mf(patchMatrix, inSize, numPatches);
  Map< Array<float, Dynamic, 1> > radamacherVector(radamacher, outSize);
  Map< Array<float, Dynamic, 1> > gaussianVector(gaussian, outSize);
  Map< Array<float, Dynamic, 1> > chisquaredVector(chiSquared, outSize);
  for (int i = 0; i < outSize; i += inSize) {
    outM.block(i, 0, inSize, numPatches) = mf;
    outM.block(i, 0, inSize, numPatches).colwise() *=  radamacherVector.segment(i, inSize);
    for (int j = 0; j < numPatches; j += 1) {
      float* patch = out + (j*outSize) + i;
      FHTFloat(patch, inSize, 2048);
    }
    outM.block(i, 0, inSize, numPatches).colwise() *= gaussianVector.segment(i, inSize);
    for (int j = 0; j < numPatches; j += 1) {
      float* patch = out + (j*outSize) + i;
      FHTFloat(patch, inSize, 2048);
    }
    outM.block(i, 0, inSize, numPatches).colwise() *= chisquaredVector.segment(i, inSize);
  }
  memcpy(output, out,outSize*numPatches*sizeof(float));
  free(out);
}

