#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N], fx_ref[N], fy_ref[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = fx_ref[i] = fy_ref[i] = 0;
  }

  __m512 xvec = _mm512_load_ps(x);
  __m512 yvec = _mm512_load_ps(y);
  __m512 mvec = _mm512_load_ps(m);

  for(int i=0; i<N; i++) {
    __m512 rxvec = _mm512_sub_ps(_mm512_set1_ps(x[i]), xvec);
    __m512 ryvec = _mm512_sub_ps(_mm512_set1_ps(y[i]), yvec);
    __m512 rvec = _mm512_rsqrt14_ps(_mm512_fmadd_ps(rxvec, rxvec, _mm512_mul_ps(ryvec, ryvec)));
    __m512 dfxvec = _mm512_mul_ps(rxvec, _mm512_mul_ps(mvec, _mm512_mul_ps(rvec, _mm512_mul_ps(rvec, rvec))));
    __m512 dfyvec = _mm512_mul_ps(ryvec, _mm512_mul_ps(mvec, _mm512_mul_ps(rvec, _mm512_mul_ps(rvec, rvec))));
    __m512 zvec = _mm512_setzero_ps();
    __mmask16 mask = _mm512_cmpneq_epi32_mask(_mm512_set1_epi32(i), _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0));
    dfxvec = _mm512_mask_blend_ps(mask, zvec, dfxvec);
    dfyvec = _mm512_mask_blend_ps(mask, zvec, dfyvec);
    fx[i] = -_mm512_reduce_add_ps(dfxvec);
    fy[i] = -_mm512_reduce_add_ps(dfyvec);

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }

  // for(int i=0; i<N; i++) {
  //   for(int j=0; j<N; j++) {
  //     if(i != j) {
  //       float rx = x[i] - x[j];
  //       float ry = y[i] - y[j];
  //       float r = std::sqrt(rx * rx + ry * ry);
  //       fx_ref[i] -= rx * m[j] / (r * r * r);
  //       fy_ref[i] -= ry * m[j] / (r * r * r);
  //     }
  //   }
  // }

  // for(int i=0; i<N; i++) {
  //   printf("%d %g %g %g %g\n", i, fx[i], fx_ref[i], fy[i], fy_ref[i]);
  // }
}
