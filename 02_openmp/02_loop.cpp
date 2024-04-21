#include <cstdio>
#include <omp.h>

int main() {
  double s = 0;
  double inp[10] = {1,2,3,4,5,6,7,8,9,10};
#pragma omp parallel for reduction(+:s)
  for (int i = 0; i < 10; i++) {
    s += inp[i];
  }
  printf("sum: %f\n",s);
}
