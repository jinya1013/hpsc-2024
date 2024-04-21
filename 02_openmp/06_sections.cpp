#include <cstdio>
#include <omp.h>

int main() {
  int i = 1;
#pragma omp parallel num_threads(2)
#pragma omp sections firstprivate(i)
  {
#pragma omp section
    printf("%d %d\n",++i, omp_get_thread_num());
#pragma omp section
    printf("%d %d\n",++i, omp_get_thread_num());
  }
}
