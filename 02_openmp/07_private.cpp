#include <cstdio>

int main() {
  int a = 1, b = 1, c = 1;
#pragma omp parallel num_threads(4)
  {
#pragma omp for private(a)
    for(int i=0; i<4; i++)
      printf("private: %d ",++a);
#pragma omp single
    printf("\n");
#pragma omp for firstprivate(b)
    for(int i=0; i<4; i++)
      printf("firstprivate: %d ",++b);
#pragma omp single
    printf("\n");
#pragma omp for lastprivate(c)
    for(int i=0; i<4; i++)
      printf("lastprivate: %d ",++c);
#pragma omp single
    printf("\n");
  }
  printf("%d %d %d\n",a,b,c);
}
