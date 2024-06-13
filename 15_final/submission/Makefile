01_cavity: 01_cavity.cpp
	g++ 01_cavity.cpp -O3
02_cavity_openmp: 02_cavity_openmp.cpp
	g++ 02_cavity_openmp.cpp -fopenmp -O3
03_cavity_mpi: 03_cavity_mpi.cpp
	mpicxx 03_cavity_mpi.cpp
04_cavity_simd: 04_cavity_simd.cpp
	g++ 04_cavity_simd.cpp -fopt-info-vec-optimized -march=native -O3
05_cavity_openacc: 05_cavity_openacc.cpp
	nvc++ 05_cavity_openacc.cpp -acc -Minfo
	
