#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <mpi.h>
using namespace std;

void print_1d_array(double* array, int size) {
    for (int i = 0; i < size; ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    int nx = 40, ny = 40, nt = 500, nit = 50;
    double dx = 2.0 / (nx - 1), dy = 2.0 / (ny - 1), dt = 0.01;
    double rho = 1.0, nu = 0.02;

    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int begin = rank * (ny * nx / size);
    int end = (rank + 1) * (ny * nx / size);

    double u[ny*nx], v[ny*nx], p[ny*nx], b[ny*nx];
    double un[ny*nx], vn[ny*nx], pn[ny*nx];

    std::cout << size << " " << rank << " " << begin << " " << end << " " << ny * nx << "\n";

    for (int idx = 0; idx < ny * nx; ++idx) {
        u[idx] = v[idx] = p[idx] = b[idx] = un[idx] = vn[idx] = pn[idx] = 0;
    }

    auto tic = chrono::steady_clock::now();

    for (int n = 0; n < nt; ++n) {
        for (int idx = begin; idx < end; ++idx) {
            if (idx < ny || idx % ny == 0 || idx % ny == ny - 1 || idx >= (ny - 1) * nx) {
                continue;
            }
            int idx_ip1 = idx + 1;
            int idx_im1 = idx - 1;
            int idx_jp1 = idx + ny;
            int idx_jm1 = idx - ny;
            b[idx] = rho * (1 / dt *
                ((u[idx_ip1] - u[idx_im1]) / (2 * dx) + (v[idx_jp1] - v[idx_jm1]) / (2 * dy)) -
                ((u[idx_ip1] - u[idx_im1]) / (2 * dx)) * ((u[idx_ip1] - u[idx_im1]) / (2 * dx)) -
                2 * ((u[idx_jp1] - u[idx_jm1]) / (2 * dy) * (v[idx_ip1] - v[idx_im1]) / (2 * dx)) -
                ((v[idx_jp1] - v[idx_jm1]) / (2 * dy)) * ((v[idx_jp1] - v[idx_jm1]) / (2 * dy)));
        }
        for (int it = 0; it < nit; ++it) {
            std::copy(p+begin, p+end, pn+begin);
            if (rank > 0) {
                MPI_Send(&pn[begin], ny, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
                MPI_Recv(&pn[begin-ny], ny, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            if (rank < size - 1) {
                MPI_Send(&pn[end-ny], ny, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
                MPI_Recv(&pn[end], ny, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            for (int idx = begin; idx < end; ++idx) {
                if (idx < ny || idx % ny == 0 || idx % ny == ny - 1 || idx >= (ny - 1) * nx) {
                    continue;
                }
                int idx_ip1 = idx + 1;
                int idx_im1 = idx - 1;
                int idx_jp1 = idx + ny;
                int idx_jm1 = idx - ny;
                p[idx] = (dy * dy * (pn[idx_ip1] + pn[idx_im1]) + 
                    dx * dx * (pn[idx_jp1] + pn[idx_jm1]) -
                    b[idx] * dx * dx * dy * dy) / (2 * (dx * dx + dy * dy));
            }

            if (rank == 0) {
                for (int i = 0; i < nx; ++i) {
                    p[i] = p[ny + i]; 
                }
            } 
            if (rank == size - 1) {
                for (int i = 0; i < nx; ++i) {
                    p[(ny - 1) * ny + i] = 0; 
                }
            } 
            for (int idx = begin; idx < end; idx += ny) {
                p[idx] = p[idx + 1]; 
                p[idx + ny - 1] = p[idx + ny - 2];
            }
        }
        if (rank > 0) {
            MPI_Send(&p[begin], ny, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&p[begin-ny], ny, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Send(&p[end-ny], ny, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&p[end], ny, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        std::copy(u+begin, u+end, un+begin);
        std::copy(v+begin, v+end, vn+begin);
        if (rank > 0) {
            MPI_Send(&un[begin], ny, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&un[begin-ny], ny, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&vn[begin], ny, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&vn[begin-ny], ny, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Send(&un[end-ny], ny, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&un[end], ny, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&vn[end-ny], ny, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&vn[end], ny, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        for (int idx = begin; idx < end; ++idx) {
            if (idx < ny || idx % ny == 0 || idx % ny == ny - 1 || idx >= (ny - 1) * nx) {
                continue;
            }
            int idx_ip1 = idx + 1;
            int idx_im1 = idx - 1;
            int idx_jp1 = idx + ny;
            int idx_jm1 = idx - ny;
            u[idx] = un[idx] - un[idx] * dt / dx * (un[idx] - un[idx_im1])
                    - un[idx] * dt / dy * (un[idx] - un[idx_jm1])
                    - dt / (2 * rho * dx) * (p[idx_ip1] - p[idx_im1])
                    + nu * dt / (dx*dx) * (un[idx_ip1] - 2 * un[idx] + un[idx_im1])
                    + nu * dt / (dy*dy) * (un[idx_jp1] - 2 * un[idx] + un[idx_jm1]);
            v[idx] = vn[idx] - vn[idx] * dt / dx * (vn[idx] - vn[idx_im1])
                            - vn[idx] * dt / dy * (vn[idx] - vn[idx_jm1])
                            - dt / (2 * rho * dx) * (p[idx_jp1] - p[idx_jm1])
                            + nu * dt / (dx*dx) * (vn[idx_ip1] - 2 * vn[idx] + vn[idx_im1])
                            + nu * dt / (dy*dy) * (vn[idx_jp1] - 2 * vn[idx] + vn[idx_jm1]);
        }
        if (rank == 0) {
            for (int i = 0; i < nx; ++i) {
                u[i] = v[i] = 0;
                
            }
        } 
        if (rank == size - 1) {
            for (int i = 0; i < nx; ++i) {
                u[(ny - 1) * ny + i] = 1;
                v[(ny - 1) * ny + i] = 0;
            }
        } 
        for (int idx = begin; idx < end; idx += ny) {
            u[idx] = v[idx] = 0;
            u[idx + ny - 1] = v[idx + ny - 1] = 0;
        }
        if (rank > 0) {
            MPI_Send(&u[begin], ny, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&u[begin-ny], ny, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&v[begin], ny, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&v[begin-ny], ny, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            MPI_Send(&u[end-ny], ny, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&u[end], ny, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(&v[end-ny], ny, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&v[end], ny, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    MPI_Gather(&u[begin], end - begin, MPI_DOUBLE, un, end - begin, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&v[begin], end - begin, MPI_DOUBLE, vn, end - begin, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&p[begin], end - begin, MPI_DOUBLE, pn, end - begin, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    auto toc = chrono::steady_clock::now();
    double time = chrono::duration<double>(toc - tic).count();

    if (rank == 0) {
        std::cout << "u" << std::endl;
        print_1d_array(un, 2*ny);
        std::cout << "v" << std::endl;
        print_1d_array(vn, 2*ny);
        std::cout << "time: " << time << "s" << std::endl;
    }

    MPI_Finalize();

    return 0;
}