#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <chrono>
using namespace std;

void print_1d_array(double* array, int size) {
    for (int i = 0; i < size; ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    int nx = 41, ny = 41, nt = 500, nit = 50;
    double dx = 2.0 / (nx - 1), dy = 2.0 / (ny - 1), dt = 0.01;
    double rho = 1.0, nu = 0.02;

    double u[ny*nx], v[ny*nx], p[ny*nx], b[ny*nx];
    double un[ny*nx], vn[ny*nx], pn[ny*nx];

    for (int idx = 0; idx < ny * nx; ++idx) {
        u[idx] = v[idx] = p[idx] = b[idx] = un[idx] = vn[idx] = pn[idx] = 0;
    }

    auto tic = chrono::steady_clock::now();

    for (int n = 0; n < nt; ++n) {
        #pragma omp parallel for collapse(2)
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                int idx = j * nx + i;
                int idx_ip1 = j * nx + (i + 1);
                int idx_im1 = j * nx + (i - 1);
                int idx_jp1 = (j + 1) * nx + i;
                int idx_jm1 = (j - 1) * nx + i;

                b[idx] = rho * (1 / dt *
                    ((u[idx_ip1] - u[idx_im1]) / (2 * dx) + (v[idx_jp1] - v[idx_jm1]) / (2 * dy)) -
                    ((u[idx_ip1] - u[idx_im1]) / (2 * dx)) * ((u[idx_ip1] - u[idx_im1]) / (2 * dx)) - 
                    2 * ((u[idx_jp1] - u[idx_jm1]) / (2 * dy) * (v[idx_ip1] - v[idx_im1]) / (2 * dx)) - 
                    ((v[idx_jp1] - v[idx_jm1]) / (2 * dy)) * ((v[idx_jp1] - v[idx_jm1]) / (2 * dy)));
            }
        }
        for (int it = 0; it < nit; ++it) {
            std::copy(p, p+ny*nx, pn);
            #pragma omp parallel for collapse(2)
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    int idx = j * nx + i;
                    int idx_ip1 = j * nx + (i + 1);
                    int idx_im1 = j * nx + (i - 1);
                    int idx_jp1 = (j + 1) * nx + i;
                    int idx_jm1 = (j - 1) * nx + i;
                    p[idx] = (dy * dy * (pn[idx_ip1] + pn[idx_im1]) + 
                        dx * dx * (pn[idx_jp1] + pn[idx_jm1]) -
                        b[idx] * dx * dx * dy * dy) / (2 * (dx * dx + dy * dy));
                }
            }
            for (int j = 0; j < ny; ++j) {
                p[j * ny + nx- 1] = p[j * ny + nx - 2];
            }
            for (int i = 0; i < nx; ++i) {
                p[i] = p[ny + i]; 
            }
            for (int j = 0; j < ny; ++j) {
                p[j * ny] = p[j * ny + 1]; 
            }
            for (int i = 0; i < nx; ++i) {
                p[(ny - 1) * ny + i] = 0; 
            }
        }
        std::copy(u, u+ny*nx, un);
        std::copy(v, v+ny*nx, vn);
        #pragma omp parallel for collapse(2)
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                int idx = j * nx + i;
                int idx_ip1 = j * nx + (i + 1);
                int idx_im1 = j * nx + (i - 1);
                int idx_jp1 = (j + 1) * nx + i;
                int idx_jm1 = (j - 1) * nx + i;
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
        }
        for (int i = 0; i < nx; ++i) {
            u[i] = 0;
            u[(ny - 1) * ny + i] = 1;
        }
        for (int j = 0; j < ny; ++j) {
            u[j * ny] = 0;
            u[j * ny + nx - 1] = 0;
        }
        for (int i = 0; i < nx; ++i) {
            v[i] = 0;
            v[(ny - 1) * ny + i] = 0;
        }
        for (int j = 0; j < ny; ++j) {
            v[j * ny] = 0;
            v[j * ny + nx - 1] = 0;
        }
    }

    auto toc = chrono::steady_clock::now();
    double time = chrono::duration<double>(toc - tic).count();

    std::cout << "u" << std::endl;
    print_1d_array(u, 2*ny);
    std::cout << "v" << std::endl;
    print_1d_array(v, 2*ny);

    std::cout << "time: " << time << "s" << std::endl;

    return 0;
}