#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <chrono>
using namespace std;

void set_zeros2d(double** result, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = 0;
        }
    }
}

double** allocate_2d_array(int rows, int cols) {
    double** array = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; ++i) {
        array[i] = (double*)malloc(cols * sizeof(double));
    }
    return array;
}
void print_2d_array(double** array, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << array[i][j] << " ";
        }
        std::cout << std::endl;

        if (i == 1) {
            break;
        }
    }
}
void print_1d_array(double* array, int size) {
    for (int i = 0; i < size; ++i) {
        std::cout << array[i] << " ";
    }
    std::cout << std::endl;
}

void copy_2d_array(double** src, double** dest, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        std::copy(src[i], src[i] + cols, dest[i]);
    }
}

void free_2d_array(double** array, int rows) {
    for (int i = 0; i < rows; ++i) {
        free(array[i]);
    }
    free(array);
}

int main() {
    int nx = 40, ny = 40, nt = 500, nit = 50;
    double dx = 2.0 / (nx - 1), dy = 2.0 / (ny - 1), dt = 0.01;
    double rho = 1.0, nu = 0.02;

    double** u = allocate_2d_array(ny, nx);
    double** v = allocate_2d_array(ny, nx);
    double** p = allocate_2d_array(ny, nx);
    double** b = allocate_2d_array(ny, nx);

    double** un = allocate_2d_array(ny, nx);
    double** vn = allocate_2d_array(ny, nx);
    double** pn = allocate_2d_array(ny, nx);

    set_zeros2d(u, ny, nx);
    set_zeros2d(v, ny, nx);
    set_zeros2d(p, ny, nx);
    set_zeros2d(b, ny, nx);

    auto tic = chrono::steady_clock::now();

    for (int n = 0; n < nt; ++n) {
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                b[j][i] = rho * (1 / dt *
                    ((u[j][i+1] - u[j][i-1]) / (2 * dx) + (v[j+1][i] - v[j-1][i]) / (2 * dy)) -
                    ((u[j][i+1] - u[j][i-1]) / (2 * dx)) * ((u[j][i+1] - u[j][i-1]) / (2 * dx)) - 2 * ((u[j+1][i] - u[j-1][i]) / (2 * dy) *
                     (v[j][i+1] - v[j][i-1]) / (2 * dx)) - ((v[j+1][i] - v[j-1][i]) / (2 * dy)) * ((v[j+1][i] - v[j-1][i]) / (2 * dy)));
            }
        }
        for (int it = 0; it < nit; ++it) {
            copy_2d_array(p, pn, ny, nx);
            for (int j = 1; j < ny - 1; ++j) {
                for (int i = 1; i < nx - 1; ++i) {
                    p[j][i] = (dy * dy * (pn[j][i+1] + pn[j][i-1]) + 
                        dx * dx * (pn[j+1][i] + pn[j-1][i]) -
                        b[j][i] * dx * dx * dy * dy) / (2 * (dx * dx + dy * dy));
                }
            }
            for (int j = 0; j < ny; ++j) {
                p[j][0] = p[j][1]; 
                p[j][nx - 1] = p[j][nx - 2];
            }
            for (int i = 0; i < nx; ++i) {
                p[0][i] = p[1][i]; 
                p[ny - 1][i] = 0; 
            }
        }
        copy_2d_array(u, un, ny, nx);
        copy_2d_array(v, vn, ny, nx);
        for (int j = 1; j < ny - 1; ++j) {
            for (int i = 1; i < nx - 1; ++i) {
                u[j][i] = un[j][i] - un[j][i] * dt / dx * (un[j][i] - un[j][i-1])
                        - un[j][i] * dt / dy * (un[j][i] - un[j-1][i])
                        - dt / (2 * rho * dx) * (p[j][i+1] - p[j][i-1])
                        + nu * dt / (dx*dx) * (un[j][i+1] - 2 * un[j][i] + un[j][i-1])
                        + nu * dt / (dy*dy) * (un[j+1][i] - 2 * un[j][i] + un[j-1][i]);
                v[j][i] = vn[j][i] - vn[j][i] * dt / dx * (vn[j][i] - vn[j][i-1])
                                - vn[j][i] * dt / dy * (vn[j][i] - vn[j-1][i])
                                - dt / (2 * rho * dx) * (p[j+1][i] - p[j-1][i])
                                + nu * dt / (dx*dx) * (vn[j][i+1] - 2 * vn[j][i] + vn[j][i-1])
                                + nu * dt / (dy*dy) * (vn[j+1][i] - 2 * vn[j][i] + vn[j-1][i]);
            }
        }
        for (int i = 0; i < nx; ++i) {
            u[0][i] = v[0][i] = 0;
            u[ny - 1][i] = 1;
            v[ny - 1][i] = 0;
        }
        for (int j = 0; j < ny; ++j) {
            u[j][0] = v[j][0] = 0;
            u[j][nx - 1] = v[j][nx - 1] = 0;
        }
    }

    auto toc = chrono::steady_clock::now();
    double time = chrono::duration<double>(toc - tic).count();

    std::cout << "u" << std::endl;
    print_2d_array(u, ny, nx);
    std::cout << "v" << std::endl;
    print_2d_array(v, ny, nx);

    std::cout << "time: " << time << "s" << std::endl;


    free_2d_array(u, ny);
    free_2d_array(v, ny);
    free_2d_array(p, ny);
    free_2d_array(b, ny);
    free_2d_array(un, ny);
    free_2d_array(vn, ny);
    free_2d_array(pn, ny);

    return 0;
}