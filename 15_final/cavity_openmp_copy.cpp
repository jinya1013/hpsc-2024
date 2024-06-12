#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <chrono>
using namespace std;

void set_linspace(double* result, double start, double end, int num) {
    if (num <= 0) {
        return;
    }
    if (num == 1) {
        result[0] = start;
        return;
    }

    double step = (end - start) / (num - 1);
    for (int i = 0; i < num; ++i) {
        result[i] = start + step * i;
    }
}

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
    int nx = 41, ny = 41, nt = 500, nit = 50;
    double dx = 2.0 / (nx - 1), dy = 2.0 / (ny - 1), dt = 0.01;
    double rho = 1.0, nu = 0.02;

    double x[nx], y[ny];

    set_linspace(x, 0, 2, nx);
    set_linspace(y, 0, 2, ny);

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

    set_zeros2d(un, ny, nx);
    set_zeros2d(vn, ny, nx);
    set_zeros2d(pn, ny, nx);

    auto tic = chrono::steady_clock::now();

    for (int n = 0; n < nt; ++n) {
        if (n % 2 == 0) {
            #pragma omp parallel for
            for (int idx = 0; idx < (ny - 2) * (nx - 2); ++idx) {
                int j = 1 + idx / (nx - 2);
                int i = 1 + idx % (nx - 2);
                b[j][i] = rho * (1 / dt *
                    ((u[j][i+1] - u[j][i-1]) / (2 * dx) + (v[j+1][i] - v[j-1][i]) / (2 * dy)) -
                    ((u[j][i+1] - u[j][i-1]) / (2 * dx)) * ((u[j][i+1] - u[j][i-1]) / (2 * dx)) - 2 * ((u[j+1][i] - u[j-1][i]) / (2 * dy) *
                        (v[j][i+1] - v[j][i-1]) / (2 * dx)) - ((v[j+1][i] - v[j-1][i]) / (2 * dy)) * ((v[j+1][i] - v[j-1][i]) / (2 * dy)));
            }
        } else {
            #pragma omp parallel for
            for (int idx = 0; idx < (ny - 2) * (nx - 2); ++idx) {
                int j = 1 + idx / (nx - 2);
                int i = 1 + idx % (nx - 2);
                b[j][i] = rho * (1 / dt *
                    ((un[j][i+1] - un[j][i-1]) / (2 * dx) + (vn[j+1][i] - vn[j-1][i]) / (2 * dy)) -
                    ((un[j][i+1] - un[j][i-1]) / (2 * dx)) * ((un[j][i+1] - un[j][i-1]) / (2 * dx)) - 2 * ((un[j+1][i] - un[j-1][i]) / (2 * dy) *
                        (vn[j][i+1] - vn[j][i-1]) / (2 * dx)) - ((vn[j+1][i] - vn[j-1][i]) / (2 * dy)) * ((vn[j+1][i] - vn[j-1][i]) / (2 * dy)));
            }
        }
        for (int it = 0; it < nit; ++it) {
            if (it % 2 != 0) {
                #pragma omp parallel for
                for (int idx = 0; idx < (ny - 2) * (nx - 2); ++idx) {
                    int j = 1 + idx / (nx - 2);
                    int i = 1 + idx % (nx - 2);
                    p[j][i] = (dy * dy * (pn[j][i+1] + pn[j][i-1]) + 
                        dx * dx * (pn[j+1][i] + pn[j-1][i]) -
                        b[j][i] * dx * dx * dy * dy) / (2 * (dx * dx + dy * dy));
                }
                for (int i = 0; i < ny; ++i) {
                    p[i][nx - 1] = p[i][nx - 2];
                }
                for (int j = 0; j < nx; ++j) {
                    p[0][j] = p[1][j]; 
                }
                for (int i = 0; i < ny; ++i) {
                    p[i][0] = p[i][1]; 
                }
                for (int j = 0; j < nx; ++j) {
                    p[ny - 1][j] = 0; 
                }
            } else {
                #pragma omp parallel for
                for (int idx = 0; idx < (ny - 2) * (nx - 2); ++idx) {
                    int j = 1 + idx / (nx - 2);
                    int i = 1 + idx % (nx - 2);
                    pn[j][i] = (dy * dy * (p[j][i+1] + p[j][i-1]) + 
                        dx * dx * (p[j+1][i] + p[j-1][i]) -
                        b[j][i] * dx * dx * dy * dy) / (2 * (dx * dx + dy * dy));
                }
                for (int i = 0; i < ny; ++i) {
                    pn[i][nx - 1] = pn[i][nx - 2];
                }
                for (int j = 0; j < nx; ++j) {
                    pn[0][j] = pn[1][j]; 
                }
                for (int i = 0; i < ny; ++i) {
                    pn[i][0] = pn[i][1]; 
                }
                for (int j = 0; j < nx; ++j) {
                    pn[ny - 1][j] = 0; 
                }
            }
        }
        if (n % 2 != 0) {
            #pragma omp parallel for
            for (int idx = 0; idx < (ny - 2) * (nx - 2); ++idx) {
                int j = 1 + idx / (nx - 2);
                int i = 1 + idx % (nx - 2);
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
            for (int j = 0; j < nx; ++j) {
                u[0][j] = 0;
                u[ny - 1][j] = 1;
            }
            for (int i = 0; i < ny; ++i) {
                u[i][0] = 0;
                u[i][nx - 1] = 0;
            }
            for (int j = 0; j < nx; ++j) {
                v[0][j] = 0;
                v[ny - 1][j] = 0;
            }
            for (int i = 0; i < ny; ++i) {
                v[i][0] = 0;
                v[i][nx - 1] = 0;
            }
        } else {
            #pragma omp parallel for
            for (int idx = 0; idx < (ny - 2) * (nx - 2); ++idx) {
                int j = 1 + idx / (nx - 2);
                int i = 1 + idx % (nx - 2);
                un[j][i] = u[j][i] - u[j][i] * dt / dx * (u[j][i] - u[j][i-1])
                        - u[j][i] * dt / dy * (u[j][i] - u[j-1][i])
                        - dt / (2 * rho * dx) * (p[j][i+1] - p[j][i-1])
                        + nu * dt / (dx*dx) * (u[j][i+1] - 2 * u[j][i] + u[j][i-1])
                        + nu * dt / (dy*dy) * (u[j+1][i] - 2 * u[j][i] + u[j-1][i]);
                vn[j][i] = v[j][i] - v[j][i] * dt / dx * (v[j][i] - v[j][i-1])
                                - v[j][i] * dt / dy * (v[j][i] - v[j-1][i])
                                - dt / (2 * rho * dx) * (p[j+1][i] - p[j-1][i])
                                + nu * dt / (dx*dx) * (v[j][i+1] - 2 * v[j][i] + v[j][i-1])
                                + nu * dt / (dy*dy) * (v[j+1][i] - 2 * v[j][i] + v[j-1][i]);
            }
            for (int j = 0; j < nx; ++j) {
                un[0][j] = 0;
                un[ny - 1][j] = 1;
            }
            for (int i = 0; i < ny; ++i) {
                un[i][0] = 0;
                un[i][nx - 1] = 0;
            }
            for (int j = 0; j < nx; ++j) {
                vn[0][j] = 0;
                vn[ny - 1][j] = 0;
            }
            for (int i = 0; i < ny; ++i) {
                vn[i][0] = 0;
                vn[i][nx - 1] = 0;
            }
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