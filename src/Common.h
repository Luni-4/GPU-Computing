#pragma once

#ifdef _WIN32
#include "Windows.h"
#endif

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <ctime>
#include <chrono>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t err = call;                                              \
    if (err != cudaSuccess)                                                    \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", err,                         \
                cudaGetErrorString(err));                                      \
    }                                                                          \
}

#define CHECK_CUBLAS(call)                                                     \
{                                                                              \
    cublasStatus_t err;                                                        \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CURAND(call)                                                     \
{                                                                              \
    curandStatus_t err;                                                        \
    if ((err = (call)) != CURAND_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUFFT(call)                                                      \
{                                                                              \
    cufftResult err;                                                           \
    if ( (err = (call)) != CUFFT_SUCCESS)                                      \
    {                                                                          \
        fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__,        \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUSPARSE(call)                                                   \
{                                                                              \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
    {                                                                          \
        fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
        cudaError_t cuda_err = cudaGetLastError();                             \
        if (cuda_err != cudaSuccess)                                           \
        {                                                                      \
            fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                    cudaGetErrorString(cuda_err));                             \
        }                                                                      \
        exit(1);                                                               \
    }                                                                          \
}

template<typename T>
inline void printVector(std::vector<T> &a, const int &dim) {

	for (std::size_t i = 0; i < a.size(); i++) {
		std::cout << a[i] << " ";
		if ((i + 1) % dim == 0)
			std::cout << " :" << i + 1 << std::endl;
		if ((i + 1) % (dim * dim) == 0)
			std::cout << std::endl;
	}
}

inline void printInputData(const double *a, const int &dim) {

	for (int i = 0; i < dim; i++)
		std::cout << a[i] << std::endl;
}

inline void printInputLabels(const uint8_t *a, const int &dim) {

	for (int i = 0; i < dim; i++)
		std::cout << unsigned(a[i]) << std::endl;
}

inline void printOnFile(std::vector<double> &a, const int &dim, std::ofstream &ofs) {

	for (std::size_t i = 0; i < a.size(); i++) {
		ofs << a[i] << " ";
		if ((i + 1) % dim == 0)
			ofs << " :" << i + 1 << std::endl;
		if ((i + 1) % (dim * dim) == 0)
			ofs << std::endl;
	}
}

inline void printLabels(std::vector<uint8_t> &a) {

	for (auto t : a)
		std::cout << unsigned(t) << std::endl;
}


inline void getTime(void(*func)(void), const std::string &fname) {
	auto start = std::chrono::high_resolution_clock::now();

	func();

	auto finish = std::chrono::high_resolution_clock::now();
	std::cout << fname << " eseguita in " << (finish - start).count() << std::endl;
}


inline void printFromCuda(const double *deb, const int &dim) {

	// DEBUG
	std::vector<double> outputC(dim);
	CHECK(cudaMemcpy(&outputC[0], deb, dim * sizeof(double), cudaMemcpyDeviceToHost));

	for (auto t : outputC)
		std::cout << t << std::endl;
}


inline void printFromCudaFormatted(const double *deb, const int wdim, const int &dim) {

	// DEBUG
	std::vector<double> outputC(wdim);
	CHECK(cudaMemcpy(&outputC[0], deb, wdim * sizeof(double), cudaMemcpyDeviceToHost));

	for (std::size_t i = 0; i < outputC.size(); i++) {
		double n = 1000000000;
		int cut = (outputC[i] * n);
		double o = (static_cast<double>(cut)) / n;
		std::cout << o << " ";
		//std::cout << outputC[i] << " ";
		if ((i + 1) % dim == 0)
			std::cout << " :" << i << std::endl;
		if ((i + 1) % (dim * dim) == 0)
			std::cout << std::endl;
	}
}

inline void pettyPrintCuda(const double *deb, const int wdim, const int &dim) {

	// DEBUG
	std::vector<double> outputC(wdim);
	CHECK(cudaMemcpy(&outputC[0], deb, wdim * sizeof(double), cudaMemcpyDeviceToHost));

	for (std::size_t i = 0; i < outputC.size(); i++) {
		std::cout << outputC[i] << std::endl; //<< " ";
		/*if ((i + 1) % dim == 0)
			std::cout << " :" << i + 1 << std::endl;
		if ((i + 1) % (dim * dim) == 0)
			std::cout << std::endl;*/
	}
}
