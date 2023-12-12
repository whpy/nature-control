#ifndef _CUDAERR_H
#define _CUDAERR_H

#include <iostream>
#include <exception>

#include <cufft.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

using namespace std;

#ifdef _CUFFT_H_
// cuFFT API errors, for some reason not defined in cufft.h
static const char *cufftGetErrorString(cufftResult error)
{
	switch (error)
	{
		case CUFFT_SUCCESS:
			return "CUFFT_SUCCESS";

		case CUFFT_INVALID_PLAN:
			return "CUFFT_INVALID_PLAN";

		case CUFFT_ALLOC_FAILED:
			return "CUFFT_ALLOC_FAILED";

		case CUFFT_INVALID_TYPE:
			return "CUFFT_INVALID_TYPE";

		case CUFFT_INVALID_VALUE:
			return "CUFFT_INVALID_VALUE";

		case CUFFT_INTERNAL_ERROR:
			return "CUFFT_INTERNAL_ERROR";

		case CUFFT_EXEC_FAILED:
			return "CUFFT_EXEC_FAILED";

		case CUFFT_SETUP_FAILED:
			return "CUFFT_SETUP_FAILED";

		case CUFFT_INVALID_SIZE:
			return "CUFFT_INVALID_SIZE";

		case CUFFT_UNALIGNED_DATA:
			return "CUFFT_UNALIGNED_DATA";
	}

	return "<unknown error>";
}
// output of error strings produced by cufft.h
inline void cufft_error_func(cufftResult result)
{
	if( result != CUFFT_SUCCESS )
	{
		std::cerr << "Error: " << cufftGetErrorString(result) << std::endl;
		std::terminate();
	}
}
#endif

// output of error strings produced by CUDA
inline void cuda_error_func(cudaError_t result)
{
	if( result != cudaSuccess )
	{
		std::cerr << "Error: " << cudaGetErrorString(result) << std::endl;
		std::terminate();
	}
}

#endif // end of cuda_error.h





