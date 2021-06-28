#pragma once

#include <iostream>

template <class X>
void println(X out)
{

	std::wcout << out << std::endl;

}

template <class X, class Y>
void println(X a1, Y a2)
{


	std::wcout << a1 << "::" << a2 << std::endl;

}

template <class X>
void printlnAll(X out, int size)
{

	for (int i = 0; i < size; i++)
	{
		
		std::wcout << i <<" :: " << out[i] << std::endl;
		//printf("%.15f\n", out[i]);
	}

}

template <class X>
void CHECK(X err)
{

	const cudaError_t error = err;
	if (error != cudaSuccess)
	{

		std::cout << "Error: " << __FILE__ << ":" << __LINE__ << std::endl;
		std::cout << "code: " << error << ", reason: " << cudaGetErrorString(error) << std::endl;

		exit(1);

	}

}