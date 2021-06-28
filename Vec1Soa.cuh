#pragma once

#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#include <thrust/copy.h>



template <typename T>
class Vec1Soa
	:public thrust::device_vector<T>
{

public:
	typedef T value_type;
	typedef typename  thrust::device_vector<T>::iterator iterator;
	T* _rawPointer;

	Vec1Soa(void){}

	Vec1Soa(unsigned int n, const T& val = T())
	{

		resize(n, val);

	}

	/*Vec1Soa(const Vec1Soa<T>& lhs)
	{

		resize(lhs.size());
		copy(lhs.begin(), lhs.end(), begin());

	}*/

	void resize(unsigned int n, T val = 0)
	{

		thrust::device_vector<T>::resize(n, val);
		_rawPointer = thrust::raw_pointer_cast(&((*this)[0]));

	}

	void clear(void)
	{

		thrust::device_vector<T>::clear();
		_rawPointer = NULL;

	}

	void push_back(T v1)
	{

		thrust::device_vector<T>::push_back(v1);
		_rawPointer = thrust::raw_pointer_cast(&((*this)[0]));
	
	}

	

	//math
	/*void operator = (const Vec1Soa<T>& rhs)
	{

		thrust::copy(rhs.begin(), rhs.end(), begin());

	}*/
	
	// component-wise vector add assign
	friend Vec1Soa<T>& operator += (Vec1Soa<T>& lhs, T d){
		thrust::transform(lhs.begin(), lhs.end(), make_constant_iterator(d), lhs.begin(), thrust::plus<T>());
		return lhs;
	}

	friend Vec1Soa<T>& operator += (Vec1Soa<T>& lhs, const Vec1Soa<T>& rhs){
		thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(), thrust::plus<T>());
		return lhs;
	}

	friend Vec1Soa<T> operator + (const Vec1Soa<T>& lhs, T rhs)
	{

		Vec1Soa<T> ans(lhs);
		return ans += rhs;

	}

	friend Vec1Soa<T> operator + (T lhs, const Vec1Soa<T>& rhs){
		Vec1Soa<T> ans(rhs);
		return ans += lhs;
	}

	friend Vec1Soa<T> operator + (const Vec1Soa<T>& lhs, const Vec1Soa<T>& rhs)
	{
		Vec1Soa<T> ans(lhs);
		return ans += rhs;
	}

};

//host
template <typename T>
class hVec1Soa
	:public thrust::host_vector<T>
{

public:
	typedef T value_type;
	typedef typename  thrust::host_vector<T>::iterator iterator;
	T* _rawPointer;

	hVec1Soa(void){}

	hVec1Soa(unsigned int n, const T& val = T())
	{

		resize(n, val);

	}

	/*hVec1Soa(const hVec1Soa<T>& lhs)
	{

		resize(lhs.size());
		copy(lhs.begin(), lhs.end(), begin());

	}*/

	void resize(unsigned int n, T val = 0)
	{

		thrust::host_vector<T>::resize(n, val);
		_rawPointer = thrust::raw_pointer_cast(&((*this)[0]));

	}

	void clear(void)
	{

		thrust::host_vector<T>::clear();
		_rawPointer = NULL;

	}

	void push_back(T v1)
	{

		thrust::host_vector<T>::push_back(v1);
		_rawPointer = thrust::raw_pointer_cast(&((*this)[0]));

	}



	//math
	/*void operator = (const Vec1Soa<T>& rhs)
	{

		thrust::copy(rhs.begin(), rhs.end(), begin());

	}*/

	// component-wise vector add assign
	friend hVec1Soa<T>& operator += (hVec1Soa<T>& lhs, T d){
		thrust::transform(lhs.begin(), lhs.end(), make_constant_iterator(d), lhs.begin(), thrust::plus<T>());
		return lhs;
	}

	friend hVec1Soa<T>& operator += (hVec1Soa<T>& lhs, const hVec1Soa<T>& rhs){
		thrust::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(), thrust::plus<T>());
		return lhs;
	}

	friend hVec1Soa<T> operator + (const hVec1Soa<T>& lhs, T rhs)
	{

		hVec1Soa<T> ans(lhs);
		return ans += rhs;

	}

	friend hVec1Soa<T> operator + (T lhs, const hVec1Soa<T>& rhs){
		hVec1Soa<T> ans(rhs);
		return ans += lhs;
	}

	friend hVec1Soa<T> operator + (const hVec1Soa<T>& lhs, const hVec1Soa<T>& rhs)
	{
		hVec1Soa<T> ans(lhs);
		return ans += rhs;
	}

};