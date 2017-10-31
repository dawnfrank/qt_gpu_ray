#pragma once

#include "define.cuh"

template <typename T>
class Vector {
public:
	typedef T* iterator;

	__device__ Vector() :_size(0), _capacity(DEFAULT_VECTOR_SIZE) {
		_p = new T[10];
	}
	__device__ ~Vector() { delete[] _p; }
	__device__ Vector &operator= (const Vector& vec) {
		delete _p;
		_size = vec._size;
		_capacity = vec._capacity;
		_p = new T[_capacity];
		for (int i = 0; i != _size; ++i) {
			_p[i] = vec._p[i];
		}
	}


	__device__ T& operator[](int index) { return _p[index]; }
	__device__ const T& operator[](int index) const { return _p[index]; }

	__device__ void clear() { _size = 0; }
	__device__ void resize(int newSize) { reserve(newSize); }
	__device__ void reserve(int newCapacity) {
		if (newCapacity < _capacity) return;

		T* temp = _p;
		_p = new T[newCapacity];
		for (int i = 0; i < _size; ++i) {
			_p[i] = temp[i];
		}
		_capacity = newCapacity;

		delete[] temp;
	}

	__device__ void push_back(T obj) {
		if (_size == _capacity) reserve(_capacity * 2 + 1);
		_p[_size++] = obj;
	}
	__device__ void pop_back() {
		if (_size != 0)--_size;
	}
	__device__ void erase(iterator first, iterator end) {
		int startIndex = 0;
		for (int i = 0; i != _size; ++i) {
			if ((_p + i) == first) {
				startIndex = i;
				break;
			}
		}
		int endIndex = end - first + startIndex;
		if (endIndex > _size)endIndex = _size + 1;

		int length = _size + 1 - endIndex;

		for (int i = 0; i != length; ++i) {
			*(_p + i + startIndex) = *(_p + i + endIndex);
		}
		_size -= end - first;
	}
	__device__ void erase(iterator obj) {
		erase(obj, obj + 1);
	}

	__device__ iterator begin() { return &_p[0]; }
	__device__ iterator end() { return &_p[_size]; }
	__device__ const iterator cbegin() const { return &_p[0]; }
	__device__ const iterator cend() const { return &_p[_size]; }

	__device__ bool emptry() { return _size == 0; }
	__device__ int size() { return _size; }
	__device__ int capacity() { return _capacity; }
private:
	int _size;
	int _capacity;
	T *_p;
};