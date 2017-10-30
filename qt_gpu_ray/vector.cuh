#pragma once

#include "define.cuh"

template <typename T>
class Vector {
public:
	typedef T* iterator;

	Vector() :_size(0), _capacity(DEFAULT_VECTOR_SIZE) {
		_p = new T[10];
	}
	~Vector() { delete[] _p; }
	Vector &operator= (const Vector& vec) {
		delete _p;
		_size = vec._size;
		_capacity = vec._capacity;
		_p = new T[_capacity];
		for (int i = 0; i != _size; ++i) {
			_p[i] = vec._p[i];
		}
	}


	T& operator[](int index) { return _p[index]; }
	const T& operator[](int index) const { return _p[index]; }

	void resize(int newSize) { reserve(newSize); }
	void reserve(int newCapacity) {
		if (newCapacity < _capacity) return;

		T* temp = _p;
		_p = new T[newCapacity];
		for (int i = 0; i < _size; ++i) {
			_p[i] = temp[i];
		}
		_capacity = newCapacity;

		delete[] temp;
	}

	void push_back(T obj) {
		if (_size == _capacity) reserve(_capacity * 2 + 1);
		_p[_size++] = obj;
	}
	void pop_back() {
		if (_size != 0)--_size;
	}

	iterator begin() { return &_p[0]; }
	iterator end() { return &_p[_size]; }
	const iterator cbegin() const {return &_p[0];}
	const iterator cend() const { return &_p[_size]; }

	bool emptry() { return _size == 0; }
	int size() { return _size; }
	int capacity() { return _capacity; }
private:
	int _size;
	int _capacity;
	T *_p;
};