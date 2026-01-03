#ifndef LINALG_H
#define LINALG_H

#include <vector>
#include <cstddef>

using Vector = std::vector<double>;
using Matrix = std::vector<Vector>;

// Create vector of given size filled with value
Vector make_vector(std::size_t n, double value = 0.0);

// Dot product of two vectors
double dot(const Vector& a, const Vector& b);

// Add two vectors
Vector add(const Vector& a, const Vector& b);

// Subtract two vectors
Vector sub(const Vector& a, const Vector& b);

// Multiply vector by scalar
Vector scalar_mul(const Vector& v, double scalar);

// Matrix transpose
Matrix transpose(const Matrix& m);

// Matrix vector multiplication
Vector matvec(const Matrix& m, const Vector& v);

// Mean squared error
double mse(const Vector& y_true, const Vector& y_pred);

#endif
