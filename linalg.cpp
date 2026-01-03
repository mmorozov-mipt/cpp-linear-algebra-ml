#include "linalg.h"
#include <stdexcept>
#include <cmath>

Vector make_vector(std::size_t n, double value) {
    return Vector(n, value);
}

double dot(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("dot: size mismatch");
    }
    double s = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        s += a[i] * b[i];
    }
    return s;
}

Vector add(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("add: size mismatch");
    }
    Vector res(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        res[i] = a[i] + b[i];
    }
    return res;
}

Vector sub(const Vector& a, const Vector& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("sub: size mismatch");
    }
    Vector res(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        res[i] = a[i] - b[i];
    }
    return res;
}

Vector scalar_mul(const Vector& v, double scalar) {
    Vector res(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        res[i] = v[i] * scalar;
    }
    return res;
}

Matrix transpose(const Matrix& m) {
    if (m.empty()) {
        return Matrix{};
    }
    std::size_t rows = m.size();
    std::size_t cols = m[0].size();
    Matrix res(cols, Vector(rows));
    for (std::size_t i = 0; i < rows; ++i) {
        if (m[i].size() != cols) {
            throw std::runtime_error("transpose: inconsistent row size");
        }
        for (std::size_t j = 0; j < cols; ++j) {
            res[j][i] = m[i][j];
        }
    }
    return res;
}

Vector matvec(const Matrix& m, const Vector& v) {
    if (m.empty()) {
        return Vector{};
    }
    std::size_t rows = m.size();
    std::size_t cols = m[0].size();
    if (v.size() != cols) {
        throw std::runtime_error("matvec: size mismatch");
    }
    Vector res(rows, 0.0);
    for (std::size_t i = 0; i < rows; ++i) {
        if (m[i].size() != cols) {
            throw std::runtime_error("matvec: inconsistent row size");
        }
        double s = 0.0;
        for (std::size_t j = 0; j < cols; ++j) {
            s += m[i][j] * v[j];
        }
        res[i] = s;
    }
    return res;
}

double mse(const Vector& y_true, const Vector& y_pred) {
    if (y_true.size() != y_pred.size()) {
        throw std::runtime_error("mse: size mismatch");
    }
    if (y_true.empty()) {
        return 0.0;
    }
    double s = 0.0;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        double diff = y_true[i] - y_pred[i];
        s += diff * diff;
    }
    return s / static_cast<double>(y_true.size());
}
