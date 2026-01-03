# cpp-linear-algebra-ml

Minimal linear algebra utilities in C++ for simple machine learning tasks (for example, linear regression trained with gradient descent).

This project demonstrates:
- work with std::vector for vectors and matrices
- basic linear algebra operations
- implementation of mean squared error
- simple gradient descent training loop for linear regression

Contents:
- linalg.h - type aliases and function declarations
- linalg.cpp - implementation of vector and matrix operations
- main.cpp - example: training 1D linear regression y = a * x + b on synthetic data

Build (macOS, Linux):

g++ -std=c++17 -O2 main.cpp linalg.cpp -o linear_regression_demo

Run:

./linear_regression_demo

You will see:
- training progress in the console (MSE by epochs)
- final learned parameters (a and b)
- comparison with true parameters used for data generation

Disclaimer:

This project is for educational and portfolio purposes only.
