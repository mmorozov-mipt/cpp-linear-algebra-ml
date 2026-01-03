#include <iostream>
#include <random>
#include "linalg.h"

int main() {
    // Synthetic data: y = a_true * x + b_true + noise
    const double a_true = 3.5;
    const double b_true = 2.0;

    std::size_t n_samples = 100;
    Matrix X(n_samples, Vector(1));
    Vector y(n_samples);

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> x_dist(-5.0, 5.0);
    std::normal_distribution<double> noise_dist(0.0, 1.0);

    for (std::size_t i = 0; i < n_samples; ++i) {
        double x = x_dist(gen);
        double noise = noise_dist(gen);
        X[i][0] = x;
        y[i] = a_true * x + b_true + noise;
    }

    // Model: y = a * x + b
    double a = 0.0;
    double b = 0.0;

    double lr = 0.01;
    std::size_t epochs = 2000;

    Vector y_pred(n_samples);

    for (std::size_t epoch = 0; epoch < epochs; ++epoch) {
        // forward
        for (std::size_t i = 0; i < n_samples; ++i) {
            double x = X[i][0];
            y_pred[i] = a * x + b;
        }

        // compute gradients
        double grad_a = 0.0;
        double grad_b = 0.0;
        for (std::size_t i = 0; i < n_samples; ++i) {
            double diff = y_pred[i] - y[i];
            grad_a += diff * X[i][0];
            grad_b += diff;
        }
        grad_a = 2.0 * grad_a / static_cast<double>(n_samples);
        grad_b = 2.0 * grad_b / static_cast<double>(n_samples);

        // update
        a -= lr * grad_a;
        b -= lr * grad_b;

        if (epoch % 200 == 0 || epoch == epochs - 1) {
            double loss = mse(y, y_pred);
            std::cout << "Epoch " << epoch
                      << " MSE = " << loss
                      << " a = " << a
                      << " b = " << b << "\n";
        }
    }

    std::cout << "\nTrue parameters: a = " << a_true
              << " b = " << b_true << "\n";
    std::cout << "Learned parameters: a = " << a
              << " b = " << b << "\n";

    return 0;
}
