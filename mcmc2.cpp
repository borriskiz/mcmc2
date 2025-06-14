#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

// Функция для генерации случайных чисел из нормального распределения
double normal_random(double mean, double stddev) {
  static random_device rd;
  static default_random_engine generator(rd());
  normal_distribution<double> distribution(mean, stddev);
  return distribution(generator);
}

// Модель F(x) — линейная зависимость
double model(const vector<double> &x) {
  return x[0] + x[1] * x[2]; // x1 + x2 * x3
}

// Функция для вычисления логарифма правдоподобия с учетом инверсии
double log_likelihood(const vector<double> &x, double m, double sigma) {
  double F_x = model(x);                   // Модель F(x)
  return -0.5 * pow((m - F_x) / sigma, 2); // Логарифм правдоподобия
}

// Функция для вычисления кинетической энергии
double kinetic_energy(const vector<double> &p) {
  double energy = 0.0;
  for (double pi : p) {
    energy += 0.5 * pi * pi; // Простая кинетическая энергия
  }
  return energy;
}

// Функция для вычисления градиента логарифма правдоподобия
vector<double> likelihood_gradient(const vector<double> &x, double m,
                                   double sigma) {
  double F_x = model(x);
  double dF_dx1 = 1.0;
  double dF_dx2 = x[2];
  double dF_dx3 = x[1];

  vector<double> grad(3);
  grad[0] = (F_x - m) / (sigma * sigma) * dF_dx1;
  grad[1] = (F_x - m) / (sigma * sigma) * dF_dx2;
  grad[2] = (F_x - m) / (sigma * sigma) * dF_dx3;

  return grad;
}

// Функция для обновления x и p с использованием метода Эйлера
void hmc_step(vector<double> &x, vector<double> &p, double epsilon, double m,
              double sigma) {
  // Шаг 1: Обновление импульса p (на основе градиента логарифма правдоподобия)
  vector<double> grad_L = likelihood_gradient(x, m, sigma);
  for (size_t i = 0; i < p.size(); ++i) {
    p[i] -= 0.5 * epsilon * grad_L[i]; // Обновление импульса
  }

  // Шаг 2: Обновление параметров x
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] += epsilon * p[i]; // Обновление параметров по импульсу
  }

  // Шаг 3: Обновление импульса p (на основе градиента логарифма правдоподобия)
  grad_L = likelihood_gradient(x, m, sigma); // Пересчитываем градиент с новым x
  for (size_t i = 0; i < p.size(); ++i) {
    p[i] -= 0.5 * epsilon * grad_L[i]; // Обновление импульса
  }
}

int main() {
  // Инициализация случайных данных
  int DIM = 3;
  double epsilon = 0.1; // Шаг интегрирования
  double m = 10.0;      // Измеренные данные
  double sigma = 0.1;   // Стандартное отклонение шума

  vector<double> x(DIM, 0.0); // Начальные параметры (параметры модели)
  vector<double> p(DIM, 0.0); // Начальные импульсы

  // Генерация случайного момента (импульса)
  for (int i = 0; i < DIM; ++i) {
    p[i] = normal_random(0.0, 1.0); // Инициализация случайным моментом
  }

  // Печать начальных значений
  cout << "Initial x: ";
  for (double xi : x)
    cout << xi << " ";
  cout << endl;

  cout << "Initial momentum p: ";
  for (double pi : p)
    cout << pi << " ";
  cout << endl;

  // Вычисление логарифма правдоподобия
  double log_like = log_likelihood(x, m, sigma);
  cout << "Initial log likelihood: " << log_like << endl;

  // Применение одного шага HMC
  hmc_step(x, p, epsilon, m, sigma);

  // Печать обновленных значений после одного шага
  cout << "\nAfter one HMC step:" << endl;
  cout << "Updated x: ";
  for (double xi : x)
    cout << xi << " ";
  cout << endl;

  cout << "Updated momentum p: ";
  for (double pi : p)
    cout << pi << " ";
  cout << endl;

  // Вычисление логарифма правдоподобия после шага
  log_like = log_likelihood(x, m, sigma);
  cout << "Updated log likelihood: " << log_like << endl;

  return 0;
}
