#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace std;

// Генерация случайных чисел из нормального распределения
double normal_random(double mean, double stddev) {
  static random_device rd;
  static default_random_engine generator(rd());
  normal_distribution<double> distribution(mean, stddev);
  return distribution(generator);
}

// Модель F(x): x_0 + x_1 * x_2
double model(const vector<double> &x) {
  return x[0] + x[1] * x[2]; // x1 + x2 * x3
}

// Генерация данных m с ошибкой
vector<double> generate_data(int batch_size, const vector<double> &true_params,
                             double noise_stddev) {
  vector<double> data(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    data[i] =
        model(true_params) + normal_random(0.0, noise_stddev); // Добавляем шум
  }
  return data;
}

// Логарифм правдоподобия
double log_likelihood(const vector<double> &x, const vector<double> &m,
                      double sigma) {
  double log_likelihood = 0.0;
  for (size_t i = 0; i < m.size(); ++i) {
    double F_x = model(x);
    log_likelihood +=
        -0.5 * pow((m[i] - F_x) / sigma, 2); // Логарифм правдоподобия
  }
  return log_likelihood;
}

// Градиент логарифма правдоподобия
vector<double> likelihood_gradient(const vector<double> &x,
                                   const vector<double> &m, double sigma) {
  vector<double> grad(x.size(), 0.0);
  for (size_t i = 0; i < m.size(); ++i) {
    double F_x = model(x);
    for (size_t j = 0; j < x.size(); ++j) {
      // Для простоты градиент по каждой переменной
      grad[j] += (F_x - m[i]) / (sigma * sigma) *
                 ((j == 0) ? 1.0 : ((j == 1) ? x[2] : x[1]));
    }
  }
  return grad;
}

// Кинетическая энергия
double kinetic_energy(const vector<double> &p) {
  double energy = 0.0;
  for (double pi : p) {
    energy += 0.5 * pi * pi; // Простая кинетическая энергия
  }
  return energy;
}

// Обновление параметров x и импульсов p
void hmc_step(vector<double> &x, vector<double> &p, double epsilon,
              const vector<double> &m, double sigma) {
  // Шаг 1: Обновление импульса p
  vector<double> grad_L = likelihood_gradient(x, m, sigma);
  for (size_t i = 0; i < p.size(); ++i) {
    p[i] -= 0.5 * epsilon * grad_L[i]; // Обновление импульса
  }

  // Шаг 2: Обновление параметров x
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] += epsilon * p[i]; // Обновление параметров по импульсу
  }

  // Шаг 3: Обновление импульса p
  grad_L = likelihood_gradient(x, m, sigma); // Пересчитываем градиент с новым x
  for (size_t i = 0; i < p.size(); ++i) {
    p[i] -= 0.5 * epsilon * grad_L[i]; // Обновление импульса
  }
}

// Полная энергия (потенциальная + кинетическая)
double total_energy(const vector<double> &x, const vector<double> &p,
                    const vector<double> &m, double sigma) {
  double U = -log_likelihood(x, m, sigma); // Потенциальная энергия
  double K = kinetic_energy(p);            // Кинетическая энергия
  return U + K;                            // Полная энергия
}

// Правило приема/отклонения
void accept_or_reject(vector<double> &x, vector<double> &p, double epsilon,
                      const vector<double> &m, double sigma,
                      double &old_energy) {
  vector<double> old_x = x;
  vector<double> old_p = p;

  // Выполняем шаг HMC
  hmc_step(x, p, epsilon, m, sigma);

  // Вычисляем новую полную энергию
  double new_energy = total_energy(x, p, m, sigma);

  // Прием/отклонение на основе изменения энергии
  double energy_diff = new_energy - old_energy;
  if (energy_diff <= 0 || normal_random(0.0, 1.0) < exp(-energy_diff)) {
    old_energy = new_energy; // Принимаем новое состояние
  } else {
    x = old_x;
    p = old_p; // Отклоняем изменения
  }
}

// Главная функция
int main() {
  const int DIM = 3;            // Размерность параметров
  const double epsilon = 0.01;  // Шаг интегрирования
  const double sigma = 0.1;     // Стандартное отклонение шума
  const int BATCH_SIZE = 2000;  // Количество наблюдений
  const int NUM_SAMPLES = 500;  // Количество сэмплов
  const int WARMUP_STEPS = 300; // Количество шагов разогрева

  vector<double> true_parameters = {1.0, 2.0, 3.0}; // Истинные параметры
  vector<double> m =
      generate_data(BATCH_SIZE, true_parameters, sigma); // Генерация данных m

  // Инициализация параметров x и импульсов p
  vector<double> x(DIM, 0.0); // Начальные параметры (x)
  vector<double> p(DIM, 0.0); // Начальные импульсы (p)

  // Генерация случайного импульса
  for (int i = 0; i < DIM; ++i) {
    p[i] = normal_random(0.0, 1.0); // Инициализация случайным моментом
  }

  // MCMC с HMC
  double old_energy =
      total_energy(x, p, m, sigma); // Инициализация полной энергии
  for (int step = 0; step < NUM_SAMPLES + WARMUP_STEPS; ++step) {
    accept_or_reject(x, p, epsilon, m, sigma, old_energy);
  }

  // Вывод результата
  cout << "Оцененные параметры x: ";
  for (double xi : x) {
    cout << xi << " ";
  }
  cout << endl;

  return 0;
}
