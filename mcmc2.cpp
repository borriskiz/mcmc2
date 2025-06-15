#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

// Генерация случайного вектора с параметрами в пределах [lowBound, upperBound]
std::vector<double> generateRandomVector(int n, double lowerBd,
                                         double upperBd) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(lowerBd, upperBd);
  std::vector<double> result;
  for (int i = 0; i < n; ++i) {
    result.push_back(dis(gen));
  }
  return result;
}

// Класс для модели
class Model {
private:
  std::vector<double> trueParams;
  double noiseStddev; // Стандартное отклонение шума
  double lowBound, upperBound;
  int batchSize;

public:
  int DIM; // Размерность модели

  Model(int dim, double noiseStddev, double lowBd, double upperBd,
        int batchSize)
      : DIM(dim), noiseStddev(noiseStddev), lowBound(lowBd),
        upperBound(upperBd), batchSize(batchSize) {
    trueParams = generateRandomVector(DIM, lowBound, upperBound);
  }

  // Функция модели, которую будем использовать для генерации данных
  double function(const std::vector<double> &x) const {
    double result = 0.0;
    for (int i = 0; i < x.size(); ++i) {
      result += x[i]; // Простая линейная модель
    }
    return result;
  }

  // Генерация данных с добавлением шума
  std::vector<double> generateData() const {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0.0, noiseStddev); // Нормальный шум

    std::vector<double> data;
    double function_value = function(trueParams);
    for (int i = 0; i < batchSize; ++i) {
      data.push_back(function_value + noise(gen));
    }
    return data;
  }

  // Потенциальная энергия с использованием данных
  double U(const std::vector<double> &x,
           const std::vector<double> &data) const {
    double sum = 0.0;
    for (int i = 0; i < data.size(); ++i) {
      double predicted = function(x);    // Прогнозируемое значение
      double diff = data[i] - predicted; // Разница с измерением
      sum += (diff * diff) / (2.0 * noiseStddev * noiseStddev); // Правдоподобие
    }
    return sum; // Потенциальная энергия
  }

  // Гамильтониан: H(x, v) = U(x) + K(v), где K(v) = 0.5 * v^2 (кинетическая
  // энергия)
  double Hamiltonian(const std::vector<double> &x, const std::vector<double> &v,
                     const std::vector<double> &data) const {
    double U_val = U(x, data); // Потенциальная энергия
    double K_val = 0.0;
    for (double vi : v) {
      K_val += vi * vi;
    }
    return U_val + 0.5 * K_val;
  }

  // Метод для генерации случайного импульса
  std::vector<double> generateRandomMomentum() const {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, 1.0); // нормальное распределение
    std::vector<double> momentum(DIM);
    for (int i = 0; i < DIM; ++i) {
      momentum[i] = dis(gen);
    }
    return momentum;
  }

  // Функция для вычисления градиента
  std::vector<double> gradient(const std::vector<double> &x,
                               const std::vector<double> &data) const {
    std::vector<double> grad(DIM, 0.0);
    for (int i = 0; i < data.size(); ++i) {
      double predicted = function(x);    // Прогнозируемое значение
      double diff = data[i] - predicted; // Разница с измерением
      for (int j = 0; j < DIM; ++j) {
        grad[j] += -diff * 1.0 / (noiseStddev * noiseStddev); // Градиент по x
      }
    }
    return grad;
  }
};

// Функция для численного интегрирования (метод Эйлера)
void integrate(std::vector<double> &x, std::vector<double> &v,
               const Model &model, const std::vector<double> &data,
               double epsilon, int num_steps) {
  for (int i = 0; i < num_steps; ++i) {
    // Обновление x по v
    for (int j = 0; j < x.size(); ++j) {
      x[j] += epsilon * v[j];
    }

    // Вычисляем градиент
    std::vector<double> grad = model.gradient(x, data);

    // Обновление v по градиенту
    for (int j = 0; j < v.size(); ++j) {
      v[j] -= epsilon * grad[j];
    }
  }
}

// Функция для выполнения HMC
std::vector<std::vector<double>> hmc(Model &model,
                                     const std::vector<double> &initial_x,
                                     int num_samples, double epsilon,
                                     int num_steps) {
  std::vector<double> x = initial_x;
  std::vector<std::vector<double>>
      samples; // Матрица выборок (num_samples x DIM)

  // Генерация данных (batch)
  std::vector<double> data = model.generateData();

  for (int n = 0; n < num_samples; ++n) {
    // 1. Генерация случайного импульса
    std::vector<double> v = model.generateRandomMomentum();

    // 2. Чтение начальных условий для HMC
    std::vector<double> x_new = x;
    std::vector<double> v_new = v;

    // 3. Численное интегрирование
    integrate(x_new, v_new, model, data, epsilon, num_steps);

    // 4. Вычисление гамильтониана
    double H_old = model.Hamiltonian(x, v, data);
    double H_new = model.Hamiltonian(x_new, v_new, data);

    // 5. Принятие решения по методу Метрополиса
    double alpha = std::min(1.0, exp(H_old - H_new));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    double u = dis(gen);

    if (u < alpha) {
      x = x_new; // Принимаем новую точку
    }

    // Добавляем выборку в матрицу
    samples.push_back(x); // Каждая выборка — вектор с размерностью DIM
  }

  return samples;
}

// Функция для вычисления среднего по каждому параметру
std::vector<double>
computeMean(const std::vector<std::vector<double>> &samples) {
  std::vector<double> mean(samples[0].size(), 0.0);
  int num_samples = samples.size();

  for (const auto &sample : samples) {
    for (int i = 0; i < sample.size(); ++i) {
      mean[i] += sample[i];
    }
  }

  // Делим на количество выборок для получения среднего
  for (int i = 0; i < mean.size(); ++i) {
    mean[i] /= num_samples;
  }

  return mean;
}

int main() {
  int dim = 3;              // Количество параметров
  int num_samples = 500;    // Количество выборок
  int num_steps = 1000;       // Количество шагов интегрирования
  double epsilon = 0.0001;     // Шаг по времени
  double noiseStddev = 0.1; // Стандартное отклонение шума
  double lowBound = -5.0;
  double upperBound = 5.0;
  int batchSize = 2000; // Размер batch

  Model model(dim, noiseStddev, lowBound, upperBound, batchSize);

  // Инициализация начального вектора
  // std::vector<double> initial_x =
  //    generateRandomVector(dim, lowBound, upperBound);
  std::vector<double> initial_x = std::vector<double>(dim, 0.0);
  // Запуск HMC
  std::vector<std::vector<double>> samples =
      hmc(model, initial_x, num_samples, epsilon, num_steps);

  // Вычисление среднего по каждому параметру
  std::vector<double> mean = computeMean(samples);

  std::cout << "\nMean of each parameter after " << num_samples
            << " samples:" << std::endl;
  for (double m : mean) {
    std::cout << m << " ";
  }
  std::cout << std::endl;

  return 0;
}
