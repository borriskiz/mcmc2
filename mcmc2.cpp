#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <string>
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
  std::vector<std::vector<double>> DATA; // Матрица данных

public:
  int DIM; // Размерность модели

  Model(int dim, double noiseStddev, double lowBd, double upperBd,
        int batchSize)
      : DIM(dim), noiseStddev(noiseStddev), lowBound(lowBd),
        upperBound(upperBd), batchSize(batchSize) {
    trueParams = {1.0, 2.0, 3.0};
  }

  // Генерация данных с добавлением шума
  std::vector<std::vector<double>> getData() {
    if (DATA.empty()) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::normal_distribution<> noise(0.0, noiseStddev); // Нормальный шум

      std::vector<std::vector<double>> data;
      std::vector<double> function_value = function(trueParams);
      for (int i = 0; i < batchSize; ++i) {
        std::vector<double> noisy_data;
        for (int j = 0; j < function_value.size(); ++j) {
          noisy_data.push_back(function_value[j] + noise(gen)); // Добавляем шум
        }
        data.push_back(noisy_data);
      }
      DATA = data;
    }
    return DATA;
  }

  // Функция модели, которая возвращает вектор значений
  std::vector<double> function(const std::vector<double> &x) const {
    std::vector<double> result;
    // x1 + x2
    result.push_back(x[0] + x[1]);
    // x1 * x2
    result.push_back(x[0] * x[1]);
    // x2 * x3
    result.push_back(x[1] * x[2]);
    return result;
  }

  // Потенциальная энергия с использованием данных
  double U(const std::vector<double> &x,
           const std::vector<std::vector<double>> &data) const {
    double sum = 0.0;
    std::vector<double> predicted =
        function(x); // Получаем вектор предсказанных значений

    // Для каждого столбца (каждого измерения) находим разницу
    for (int i = 0; i < data.size(); ++i) {
      for (int j = 0; j < data[i].size(); ++j) {
        double diff = data[i][j] - predicted[j];
        sum +=
            (diff * diff) / (2.0 * noiseStddev * noiseStddev); // Правдоподобие
      }
    }
    return sum; // Потенциальная энергия
  }

  // Гамильтониан: H(x, v) = U(x) + K(v), где K(v) = 0.5 * v^2 (кинетическая
  // энергия)
  double Hamiltonian(const std::vector<double> &x, const std::vector<double> &v,
                     const std::vector<std::vector<double>> &data) const {
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
  std::vector<double>
  gradient(const std::vector<double> &x,
           const std::vector<std::vector<double>> &data) const {
    std::vector<double> grad(DIM, 0.0);
    std::vector<double> predicted =
        function(x); // Получаем вектор предсказанных значений

    for (int i = 0; i < data.size(); ++i) {
      for (int j = 0; j < data[i].size(); ++j) {
        double diff = data[i][j] - predicted[j];
        grad[j] += -diff * 1.0 / (noiseStddev * noiseStddev); // Градиент по x
      }
    }
    return grad;
  }
};

// Функция для численного интегрирования (метод Эйлера)
void integrate(std::vector<double> &x, std::vector<double> &v,
               const Model &model, const std::vector<std::vector<double>> &data,
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
  std::vector<std::vector<double>> data = model.getData();

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
  double num_samples = static_cast<double>(samples.size());

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

// Функция для вычисления гистограммы с частотами
std::map<double, int> computeHistogram(const std::vector<double> &data,
                                       int num_bins) {
  std::map<double, int> histogram;
  double min_val = *std::min_element(data.begin(), data.end());
  double max_val = *std::max_element(data.begin(), data.end());
  double bin_width = (max_val - min_val) / num_bins;

  for (double val : data) {
    int bin = static_cast<int>((val - min_val) / bin_width);
    if (bin == num_bins)
      bin--; // В случае, если значение на верхней границе
    double bin_center = min_val + (bin + 0.5) * bin_width;
    histogram[bin_center]++;
  }
  return histogram;
}

// Функция для записи данных гистограммы в файл
void saveHistogramToFile(const std::map<double, int> &histogram,
                         const std::string &filename, int dataSize) {
  std::ofstream outFile(filename + ".txt");
  double dSize = static_cast<double>(dataSize);
  for (const auto &elem : histogram) {
    double bin_center = elem.first;
    double count = static_cast<double>(elem.second);
    outFile << bin_center << " " << count / dSize << "\n";
  }
  outFile.close();
}

// Функция для построения гистограммы с использованием gnuplot
void plotHistogram(const std::string &filename) {
  std::string command = "gnuplot -e \"set terminal png; set output '" +
                        filename + ".png'; plot '" + filename +
                        ".txt' using 1:2 with boxes\"";
  system(command.c_str()); // Запуск gnuplot
}

int main() {
  int dim = 3;              // Количество параметров
  int num_samples = 200;    // Количество выборок
  int num_steps = 100;      // Количество шагов интегрирования
  double epsilon = 0.0001;  // Шаг по времени
  double noiseStddev = 0.1; // Стандартное отклонение шума
  double lowBound = -5.0;
  double upperBound = 5.0;
  int batchSize = 2000; // Размер batch

  Model model(dim, noiseStddev, lowBound, upperBound, batchSize);

  // Инициализация начального вектора
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

  // Получаем сгенерированные данные
  auto inputData = model.getData();
  int num_bins = 50; // Количество бинов для гистограммы

  // Генерация и сохранение гистограмм для каждого столбца
  for (int i = 0; i < inputData[0].size(); ++i) {
    std::vector<double> column_data;
    for (const auto &row : inputData) {
      column_data.push_back(row[i]);
    }

    // Вычисление гистограммы
    auto histogram = computeHistogram(column_data, num_bins);

    // Сохраняем гистограмму для каждого столбца
    std::string filename = "inputData_column_" + std::to_string(i);
    saveHistogramToFile(histogram, filename, batchSize);

    // Строим гистограмму с использованием gnuplot
    plotHistogram(filename);
  }

  return 0;
}
