#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

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

class Model {
private:
  std::vector<double> trueParams;
  double noiseStddev;
  double lowBound, upperBound;
  int batchSize;
  std::vector<std::vector<double>> DATA;

public:
  int DIM;

  Model(int dim, double noiseStddev, double lowBd, double upperBd,
        int batchSize)
      : DIM(dim), noiseStddev(noiseStddev), lowBound(lowBd),
        upperBound(upperBd), batchSize(batchSize) {
    trueParams = generateRandomVector(dim, lowBound, upperBound);
  }

  std::vector<std::vector<double>> getData() {
    if (DATA.empty()) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::normal_distribution<> noise(0.0, noiseStddev);

      std::vector<std::vector<double>> data;
      std::vector<double> function_value = function(trueParams);
      for (int i = 0; i < batchSize; ++i) {
        std::vector<double> noisy_data;
        for (int j = 0; j < function_value.size(); ++j) {
          noisy_data.push_back(function_value[j] + noise(gen));
        }
        data.push_back(noisy_data);
      }
      DATA = data;
    }
    return DATA;
  }

  std::vector<double> getTrue() const { return trueParams; }

  std::vector<double> function(const std::vector<double> &x) const {
    std::vector<double> result;
    result.push_back(x[0] * x[1]); // x1 * x2
    result.push_back(x[1] + x[2]); // x1 + x2
    result.push_back(x[2] * x[0]); // x2 * x3
    return result;
  }

  double U(const std::vector<double> &x,
           const std::vector<double> &data) const {
    double sum = 0.0;
    std::vector<double> predicted = function(x);
    for (int j = 0; j < data.size(); ++j) {
      double diff = data[j] - predicted[j];
      sum += (diff * diff) / (2.0 * noiseStddev * noiseStddev);
    }
    return sum;
  }

  double Hamiltonian(const std::vector<double> &x, const std::vector<double> &v,
                     const std::vector<double> &data) const {
    double U_val = U(x, data); // Потенциальная энергия
    double K_val = 0.0;
    for (double vi : v) {
      K_val += vi * vi;
    }
    return U_val + 0.5 * K_val; // Полный гамильтониан
  }

  std::vector<double> generateRandomMomentum() const {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0.0, 1.0);
    std::vector<double> momentum(DIM);
    for (int i = 0; i < DIM; ++i) {
      momentum[i] = dis(gen);
    }
    return momentum;
  }

  std::vector<double> gradient(const std::vector<double> &x,
                               const std::vector<double> &data) const {
    std::vector<double> grad(DIM, 0.0);
    std::vector<double> predicted = function(x);
    double noiseStddev2 = noiseStddev * noiseStddev;
    for (int j = 0; j < data.size(); ++j) {
      double diff = data[j] - predicted[j];
      grad[j] += -diff / noiseStddev2;
    }
    return grad;
  }
};

void integrate(std::vector<double> &x, std::vector<double> &v,
               const Model &model, const std::vector<double> &data,
               double epsilon, int num_steps) {
  std::vector<double> grad, v_temp, x_temp;
  for (int i = 0; i < num_steps; ++i) {
    for (int j = 0; j < x.size(); ++j) {
      x[j] += epsilon * v[j];
    }

    grad = model.gradient(x, data);

    v_temp = v;
    x_temp = x;
    for (int j = 0; j < v.size(); ++j) {
      v_temp[j] -= epsilon * grad[j];
    }

    for (int j = 0; j < v.size(); ++j) {
      v[j] += epsilon * v_temp[j];
    }
  }
}

std::vector<std::vector<double>> hmc(Model &model,
                                     const std::vector<double> &initial_x,
                                     int num_samples, double epsilon,
                                     int num_steps) {
  std::vector<double> x = initial_x;
  std::vector<std::vector<double>> samples;
  std::vector<double> v, x_new, v_new;
  std::vector<std::vector<double>> data = model.getData();

  for (int n = 0; n < num_samples; ++n) {
    if (n % (num_samples / 10) == 0) {
      std::cout << "Progress: " << (n * 100) / num_samples << "%\n";
    }

    v = model.generateRandomMomentum();
    x_new = x;
    v_new = v;

    integrate(x_new, v_new, model, data[0], epsilon, num_steps);

    double H_old = model.Hamiltonian(x, v, data[0]);
    double H_new = model.Hamiltonian(x_new, v_new, data[0]);

    double alpha = std::min(1.0, exp(H_old - H_new));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    double u = dis(gen);

    if (u < alpha) {
      x = x_new;
    }

    samples.push_back(x);
  }

  return samples;
}

std::vector<double>
computeMean(const std::vector<std::vector<double>> &samples) {
  std::vector<double> mean(samples[0].size(), 0.0);
  for (const auto &sample : samples) {
    for (int i = 0; i < sample.size(); ++i) {
      mean[i] += sample[i];
    }
  }
  for (int i = 0; i < mean.size(); ++i) {
    mean[i] /= samples.size();
  }
  return mean;
}

void printVector(const std::vector<double> &vec) {
  for (double v : vec) {
    std::cout << v << " ";
  }
  std::cout << std::endl;
}
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
  system(command.c_str());
}
int main() {
  int dim = 3;
  int sampleSize = 20000;
  int num_steps = 1000;
  double epsilon = 0.0001;
  double noiseStddev = 0.1;
  double lowBound = -5.0;
  double upperBound = 5.0;
  int batchSize = 2000;

  Model model(dim, noiseStddev, lowBound, upperBound, batchSize);

  // Инициализация начального вектора
  std::vector<double> initial_x = std::vector<double>(dim, 0.0);

  // Запуск HMC
  std::vector<std::vector<double>> samples =
      hmc(model, initial_x, sampleSize, epsilon, num_steps);

  // Вычисление среднего по каждому параметру
  std::vector<double> mean = computeMean(samples);

  std::cout << "\nMean of each parameter after " << sampleSize
            << " samples:" << std::endl;
  printVector(mean);

  std::cout << "\nTrue parameters\n ";
  std::vector<double> trueParams = model.getTrue();
  printVector(trueParams);

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
    saveHistogramToFile(histogram, filename, sampleSize);

    // Строим гистограмму с использованием gnuplot
    plotHistogram(filename);
  }

  return 0;
}
