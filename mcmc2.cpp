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
    trueParams = {1.0, 2.0, 3.0};
    trueParams = generateRandomVector(dim, lowBound, upperBound);
  }

  std::vector<std::vector<double>> getData() {
    if (!DATA.empty()) {
      return DATA;
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0.0, noiseStddev);

    std::vector<std::vector<double>> data;
    std::vector<double> function_value = function(trueParams);
    std::vector<double> noisy_data;
    int size = function_value.size();
    for (int i = 0; i < batchSize; ++i) {
      noisy_data.clear();
      for (int j = 0; j < size; ++j) {
        noisy_data.push_back(function_value[j] + noise(gen));
      }
      data.push_back(noisy_data);
    }
    DATA = data;
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
    double diff;
    int size = data.size();
    for (int i = 0; i < size; ++i) {
      diff = data[i] - predicted[i];
      sum += (diff * diff);
    }
    return sum / (2.0 * noiseStddev * noiseStddev);
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
    double diff;
    int size = data.size();
    for (int j = 0; j < size; ++j) {
      diff = data[j] - predicted[j];
      grad[j] += -diff / noiseStddev2;
    }
    return grad;
  }
};

void integrate(std::vector<double> &x, std::vector<double> &v,
               const Model &model, const std::vector<double> &data,
               double epsilon, int num_steps) {
  std::vector<double> grad, v_temp, x_temp;
  int xSize = x.size();
  int vSize = v.size();
  for (int i = 0; i < num_steps; ++i) {
    for (int j = 0; j < xSize; ++j) {
      x[j] += epsilon * v[j];
    }

    grad = model.gradient(x, data);

    v_temp = v;
    x_temp = x;
    for (int j = 0; j < vSize; ++j) {
      v_temp[j] -= epsilon * grad[j];
    }

    for (int j = 0; j < vSize; ++j) {
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
  double H_old, H_new, alpha, u;
  int accepted = 0;
  for (int n = 0; n < num_samples; ++n) {
    if (n % (num_samples / 10) == 0) {
      std::cout << "Progress: " << (n * 100) / num_samples << "%\n";
    }

    v = model.generateRandomMomentum();
    x_new = x;
    v_new = v;

    integrate(x_new, v_new, model, data[0], epsilon, num_steps);

    H_old = model.Hamiltonian(x, v, data[0]);
    H_new = model.Hamiltonian(x_new, v_new, data[0]);

    alpha = std::min(1.0, exp(H_old - H_new));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    u = dis(gen);

    if (u < alpha) {
      x = x_new;
      accepted++;
    }

    samples.push_back(x);
  }
  // Диагностика: процент принятия
  double acceptance_rate = static_cast<double>(accepted) / num_samples;
  std::cout << "Acceptance rate: " << acceptance_rate * 100.0 << "%"
            << std::endl;
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
// Функция для записи трассировки для каждого параметра в отдельный файл
void saveTracePlot(const std::vector<std::vector<double>> &samples,
                   int param_idx) {
  std::ofstream outFile("trace_plot_x" + std::to_string(param_idx + 1) +
                        ".txt");
  for (const auto &sample : samples) {
    outFile << sample[param_idx] << "\n"; // Записываем только значения для x_i
  }
  outFile.close();

  // Строим график трассировки с помощью gnuplot
  std::string command =
      "gnuplot -e \"set terminal png; set output 'trace_plot_x" +
      std::to_string(param_idx + 1) + ".png'; plot 'trace_plot_x" +
      std::to_string(param_idx + 1) + ".txt' with lines title 'x" +
      std::to_string(param_idx + 1) + "\"";
  system(command.c_str());
}
std::vector<double> autocorrelation(const std::vector<double> &chain,
                                    int max_lag = 100) {
  int n = chain.size();
  std::vector<double> acors(max_lag + 1, 0.0);

  // Вычисляем среднее значение для цепи
  double mean = 0.0;
  for (double value : chain) {
    mean += value;
  }
  mean /= n;

  // Создаем копию цепи с нулевым средним
  std::vector<double> chain_centered(n);
  for (int i = 0; i < n; ++i) {
    chain_centered[i] = chain[i] - mean;
  }

  // Вычисляем автокорреляцию для каждого лага
  for (int lag = 0; lag <= max_lag; ++lag) {
    double unshifted_norm = 0.0;
    double shifted_norm = 0.0;
    double cross_correlation = 0.0;

    for (int i = 0; i < n - lag; ++i) {
      unshifted_norm += chain_centered[i] * chain_centered[i];
      shifted_norm += chain_centered[i + lag] * chain_centered[i + lag];
      cross_correlation += chain_centered[i] * chain_centered[i + lag];
    }

    acors[lag] = cross_correlation / std::sqrt(unshifted_norm * shifted_norm);
  }
  return acors;
}
void saveAutocorrelationToFile(const std::vector<double> &acors,
                               int param_idx) {
  std::ofstream outFile("autocorrelation_param_" +
                        std::to_string(param_idx + 1) + ".txt");
  for (int i = 0; i < acors.size(); ++i) {
    outFile << i << " " << acors[i] << "\n";
  }
  outFile.close();
}
void plotAutocorrelation(int param_idx) {
  std::string command =
      "gnuplot -e \"set terminal png; set output 'autocorrelation_param_" +
      std::to_string(param_idx + 1) + ".png'; plot 'autocorrelation_param_" +
      std::to_string(param_idx + 1) +
      ".txt' using 1:2 with lines title 'ACF of x" +
      std::to_string(param_idx + 1) + "\"";
  system(command.c_str());
}
std::vector<std::vector<double>> filterChainByACF(
    const std::vector<std::vector<double>> &samples,
    double acf_threshold = 0.1, // Порог автокорреляции для фильтрации
    int acf_lag = 100,          // Максимальный лаг для автокорреляции
    int skip_rate = 100 // Сколько точек пропускать для оценки автокорреляции
) {
  std::vector<std::vector<double>> filtered_samples;
  std::vector<std::vector<double>> chain(samples.begin(), samples.end());

  // Применяем фильтрацию для каждого параметра
  for (int param_idx = 0; param_idx < samples[0].size(); ++param_idx) {
    std::vector<double> param_chain;
    for (const auto &sample : samples) {
      param_chain.push_back(sample[param_idx]);
    }

    // Вычисление автокорреляции
    std::vector<double> acors = autocorrelation(param_chain, acf_lag);

    // Ищем момент, когда автокорреляция стабилизируется
    bool stop_filtering = false;
    int start_index = 0;
    for (int i = 0; i < acors.size(); ++i) {
      if (abs(acors[i]) < acf_threshold) {
        start_index = i;
        stop_filtering = true;
        break;
      }
    }

    // Применяем фильтрацию, начиная с момента, когда автокорреляция стала
    // меньше порога
    for (int i = start_index; i < samples.size(); i += skip_rate) {
      filtered_samples.push_back(samples[i]);
    }
  }

  return filtered_samples;
}
std::vector<std::vector<double>> nuts(Model &model,
                                      const std::vector<double> &initial_x,
                                      int num_samples, double epsilon,
                                      int max_depth) {
  std::vector<double> x = initial_x;
  std::vector<std::vector<double>> samples;
  std::vector<double> v, x_new, v_new;
  std::vector<std::vector<double>> data = model.getData();
  double H_new, alpha, u;
  int accepted = 0;

  // Для отслеживания траектории
  std::vector<std::vector<std::vector<double>>> trajectory;

  for (int n = 0; n < num_samples; ++n) {
    if (n % (num_samples / 10) == 0) {
      std::cout << "Progress: " << (n * 100) / num_samples << "%\n";
    }

    // Сгенерируем случайный импульс
    v = model.generateRandomMomentum();
    x_new = x;
    v_new = v;

    trajectory.clear(); // очищаем траекторию перед каждым новым шагом

    // Рекурсивно интегрируем (аналогично NUTS)
    double epsilon_temp = epsilon;
    int depth = 1;
    double lambda = 10.0;

    trajectory.push_back({x_new, v_new});
    double initial_H = model.Hamiltonian(x, v, data[0]);

    // Моделируем "двойной" взрыв по траектории
    for (int depth_iter = 0; depth_iter < max_depth; ++depth_iter) {
      integrate(x_new, v_new, model, data[0], epsilon_temp, 1);

      trajectory.push_back({x_new, v_new});

      H_new = model.Hamiltonian(x_new, v_new, data[0]);

      alpha = std::min(1.0, exp(initial_H - H_new));

      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<> dis(0.0, 1.0);
      u = dis(gen);

      if (u < alpha) {
        x = x_new; // Обновление состояния
        accepted++;
      }

      if (depth_iter == max_depth - 1) {
        break;
      }

      epsilon_temp *= lambda; // адаптивный шаг
    }

    samples.push_back(x);
  }

  double acceptance_rate =
      static_cast<double>(accepted) / num_samples / max_depth;
  std::cout << "Acceptance rate: " << acceptance_rate * 100.0 << "%"
            << std::endl;
  return samples;
}
int main() {
  int dim = 3;
  int sampleSize = 20000;
  int num_steps = 10000;
  double epsilon = 0.00001;
  double noiseStddev = 0.1;
  double lowBound = -5.0;
  double upperBound = 5.0;
  int batchSize = 20000;

  int max_depth = 10;
  Model model(dim, noiseStddev, lowBound, upperBound, batchSize);

  // Инициализация начального вектора
  std::vector<double> initial_x = std::vector<double>(dim, 0.0);

  // Запуск HMC
  char method_choice;
  std::cout << "Choose the method: HMC (h) or NUTS (n): ";
  std::cin >> method_choice;

  std::vector<std::vector<double>> samples;

  if (method_choice == 'h') {
    std::cout << "Sampling with HMC.\n";

    samples = hmc(model, initial_x, sampleSize, epsilon, num_steps);
  } else if (method_choice == 'n') {
    std::cout << "Sampling with NUTS.\n";

    samples = nuts(model, initial_x, sampleSize, epsilon, max_depth);
  } else {
    std::cout << "Invalid choice, defaulting to HMC.\n";
    samples = hmc(model, initial_x, sampleSize, epsilon, num_steps);
  }

  std::vector<std::vector<double>> filtered_samples =
      filterChainByACF(samples, 0.3, 1000, 100);

  // Вычисление среднего по каждому параметру
  std::vector<double> mean = computeMean(filtered_samples);

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

    // Трассировка параметров
    saveTracePlot(filtered_samples, i);
  }
  for (int i = 0; i < dim; ++i) {
    std::vector<double> param_chain;
    for (const auto &sample : filtered_samples) {
      param_chain.push_back(sample[i]);
    }

    // Вычисление автокорреляции
    std::vector<double> acors = autocorrelation(param_chain, 100);

    // Сохраняем автокорреляцию в файл
    saveAutocorrelationToFile(acors, i);

    // Строим график автокорреляции
    plotAutocorrelation(i);
  }
  return 0;
}
