#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <tuple>
#include <vector>

static inline std::vector<double> generateRandomVector(int n, double lowerBd,
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
    trueParams = {1.0, 2.0, 3.0};
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
                               const std::vector<double> &y,
                               double h = 1e-6) const {
    const double U0 = U(x, y);
    std::vector<double> g(DIM);
    std::vector<double> x_shift = x;

    for (int j = 0; j < DIM; ++j) {
      x_shift[j] = x[j] + h;
      double U_plus = U(x_shift, y);

      x_shift[j] = x[j] - h;
      double U_minus = U(x_shift, y);

      g[j] = (U_plus - U_minus) / (2.0 * h);

      x_shift[j] = x[j]; // вернуть координату
    }
    return g;
  }
};

static inline std::vector<double>
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

static inline void printVector(const std::vector<double> &vec) {
  for (double v : vec) {
    std::cout << v << " ";
  }
  std::cout << "\n";
}
static inline std::map<double, int>
computeHistogram(const std::vector<double> &data, int num_bins) {
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
static inline void saveHistogramToFile(const std::map<double, int> &histogram,
                                       const std::string &filename,
                                       int dataSize) {
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
static inline void plotHistogram(const std::string &filename) {
  std::string command = "gnuplot -e \"set terminal png; set output '" +
                        filename + ".png'; plot '" + filename +
                        ".txt' using 1:2 with boxes\"";
  system(command.c_str());
}
// Функция для записи трассировки для каждого параметра в отдельный файл
static void saveTracePlot(const std::vector<std::vector<double>> &samples,
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
static inline std::vector<double>
autocorrelation(const std::vector<double> &chain, int max_lag = 100) {
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
static inline void saveAutocorrelationToFile(const std::vector<double> &acors,
                                             int param_idx) {
  std::ofstream outFile("autocorrelation_param_" +
                        std::to_string(param_idx + 1) + ".txt");
  for (int i = 0; i < acors.size(); ++i) {
    outFile << i << " " << acors[i] << "\n";
  }
  outFile.close();
}
static inline void plotAutocorrelation(int param_idx) {
  std::string command =
      "gnuplot -e \"set terminal png; set output 'autocorrelation_param_" +
      std::to_string(param_idx + 1) + ".png'; plot 'autocorrelation_param_" +
      std::to_string(param_idx + 1) +
      ".txt' using 1:2 with lines title 'ACF of x" +
      std::to_string(param_idx + 1) + "\"";
  system(command.c_str());
}
static inline std::vector<std::vector<double>> filterChainByACF(
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

    for (int i = start_index; i < samples.size(); i += skip_rate) {
      filtered_samples.push_back(samples[i]);
    }
  }

  return filtered_samples;
}
static inline void integrate(std::vector<double> &x, std::vector<double> &v,
                             const Model &model,
                             const std::vector<double> &data, double epsilon,
                             int num_steps) {
  std::vector<double> grad, v_temp, x_temp;
  int xSize = static_cast<int>(x.size());
  int vSize = static_cast<int>(v.size());
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

static inline std::vector<std::vector<double>>
hmc(Model &model, const std::vector<double> &initial_x, int num_samples,
    double epsilon, int num_steps) {
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
  std::cout << "Acceptance rate: " << acceptance_rate * 100.0 << "%\n";
  return samples;
}

static inline double dot(const std::vector<double> &a,
                         const std::vector<double> &b) {
  double s = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    s += a[i] * b[i];
  return s;
}

// Leapfrog‑шаг (симплектовый интегратор)
static void leapfrogNuts(const Model &model, std::vector<double> &x,
                         std::vector<double> &r, double eps,
                         const std::vector<double> &data) {
  // half‑step для импульса
  auto g = model.gradient(x, data);
  for (size_t i = 0; i < r.size(); ++i)
    r[i] -= 0.5 * eps * g[i];

  // full‑step для координат
  for (size_t i = 0; i < x.size(); ++i)
    x[i] += eps * r[i];

  // ещё half‑step для импульса
  g = model.gradient(x, data);
  for (size_t i = 0; i < r.size(); ++i)
    r[i] -= 0.5 * eps * g[i];
}

// Критерий разворота (No‑U‑turn)
static bool stopCriterion(const std::vector<double> &x_minus,
                          const std::vector<double> &x_plus,
                          const std::vector<double> &r_minus,
                          const std::vector<double> &r_plus) {
  std::vector<double> dx(x_minus.size());
  for (size_t i = 0; i < dx.size(); ++i)
    dx[i] = x_plus[i] - x_minus[i];
  return (dot(dx, r_minus) < 0.0) || (dot(dx, r_plus) < 0.0);
}

//------------------------------------------------------------------------------
// Построение бинарного дерева (рекурсивно)
//------------------------------------------------------------------------------
// Возвращаемый кортеж:
//   x_minus, r_minus   – край траектории в минус‑направлении
//   x_plus,  r_plus    – край траектории в плюс‑направлении
//   x_candidate        – кандидат в выборку
//   n_valid            – число валидных точек в поддереве
//   keep_sampling      – можно ли продолжать расширять дерево (true/false)
//   sum_alpha          – суммарная alpha (для статистики)
//   num_alpha          – число alpha (для статистики)
//------------------------------------------------------------------------------
static std::tuple<std::vector<double>, std::vector<double>, std::vector<double>,
                  std::vector<double>, std::vector<double>, size_t, bool,
                  double, size_t>
buildTree(const Model &model, const std::vector<double> &x,
          const std::vector<double> &r, double log_u, int direction, int depth,
          double eps, const std::vector<double> &data, std::mt19937 &gen) {
  if (depth == 0) {
    // Базовый случай: один leapfrog‑шаг
    std::vector<double> x1 = x;
    std::vector<double> r1 = r;
    leapfrogNuts(model, x1, r1, direction * eps, data);

    double H1 = model.Hamiltonian(x1, r1, data);
    bool valid = (log_u < -H1);
    bool diverge = (H1 + log_u) > 1000.0; // порог дивергенции
    size_t n_valid = (valid && !diverge) ? 1 : 0;

    double alpha = std::min(1.0, std::exp(model.Hamiltonian(x, r, data) - H1));
    return {x1, r1, x1, r1, valid ? x1 : x, n_valid, !diverge, alpha, 1};
  }

  // Рекурсивно строим левое поддерево
  std::vector<double> x_minus, r_minus, x_plus, r_plus, x_candidate;
  size_t n_valid = 0;
  bool keep = false;
  double sum_alpha = 0.0;
  size_t num_alpha = 0;

  std::tie(x_minus, r_minus, x_plus, r_plus, x_candidate, n_valid, keep,
           sum_alpha, num_alpha) =
      buildTree(model, x, r, log_u, direction, depth - 1, eps, data, gen);

  if (keep) {
    // Строим правое поддерево
    std::vector<double> x_m2, r_m2, x_p2, r_p2, x_prop2;
    size_t n_valid2;
    bool keep2;
    double sum_alpha2;
    size_t num_alpha2;

    if (direction == -1) {
      std::tie(x_m2, r_m2, x_p2, r_p2, x_prop2, n_valid2, keep2, sum_alpha2,
               num_alpha2) = buildTree(model, x_minus, r_minus, log_u,
                                       direction, depth - 1, eps, data, gen);
      x_minus = x_m2;
      r_minus = r_m2;
    } else {
      std::tie(x_m2, r_m2, x_p2, r_p2, x_prop2, n_valid2, keep2, sum_alpha2,
               num_alpha2) = buildTree(model, x_plus, r_plus, log_u, direction,
                                       depth - 1, eps, data, gen);
      x_plus = x_p2;
      r_plus = r_p2;
    }

    //  Выбираем предложение пропорционально размерам поддеревьев
    std::uniform_real_distribution<> dis(0.0, 1.0);
    if ((n_valid + n_valid2) > 0 &&
        dis(gen) < static_cast<double>(n_valid2) /
                       static_cast<double>(n_valid + n_valid2)) {
      x_candidate = x_prop2;
    }

    n_valid += n_valid2;
    keep = keep2 && !stopCriterion(x_minus, x_plus, r_minus, r_plus);
    sum_alpha += sum_alpha2;
    num_alpha += num_alpha2;
  }

  return {x_minus, r_minus, x_plus,    r_plus,   x_candidate,
          n_valid, keep,    sum_alpha, num_alpha};
}

static inline std::vector<std::vector<double>>
nuts(Model &model, const std::vector<double> &initial_x, int num_samples,
     double eps, int max_depth = 10) {
  std::vector<double> x = initial_x;
  std::vector<std::vector<double>> samples;

  auto data_mat = model.getData();
  const std::vector<double> &data = data_mat[0];

  std::random_device rd;
  std::mt19937 gen(rd());

  double alpha_sum_total = 0.0;
  size_t alpha_cnt_total = 0;

  for (int n = 0; n < num_samples; ++n) {
    if (n % (num_samples / 10) == 0) {
      std::cout << "Progress: " << (n * 100) / num_samples << "%\n";
    }

    // Случайный импульс
    std::vector<double> r0 = model.generateRandomMomentum();
    double H0 = model.Hamiltonian(x, r0, data);

    std::uniform_real_distribution<> dis_u(0.0, 1.0);
    double log_u = std::log(dis_u(gen)) - H0;

    // Инициализируем дерево
    std::vector<double> x_minus = x, x_plus = x;
    std::vector<double> r_minus = r0, r_plus = r0;
    std::vector<double> x_prop = x;
    size_t n_valid = 1;
    bool keep_sampling = true;

    for (int depth = 0; depth < max_depth && keep_sampling; ++depth) {
      int direction = (dis_u(gen) < 0.5) ? -1 : 1;

      std::vector<double> x_m2, r_m2, x_p2, r_p2, x_prop2;
      size_t n_valid2;
      bool keep2;
      double alpha_sum;
      size_t num_alpha;

      if (direction == -1) {
        std::tie(x_m2, r_m2, x_p2, r_p2, x_prop2, n_valid2, keep2, alpha_sum,
                 num_alpha) = buildTree(model, x_minus, r_minus, log_u,
                                        direction, depth, eps, data, gen);
        x_minus = x_m2;
        r_minus = r_m2;
      } else {
        std::tie(x_m2, r_m2, x_p2, r_p2, x_prop2, n_valid2, keep2, alpha_sum,
                 num_alpha) = buildTree(model, x_plus, r_plus, log_u, direction,
                                        depth, eps, data, gen);
        x_plus = x_p2;
        r_plus = r_p2;
      }

      // Обновляем глобальную alpha‑статистику
      alpha_sum_total += alpha_sum;
      alpha_cnt_total += num_alpha;

      if (keep2 == false)
        break; // divergence detected

      // Случайный выбор предложения из объединённого поддерева
      std::uniform_real_distribution<> dis(0.0, 1.0);
      if ((n_valid + n_valid2) > 0 &&
          dis(gen) < static_cast<double>(n_valid2) /
                         static_cast<double>(n_valid + n_valid2)) {
        x_prop = x_prop2;
      }

      n_valid += n_valid2;
      keep_sampling = !stopCriterion(x_minus, x_plus, r_minus, r_plus);
    }

    x = x_prop;
    samples.push_back(x);
  }

  std::cout << "Mean accept prob alpha: "
            << (alpha_cnt_total
                    ? alpha_sum_total / static_cast<double>(alpha_cnt_total)
                    : 0.0)
            << "\n";

  return samples;
}

int main() {
  int dim = 3;
  int sampleSize = 20000;
  int L = 25;
  double epsilon = 0.001;
  double noiseStddev = 0.1;
  double lowBound = -5.0;
  double upperBound = 5.0;
  int batchSize = 20000;

  int max_depth = 5;
  Model model(dim, noiseStddev, lowBound, upperBound, batchSize);

  // Инициализация начального вектора
  std::vector<double> initial_x = std::vector<double>(dim, 0.0);

  // Запуск HMC
  char method_choice;
  std::cout << "Choose the method: HMC (h) or NUTS (n): ";
  std::cin >> method_choice;

  std::vector<std::vector<double>> samples;

  auto start = std::chrono::high_resolution_clock::now();

  if (method_choice == 'h') {
    std::cout << "Sampling with HMC.\n";

    samples = hmc(model, initial_x, sampleSize, epsilon, L);
  } else if (method_choice == 'n') {
    std::cout << "Sampling with NUTS.\n";

    samples = nuts(model, initial_x, sampleSize, epsilon, max_depth);
  } else {
    std::cout << "Invalid choice, defaulting to HMC.\n";
    samples = hmc(model, initial_x, sampleSize, epsilon, L);
  }
  auto end = std::chrono::high_resolution_clock::now();

  // Вычисляем разницу времени
  std::chrono::duration<double> duration = end - start;
  std::cout << "Time taken: " << duration.count() << " seconds\n";

  std::vector<std::vector<double>> filtered_samples =
      filterChainByACF(samples, 0.1, 1000, 100);

  // Вычисление среднего по каждому параметру
  std::vector<double> mean = computeMean(filtered_samples);
  size_t filtered_samples_size = filtered_samples.size();
  std::cout << "\nMean of each parameter after " << filtered_samples_size
            << " samples:\n";
  printVector(mean);

  std::cout << "\nTrue parameters\n ";
  std::vector<double> trueParams = model.getTrue();
  printVector(trueParams);

  // Получаем сгенерированные данные
  auto inputData = model.getData();
  int num_bins = 100; // Количество бинов для гистограммы

  // Генерация и сохранение гистограмм для каждого столбца
  for (int i = 0; i < inputData[0].size(); ++i) {
    std::vector<double> column_data_input;
    std::vector<double> column_data_output;
    for (const auto &row : inputData) {
      column_data_input.push_back(row[i]);
    }
    for (const auto &row : filtered_samples) {
      column_data_output.push_back(row[i]);
    }
    // Вычисление гистограммы
    auto histogramInput = computeHistogram(column_data_input, num_bins);
    auto histogramOutput = computeHistogram(column_data_output, num_bins);

    // Сохраняем гистограмму для каждого столбца
    std::string filenameInput = "inputData_x" + std::to_string(i + 1);
    std::string filenameOutput = "posterior_x" + std::to_string(i + 1);

    saveHistogramToFile(histogramInput, filenameInput, sampleSize);
    saveHistogramToFile(histogramOutput, filenameOutput,
                        filtered_samples.size());

    // Строим гистограмму с использованием gnuplot
    plotHistogram(filenameInput);
    plotHistogram(filenameOutput);

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
