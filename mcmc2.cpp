#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

// Генерация случайного вектора с параметрами в пределах [lowBound,
// upperBound]
std::vector<double> generateRandomVector(double n, double lowerBd,
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
  std::vector<double> data;
  double lowBound;
  double upperBound;
  double noiseStddev;             // Стандартное отклонение шума
  std::vector<double> trueParams; // Истинные параметры (не будут выводиться)

  // Генерация данных с добавлением шума
  std::vector<double> generateData() const {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0.0, noiseStddev); // Нормальный шум

    std::vector<double> x;
    double function_value = function(trueParams);
    for (int i = 0; i < BATCH_SIZE; ++i) { // 2000 измерений
      x.push_back(function_value + noise(gen));
    }
    return x;
  }

  // Функция модели, которую будем использовать для генерации данных
  double function(const std::vector<double> &x) const {
    // Линейная функция вида f(x) = x[0] + x[1] * x[2] + ... (для простоты)
    double result = 0.0;
    for (int i = 0; i < DIM; ++i) {
      result += x[i];
    }
    return result;
  }

public:
  int DIM;        // Размерность модели
  int BATCH_SIZE; // Размерность данных

  Model(int dim, int batchSize, double noiseStddev, double lowBd,
        double upperBd)
      : DIM(dim), BATCH_SIZE(batchSize), noiseStddev(noiseStddev),
        lowBound(lowBd), upperBound(upperBd) {
    trueParams = generateRandomVector(DIM, lowBound, upperBound);
  }

  std::vector<double> getData() {
    if (data.empty() || data.size() != BATCH_SIZE) {
      data = generateData();
    }
    return data;
  }

  std::vector<double> generateInitialGuess() const {
    return generateRandomVector(DIM, lowBound, upperBound);
  }

  std::vector<double> getTrueParams() const { return trueParams; }
};
void printVector(const std::vector<double> &vec, int upBound = -1) {

  if (upBound == -1) {
    upBound = static_cast<int>(vec.size());
  }
  for (int i = 0; i < upBound; i++) {
    std::cout << vec[i] << " ";
  }

  std::cout << "\n";
}
int main() {
  // Параметры модели
  int dim = 3;              // Размерность модели
  int batchSize = 2000;     // Размерность модели
  double noiseStddev = 0.1; // Стандартное отклонение шума
  double up = 5.0;          // Верхняя граница параметров
  double down = -5.0;       // Нижняя граница параметров

  // Создание объекта модели
  Model model(dim, batchSize, noiseStddev, down, up);

  // Генерация данных
  std::vector<double> data = model.getData();

  // Генерация начальных значений для параметров
  std::vector<double> initialGuess = model.generateInitialGuess();
  std::vector<double> trueParams = model.getTrueParams();

  // Вывод для проверки (не показываем истинные параметры!)
  std::cout << "Generated data (first 10 values): ";
  printVector(data, 10);

  std::cout << "True Params (hidden from output): ";
  printVector(trueParams);

  std::cout << "Initial guess for parameters: ";
  printVector(initialGuess);

  return 0;
}
