# Reconfigurable Neural Network
Реализация реконфигурируемой нейронной сети и метода анализа выходов нейронов
---
### Краткое описание библиотек
---
    /neural_network_lib/neural_network.hpp
Содержит реализацию нейронной сети с обучением, основанном на алгоритме обратного распространения ошибки и метод анализа выходов нейронов. В коде лучше и удобнее будет использовать следующую по списку библиотеку. Из данной библиотеки могут понадобиться следующие функции и константы:
```c++
//Изменение коэффициента при экспоненте в сигмоидальной функции активации
//По умолчанию 1
void set_function_coefficient(double);

//Изменение коэффициента для градиентного спуска
//По умолчанию 0.5
void set_delta(double);

//Константы, при которых функция активации стремится к своему максимуму и минимуму
//В данном случае f(x) in 0..1
const double TRUE_VALUE = 1000.0;
const double FALSE_VALUE = -1000.0;
```
---
     /neural_network_lib/NeuralModel.hpp
Содержит более удобный в использовании класс `NeuralModel`.
##### Пример использования конструктора класса `NeuralModel`
```c++
/*
    Создаётся нейронная сеть с двумя скрытыми стоями по 25 и 12 нейронов,
    принимающая на входе вектор из 12 значений и возвращающая
    вектор из двух значений
    (последний элемент вектора структуры НС характеризует выходной слой)
*/
NeuralModel Model(12, {25, 12, 2});

//Создаётся нейронная сеть с одним выходым слоем, принимающая один входной элемент
NeuralModel Model2();
```
##### Обучение `NeuralModel`
```c++
//Заполнение обучающей выбоки, размер каждого вектора должен совпадать с размерностью входа НС
vector<vector<double> > train_X(N);
vector<vector<double> > train_Y(N);

for (auto& i : train_X)
	i.assign(12, ...);	
	
for (auto& i : train_Y)
	i.assign(2, ...);

//Процесс обучения
for (auto i = 0; i < N; ++i) {
	int err = Model.educate(train_X[i], train_Y[i]);
	if (err) {
		//Обработка ошибки
		//В случае успешного завершения все методы возвращают 0 (EXIT_SUCCESS)
		.....
	}
}
```
##### Работа `NeuralModel`
```c++
vector<double> input(12);
vector<double> output(2);

int err = Model.educate(input, output);
```
Стоит отметить, что в классах `NeuralNetwork` и `NeuralModel` есть методы, возвращающие код последней ошибки и строку типа `std::string` с её кратким описанием.
```c++
int err = Model.get_last_error();
string err_string = Model.get_last_error_str();
```
Для класса `NeuralModel` перегружены операции записи и чтения с потока.
```c++
//Запись в файл
ofstream file("neualnetwork.nn");
file << Model;
file.close();

//Чтение с файла
ifstream nn_file("neualnetwork.nn");
nn_file >> Model;
file.close();
```
##### Пример использования метода анализа нейронной сети `NeuralModel::analysis_layers_output`
```c++
vector<vector<pair<double, double> > > analysis_res = Model.analysis_layers_output(nn_inputs);

for (auto i = 0; i < analysis_res.size(); ++i){
    for (auto j = 0; j < analysis_res[i].size(); ++j)
        cout << "[" << analysis_res[i][j].first << ", " << analysis_res[i][j].second << "] ";
    cout << endl;
}
```
---
	/ReconfigurableNN/NeuralModel.hpp
Содержит реализацию самореконфигурируемой нейронной сети. Класс `ReconfigurableNeuralNetwork` наследуется от класса `NeuralModel`, поэтому методы обучения и работы остаются теми же.
Конструктору данной нейронной сети требуется передать лишь матрицу входных данных для обучения с ответами:
```c++
ReconfigurableNeuralNetwork RNN(train_X, train_Y);
```
После чего вызвать метод "тренировки" сети `ReconfigurableNeuralNetwork::fit`, принимающий функцию, определяющую структуру сети и желаемую ошибку сети на тестовых данных. В случае, если данная ошибка не была достигнута, нейронная сеть вернётся в "наилучшее" состояние (в состояние с наименьшей ошибкой). Так же на каждом шаге в поток `std::cout` будет выводиться изменения структуры сети, текущая ошибка и минимальная ошибка, которую удалось достичь.
```c++
int err = RNN.fit(DEFAULT_TRAPEZE<5>, 0.15);
```
На данный момент в библиотеке есть заготовленная функция структуры сети `DEFAULT_TRAPEZE<unsigned int coeff>`, описывающая трапецию, где `coeff` - коэффициент, определяющий во сколько раз первый скрытый слой будет больше размерности входных данных.

---
	/FuncAnalyze/functional_lib.h
Данная библиотека содержит функции для нахождения минимума и максимума функций в заданной области.
Основными функциями являются функция `funclib::grad_min_max_of_func`, принимающая ламбда-выражение (исследуемую функцию), шаг по аргументам и коэффициент для градиентного спуска, а так же количество точек, откуда будет начинать ход град. спуск (позволяет анализировать асцилирующие функции) и `funclib::grad_min_max_of_multidimensional_func`, отличается от первой, только тем, что исследует функцию действующую в многомерном пространстве.
##### Пример нахождения минимума и максимума двумерной функции в заданной области
```c++
auto res = funclib::grad_min_max_of_multidimensional_func<double>({ { -0.3, 1 },{ -0.3, 1 } },
	[&](const vector<double>& v) {
		double x = *v.begin(), y = *v.rbegin();
		return vector<double>({ cos(sin(x))*x, exp(cos(y)) });
	},
		1E-2, 1E-5, 10);

for (auto& i : res)
	cout << i.first << " " << i.second << endl;
```
##### Пример нахождения минимимума и максимума `cos(x)` на отрезке `[0..Pi]`
```c++
pair<double, double> res = funclib::grad_min_max_of_func<double>({ { 0, PI } },
	[](const vector<double>& v) { return cos(*v.begin()); },
	1E-2, 1E-5, 10);
```
---
## Примечания
* Данные библиотеки требуют компиляции под стандарт `C++17`, т.к. в них были использованны *inline*-переменные
* Пример `Cmake` файла для сборки библеотеки:
```cmake
cmake_minimum_required(VERSION 3.8)
project(your_project)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS -O3)

set(NN_LIBRARY "../neural_network_lib/neural_network.cpp")
set(NN_MODEL_LIBRARY "../neural_network_lib/NeuralModel.cpp")

set(NN_MODEL_HEADER "../neural_network_lib/NeuralModel.cpp")
set(NN_HEADER "../neural_network_lib/neural_network.hpp")

set(FUNC_HEADER "../FindMinMaxFunc/functional_lib.h")
set(FUNC_LIBRARY "../FindMinMaxFunc/functional_lib.cpp")

set(RECNN_HEADER "../ReconfigurableNN/reconfigurable_nn.hpp")
set(RECNN_LIBRARY "../ReconfigurableNN/reconfigurable_nn.cpp")

set(HEADERS ${FUNC_HEADER} ${NN_HEADER} ${NN_MODEL_LIBRARY} ${RECNN_HEADER})
set(SOURCE_LIBS ${FUNC_LIBRARY} ${NN_LIBRARY} ${NN_MODEL_LIBRARY} ${RECNN_LIBRARY})

set(SOURCE_FILES ${SOURCE_LIBS} your_main.cpp)
add_executable(your_project ${SOURCE_FILES} ${HEADERS})
```
