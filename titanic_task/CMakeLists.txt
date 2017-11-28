cmake_minimum_required(VERSION 3.8)
project(titanic_task)

set(CMAKE_CXX_COMPILER clang++)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS -O3)

set(NN_LIBRARY "../neural_network_lib/neural_network.cpp")
set(NN_MODEL_LIBRARY "../neural_network_lib/NeuralModel.cpp")

set(NN_MODEL_HEADER "../neural_network_lib/NeuralModel.cpp")
set(NN_HEADER "../neural_network_lib/neural_network.hpp")

set(FUNC_HEADER "../FuncAnalyze/functional_lib.h")
set(FUNC_LIBRARY "../FuncAnalyze/functional_lib.cpp")

set(RECNN_HEADER "../ReconfigurableNN/reconfigurable_nn.hpp")
set(RECNN_LIBRARY "../ReconfigurableNN/reconfigurable_nn.cpp")

set(SOURCE_FILES main.cpp ${RECNN_HEADER} ${RECNN_LIBRARY} ${NN_LIBRARY} ${FUNC_HEADER} ${FUNC_LIBRARY})
add_executable(titanic_task ${NN_LIBRARY} ${SOURCE_FILES} ${RECNN_HEADER} ${RECNN_LIBRARY} ${NN_MODEL_HEADER} ${NN_HEADER} ${FUNC_HEADER} ${FUNC_LIBRARY})