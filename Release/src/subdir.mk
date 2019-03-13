################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/GetFeatPoint.cpp \
../src/GlobalMotion.cpp \
../src/MaxVarFeatTrk.cpp \
../src/MaxVarRegionTrk.cpp \
../src/MedianFlowTrk.cpp \
../src/RejectOutliner.cpp \
../src/SceneMVCal.cpp \
../src/SceneOptFlow.cpp \
../src/augmented_unscented_kalman.cpp \
../src/feature.cpp \
../src/sceneProc.cpp \
../src/unscented_kalman.cpp 

OBJS += \
./src/GetFeatPoint.o \
./src/GlobalMotion.o \
./src/MaxVarFeatTrk.o \
./src/MaxVarRegionTrk.o \
./src/MedianFlowTrk.o \
./src/RejectOutliner.o \
./src/SceneMVCal.o \
./src/SceneOptFlow.o \
./src/augmented_unscented_kalman.o \
./src/feature.o \
./src/sceneProc.o \
./src/unscented_kalman.o 

CPP_DEPS += \
./src/GetFeatPoint.d \
./src/GlobalMotion.d \
./src/MaxVarFeatTrk.d \
./src/MaxVarRegionTrk.d \
./src/MedianFlowTrk.d \
./src/RejectOutliner.d \
./src/SceneMVCal.d \
./src/SceneOptFlow.d \
./src/augmented_unscented_kalman.d \
./src/feature.d \
./src/sceneProc.d \
./src/unscented_kalman.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I/usr/include/opencv -I../inc -O3 -Xcompiler -fPIC -Xcompiler -fopenmp -ccbin aarch64-linux-gnu-g++ -gencode arch=compute_50,code=sm_50 -m64 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I/usr/include/opencv -I../inc -O3 -Xcompiler -fPIC -Xcompiler -fopenmp --compile -m64 -ccbin aarch64-linux-gnu-g++  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


