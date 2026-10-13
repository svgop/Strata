@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
"C:\Program Files\CMake\bin\cmake.EXE" -G Ninja -DCMAKE_MAKE_PROGRAM=C:\Users\Savage\AppData\Local\Programs\Python\Python312\Scripts\ninja.EXE -S E:\LLM\Engine\strata-custom-2 -B E:\LLM\Engine\strata-custom-2\build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=120 "-DCMAKE_CUDA_COMPILER=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\nvcc.EXE" -DSTRATA_GGML_DIR=E:\LLM\Engine\strata-custom-2\third_party\llama.cpp || exit /b 1
"C:\Program Files\CMake\bin\cmake.EXE" --build E:\LLM\Engine\strata-custom-2\build --target strata -j 16 && exit /b 0
echo   (the build stopped - trying it once more)
"C:\Program Files\CMake\bin\cmake.EXE" --build E:\LLM\Engine\strata-custom-2\build --target strata -j 16 || exit /b 1
