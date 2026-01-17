@echo on
setlocal

:: Check for cl.exe and setup env if needed
where cl.exe >nul 2>nul
if %errorlevel% neq 0 (
    echo [Build] cl.exe not found. Initializing VS 2022 Environment...
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
)


:: ensure target/release/tracea.dll.lib exists
if not exist "..\..\target\release\tracea.dll.lib" (
    echo Error: tracea.dll.lib not found. Run 'cargo build --lib --release' first.
    exit /b 1
)

:: Compile
echo Building latency_bench...
nvcc -O3 -Xcompiler "/utf-8" -o latency_bench.exe latency_bench.cu -I../../include ../../target/release/tracea.dll.lib
if %errorlevel% neq 0 exit /b %errorlevel%

:: Copy DLL for execution
echo Copying tracea.dll...
copy /Y "..\..\target\release\tracea.dll" .

:: Shim NVRTC for cudarc (expects 11.2 or 12.0, system has 13.0)
copy /Y "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\nvrtc64_130_0.dll" ".\nvrtc64_112_0.dll"
copy /Y "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\nvrtc64_130_0.dll" ".\nvrtc64_120_0.dll"
copy /Y "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\nvrtc-builtins64_131.dll" .


echo Build Success. Run latency_bench.exe
dir
