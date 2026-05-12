@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
cd /d "%~dp0"
echo Building in: %CD% > build_log.txt 2>&1
echo Compiling... >> build_log.txt 2>&1
REM Release build: no -DDEBUG, no -DTRACE.  Add -DTRACE for enumeration trace.
nvcc -arch=sm_86 -O3 -ftz=false -prec-div=true -prec-sqrt=true -fmad=false -lineinfo cascade_kernel.cu cascade_host.cu -o cascade_prover.exe >> build_log.txt 2>&1
echo Exit code: %ERRORLEVEL% >> build_log.txt 2>&1
if exist cascade_prover.exe (
    echo BUILD SUCCESS >> build_log.txt
) else (
    echo BUILD FAILED >> build_log.txt
)
