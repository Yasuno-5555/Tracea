import ctypes
import os

# Load CUDA Driver API
try:
    nvcuda = ctypes.WinDLL("nvcuda.dll")
except:
    print("Could not load nvcuda.dll")
    exit(1)

# Constants
CU_JIT_ERROR_LOG_BUFFER = 1
CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 2
CU_JIT_INFO_LOG_BUFFER = 3
CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4

def check_cuda(res):
    if res != 0:
        err = ctypes.c_char_p()
        nvcuda.cuGetErrorString(res, ctypes.byref(err))
        raise RuntimeError(f"CUDA Error {res}: {err.value.decode()}")

# Initialize
check_cuda(nvcuda.cuInit(0))
device = ctypes.c_int()
check_cuda(nvcuda.cuDeviceGet(ctypes.byref(device), 0))
context = ctypes.c_void_p()
check_cuda(nvcuda.cuCtxCreate(ctypes.byref(context), 0, device))

# Load PTX file
ptx_file = "E:/Projects/Tracea/debug_dump_flash_attention_v2_kernel.ptx"
if not os.path.exists(ptx_file):
    print(f"PTX file not found: {ptx_file}")
    exit(1)

with open(ptx_file, "r") as f:
    ptx_src = f.read()

# Prepare JIT options
log_size = 16384
error_log = ctypes.create_string_buffer(log_size)
info_log = ctypes.create_string_buffer(log_size)

options = (ctypes.c_int * 4)(
    CU_JIT_ERROR_LOG_BUFFER,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
    CU_JIT_INFO_LOG_BUFFER,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES
)

option_values = (ctypes.c_void_p * 4)(
    ctypes.cast(error_log, ctypes.c_void_p).value,
    ctypes.c_void_p(log_size).value,
    ctypes.cast(info_log, ctypes.c_void_p).value,
    ctypes.c_void_p(log_size).value
)

module = ctypes.c_void_p()
print(f"Loading {ptx_file} via cuModuleLoadDataEx...")
try:
    res = nvcuda.cuModuleLoadDataEx(
        ctypes.byref(module),
        ctypes.c_char_p(ptx_src.encode()),
        ctypes.c_uint(4),
        options,
        option_values
    )
    if res != 0:
        print(f"JIT Linker Error Log:\n{error_log.value.decode('utf-8', errors='ignore')}")
        check_cuda(res)
    else:
        print("Successfully loaded module!")
        # Try to get the function
        kernel_name = "flash_attention_v2_kernel"
        function_handle = ctypes.c_void_p()
        res_fn = nvcuda.cuModuleGetFunction(ctypes.byref(function_handle), module, kernel_name.encode())
        if res_fn == 0:
            print(f"Successfully found function: {kernel_name}")
        else:
            print(f"Err: Could not find function {kernel_name} (Res: {res_fn})")
        print(f"JIT Info Log:\n{info_log.value.decode('utf-8', errors='ignore')}")

finally:
    nvcuda.cuCtxDestroy(context)
