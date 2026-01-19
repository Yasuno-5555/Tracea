
import os
import glob

cuda_path = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
print(f"CUDA_PATH: {cuda_path}")

if os.path.exists(cuda_path):
    # Try to find bin
    bin_path = os.path.join(cuda_path, "bin")
    if os.path.exists(bin_path):
        print(f"Found bin: {bin_path}")
        dlls = glob.glob(os.path.join(bin_path, "nvrtc64*.dll"))
        print(f"NVRTC DLLs: {dlls}")
    else:
        # Maybe version folder?
        print("Looking in subfolders...")
        for root, dirs, files in os.walk(cuda_path):
             if "bin" in dirs:
                 b = os.path.join(root, "bin")
                 dlls = glob.glob(os.path.join(b, "nvrtc64*.dll"))
                 if dlls:
                     print(f"Found bin with nvrtc: {b}")
                     print(f"DLLs: {dlls}")
                     break
else:
    print("CUDA_PATH does not exist")
