import os

from torch.utils.cpp_extension import load
# from torch.version import cuda as torch_cuda_version

# os.environ.pop('CUDA_PATH', None)
# os.environ.pop('CUDA_PATH_V12_4', None)

# os.environ['CUDA_HOME'] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8"
# os.environ['NVCC'] = r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin\\nvcc.exe"


_src_path = os.path.dirname(os.path.abspath(__file__))
_backend = load(
    name="_pvcnn_backend",
    verbose=True,
    extra_cflags=["/O2", "/std:c++17"],  # Optimizations for MSVC
    # extra_cuda_cflags=["--compiler-bindir=C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Tools\\MSVC\\14.34.31933\\bin\\Hostx64\\x64"],
    sources=[
        os.path.join(_src_path, "src", f)
        for f in [
            "ball_query\\ball_query.cpp",
            "ball_query\\ball_query.cu",
            "grouping\\grouping.cpp",
            "grouping\\grouping.cu",
            "interpolate\\neighbor_interpolate.cpp",
            "interpolate\\neighbor_interpolate.cu",
            "interpolate\\trilinear_devox.cpp",
            "interpolate\\trilinear_devox.cu",
            "sampling\\sampling.cpp",
            "sampling\\sampling.cu",
            "voxelization\\vox.cpp",
            "voxelization\\vox.cu",
            "bindings.cpp",
        ]
    ],
)


__all__ = ["_backend"]
