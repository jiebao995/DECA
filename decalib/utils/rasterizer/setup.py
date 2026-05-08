# To install, run 
# python setup.py build_ext -i
# Ref: https://github.com/pytorch/pytorch/blob/11a40410e755b1fe74efe9eaa635e7ba5712846b/test/cpp_extensions/setup.py#L62

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import shutil

# USE_NINJA = os.getenv('USE_NINJA') == '1'
ccbin = os.environ.get("DECA_CUDA_CC") or shutil.which("gcc-7")
if ccbin:
    os.environ["CC"] = ccbin
    os.environ["CXX"] = ccbin

USE_NINJA = os.getenv('USE_NINJA') == '1'

setup(
    name='standard_rasterize_cuda',
    ext_modules=[
	CUDAExtension('standard_rasterize_cuda', [
        'standard_rasterize_cuda.cpp',
        'standard_rasterize_cuda_kernel.cu',
        ],
        extra_compile_args={'nvcc': ['-std=c++17']})
	],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=USE_NINJA)}
)
