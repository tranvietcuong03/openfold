import os
from setuptools import setup, find_packages
import subprocess

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

from scripts.utils import get_nvidia_cc


version_dependent_macros = [
    '-DVERSION_GE_1_1',
    '-DVERSION_GE_1_3',
    '-DVERSION_GE_1_5',
]

extra_cuda_flags = [
    '-std=c++17',
    '-maxrregcount=50',
    '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__',
    '--expt-relaxed-constexpr',
    '--expt-extended-lambda'
]

def get_cuda_bare_metal_version(cuda_dir):
    if cuda_dir==None or torch.version.cuda==None:
        print("CUDA is not found, cpu version is installed")
        return None, -1, 0
    else:
        raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
        output = raw_output.split()
        release_idx = output.index("release") + 1
        release = output[release_idx].split(".")
        bare_metal_major = release[0]
        bare_metal_minor = release[1][0]
        
        return raw_output, bare_metal_major, bare_metal_minor

compute_capabilities = set([
    (5, 2), # Titan X
    (6, 1), # GeForce 1000-series
    (9, 0), # Hopper
])

compute_capabilities.add((7, 0))
_, bare_metal_major, _ = get_cuda_bare_metal_version(CUDA_HOME)
if int(bare_metal_major) >= 11:
    compute_capabilities.add((8, 0))

compute_capability, _ = get_nvidia_cc()
if compute_capability is not None:
    compute_capabilities = set([compute_capability])

cc_flag = []
for major, minor in list(compute_capabilities):
    cc_flag.extend([
        '-gencode',
        f'arch=compute_{major}{minor},code=sm_{major}{minor}',
    ])

extra_cuda_flags += cc_flag

cc_flag = ['-gencode', 'arch=compute_70,code=sm_70']

if bare_metal_major != -1:
    modules = [CUDAExtension(
        name="attn_core_inplace_cuda",
        sources=[
            "vinafold/utils/kernel/csrc/softmax_cuda.cpp",
            "vinafold/utils/kernel/csrc/softmax_cuda_kernel.cu",
        ],
        include_dirs=[
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'vinafold/utils/kernel/csrc/'
            )
        ],
        extra_compile_args={
            'cxx': ['-O3'] + version_dependent_macros,
            'nvcc': (
                ['-O3', '--use_fast_math'] +
                version_dependent_macros +
                extra_cuda_flags
            ),
        }
    )]
else:
    modules = [CppExtension(
        name="attn_core_inplace_cuda",
        sources=[
            "vinafold/utils/kernel/csrc/softmax_cuda.cpp",
            "vinafold/utils/kernel/csrc/softmax_cuda_stub.cpp",
        ],
        extra_compile_args={
            'cxx': ['-O3'],
        }
    )]

setup(
    name='openfold',
    version='2.2.0',

    packages=find_packages(exclude=["tests", "scripts"]),
    include_package_data=True,
    package_data={
        "vinafold": ['utils/kernel/csrc/*'],
        "": ["resources/stereo_chemical_props.txt"]
    },

    ext_modules=modules,
    cmdclass={'build_ext': BuildExtension},
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.10,'
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
