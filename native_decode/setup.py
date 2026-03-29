from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "native_decode",
        ["decode_worker.cpp"],
        include_dirs=["/usr/include/x86_64-linux-gnu"],
        libraries=["avformat", "avcodec", "swscale", "avutil"],
        cxx_std=17,
        extra_compile_args=["-O3", "-march=native"],
    ),
]

setup(
    name="native_decode",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
