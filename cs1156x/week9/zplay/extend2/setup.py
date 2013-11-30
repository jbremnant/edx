from distutils.core import setup, Extension
import numpy.distutils.misc_util

c_ext = Extension("_chi2", ["_chi2.c", "chi2.c"])

setup(
    ext_modules=[c_ext],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)
