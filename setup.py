from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("cy_code.bipoo_one_synapse_gaussian_ic", ["cy_code/bipoo_one_synapse_gaussian_ic.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension("cy_code.integrate_langevin_interp", ["cy_code/integrate_langevin_interp.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension("cy_code.bipoo_ff_fourpop", ["cy_code/bipoo_ff_fourpop.pyx"],
              include_dirs=[numpy.get_include()]),
    Extension("cy_code.bipoo_ff_into_brunel_training", ["cy_code/bipoo_ff_into_brunel_training.pyx"],
              include_dirs=[numpy.get_include()]),
]

setup(
    name="synaptic-dynamics-code",
    version="1.0.0",
    packages=['cy_code'],
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'cython',
        'multiprocess'
    ],
    python_requires='>=3.7',
)
