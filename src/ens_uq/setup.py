from setuptools import setup, find_packages

setup(name='appuqa',
      version='0.1.0',
      description="A Parallel Platform for Uncertain Quantification Algorithms",
      author="Yimin Liu",
      author_email="yiminliu@stanford.edu",
      license="GPL",
      zip_safe=True,
      packages=find_packages(),
      # install_requires=[
      #     'mpi4py>=2.0.0',
      #     'multimethod>=0.6',
      #     'numpy>=1.11.1'
      # ]
)