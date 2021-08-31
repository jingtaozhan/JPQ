from setuptools import setup

with open('README.md') as f:
    readme = f.read()

setup(
   name='JPQ',
   version='0.1.0',
   description='Jointly Optimizing Query Encoder and Product Quantization to Improve Retrieval Performance',
   url='https://github.com/jingtaozhan/JPQ',
   classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
   license="MIT License",
   long_description=readme,
   install_requires=[
        'torch==1.9.0', 
        'transformers==4.3.3', 
        'faiss-gpu==1.7.1',
        'tensorboard==2.5.0'
    ],
)