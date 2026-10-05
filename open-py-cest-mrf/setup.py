from setuptools import setup, find_packages

setup(
    name='cest_mrf',
    author='Nikita Vladimirov',
    author_email='nikitav@mail.tau.ac.il',
    version='0.4',
    description='Python code to use C++ pulseq-CEST to simulate MRI signal and MRF dictionary generation.',
    install_requires=[
        'bmctool==0.5.0',
        'numpy==1.23.5',
        'scipy==1.11.1',
        'PyYAML==6.0',
        'sigpy==0.1.22',
        'ipykernel==6.29.0',
        'tqdm==4.66.2',
        'h5py==3.10.0',
    ],
    keywords='MRI, Bloch, CEST, simulations',
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.9',
)
