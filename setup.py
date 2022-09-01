from setuptools import setup, find_packages

def get_readme():
    with open('README.md', encoding='utf-8') as f:
        return f.read()

def license():
    with open('LICENSE', encoding='utf-8') as f:
        return f.read()

setup(
    name='inspire_experiments',
    version='0.1',
    author='codecrap',
    author_email='oolexiy@pm.me',
    description='Code and notebooks for running experiments on'
                'QuTech Quantum Inspire https://www.quantum-inspire.com/',
    long_description=get_readme(),
    url='https://github.com/codecrap/inspire-experiments',
    download_url='https://github.com/codecrap/inspire-experiments.git',
    license='GNU General Public License v3.0',
    packages=find_packages('.'),
    python_requires='>=3.8',
    install_requires=open('requirements.txt').read().strip().split('\n'),
    zip_safe=False,
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Physicists',
        'Intended Audience :: Quantum Engineers',
        'Intended Audience :: Experimentalists',
        'Environment :: Jupyter Notebook',
        'Natural Language :: English',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3.0',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        # 'Programming Language :: Python :: 2',
        # 'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.2',
        # 'Programming Language :: Python :: 3.3',
        # 'Programming Language :: Python :: 3.4',
        # 'Programming Language :: Python :: 3.5',
        # 'Programming Language :: Python :: 3.6',
        # 'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='quantum computing quantumcomputing quantum-computing qubit qubits inspire quantuminspire qiskit'
             'physics experiments experiment entanglement measurement measurements dephasing'.split(),
    data_files=[('', ['LICENSE'])],
)



