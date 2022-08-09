from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

def license():
    with open('LICENSE') as f:
        return f.read()

setup(
    name='inspire-experiments',
    version='0.1',
    packages=find_packages(),
    url='https://github.com/codecrap/inspire-experiments',
    license='GNU General Public License v3.0',
    author='codecrap',
    author_email='oolexiy@pm.me',
    description='Code and notebooks for running experiments on'
                'QuTech Quantum Inspire https://www.quantum-inspire.com/',
    long_description=readme(),
    install_requires=list(open('requirements.txt')
                            .read()
                            .strip()
                            .split('\n')),
)



