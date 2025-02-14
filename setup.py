from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

def read_file(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError as e:
        print(f"Error reading file: {e}")
        return ""

requirements_file = path.join(here, 'requirements.txt')
long_description_file = path.join(here, 'README.md')

all_reqs = read_file(requirements_file).split('\n')
long_description = read_file(long_description_file)

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]

setup(
    name='PubBCRPredictor',  
    version='1.0.0',  
    author='Qihang Xu, Jian Zhang',  
    description='PubBCRPredictor, the public antibody prediction module leverages the pre-trained BCR-V-BERT model to classify heavy chain antibodies (binary classification) and predict light chain antibodies (regression).', 
    long_description=long_description, 
    long_description_content_type='text/markdown', 
    author_email='jian_zhang@tju.edu.cn',  
    url='https://github.com/ZhangLabTJU/PubBCRPredictor',  
    license='CC BY-NC-SA 4.0',
    include_package_data=True, 
    packages=find_packages(),  
    install_requires=install_requires, 
    classifiers=[
        'License :: OSI Approved :: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
    python_requires=">=3.9"
)
