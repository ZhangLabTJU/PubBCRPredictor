from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'requirements.txt'), 'r', encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

with open(path.join(here, 'README.md'), 'r', encoding='utf-8') as f:
    long_description = f.read()

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]

setup(
    name='PubBCRPredictor',  
    version='1.0.0',  
    author='ZJ-lab',  
    description='PubBCRPredictor,the public antibody prediction module leverages the pre-trained **BCR-V-BERT** model to classi-fy heavy chain antibodies (binary classification) and predict light chain antibodies (regression). This feature allows for the identification of public antibodies using advanced transformer-based sequence analysis.',  
    long_description=long_description, 
    long_description_content_type='text/markdown', 
    author_email='jian_zhang@tju.edu.cn',  
    url='https://github.com/ZhangLabTJU/PubBCRPredictor',  
    license='CC BY-NC-SA 4.0',
    include_package_data=True, 
    packages=find_packages(),  
    install_requires=install_requires, 

    classifiers=[
        'License :: OSI Approved :: CC BY-NC-SA 4.0',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
    ],
    python_requires=">=3.9"
)
