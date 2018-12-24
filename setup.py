import setuptools

with open('README.md', 'r') as d:
    long_description = d.read()

setuptools.setup(name='GaiaLab',
                 version='0.2.0',
                 author='Alex Bombrun, Maria del Valle Varo, Toby James, Luca Zampieri',
                 author_email='abombrun@sciops.esa.int',
                 description='Toy model for the Gaia scanning law and '
                             'micrometeoroid detection and simulation '
                             'functions for Gaia data.',
                 long_description=long_description,
                 long_description_content_type='text/markdown',
                 url='https://github.com/bombrun/GaiaLab',
                 packages=setuptools.find_packages(),
                 classifiers=["Programming Language :: Python :: 3",
                              "License :: OSI Approved :: MIT License",
                              "Operating System :: OS Independent"])
