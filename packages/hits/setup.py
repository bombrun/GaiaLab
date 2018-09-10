import setuptools

with open('README.md', 'r') as d:
    long_description = d.read()

setuptools.setup(name='gaia-hits',
                 version='0.1.0',
                 author='Toby James',
                 author_email='tobyswjames@gmail.com',
                 description='Micrometeoroid detection and simulation for Gaia'
                             ' data.',
                 long_description=long_description,
                 long_description_content_type='text/markdown',
                 url='https://github.com/bombrun/GaiaLab/tree/master/'
                     'packages/hits',
                 packages=setuptools.find_packages(),
                 classifiers=["Programming Language :: Python :: 3",
                              "License :: OSI Approved :: MIT License",
                              "Operating System :: OS Independent"])
