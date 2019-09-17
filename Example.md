# GaiaLab
Software used to generate reports, taking data from a running SonarQube instance, for the ESAC PA/QA group

## Project layout
The project has different files and directories
```
    setup.py        # The configuration file
    README.md       #
    gaialab/
        constants.py
        frame_transformations
        helpers.py
        source.py    # The documentation homepage
            # Brief introduction to the package
        metrics.md  # Module metrics.py and its functions
        tables.md   # Module tables.py and its functions
    environent.yml  # Package list and channels needed to create the virtual environment
    resources/      # Files needed to establish connection with SonarQube (config and security)
    test/           # Directory in which unitary tests are allocated
    tools/          # Directory containing all the library modules
        metrics.py  # Module containing functions to get metrics from SonarQube
        tables.py   # Module containing functions to work with the metrics extracted from SonarQube
```


## Getting started
### Prerequisites
In order to make it easier to use it and to not risk the machine python installation, it is strongly recommended to create a virtual environment with all the packages specified in the 'environment.yml' file, so that to copy the characteristics of the machine in which everything has been successfully tested.
If using Conda to manage virtual environments, create it by going to the project root directory and typing in the command line
```
conda env create --name <virtual environent name> --file environment.yml
```
**Important:**
It is also necessary to customize the 'sonar.security.example' and 'sonar.config.example' with your own credentials and configuration. Once you have done it, delete '.example' from the name.

## Automatic documentation build
It is possible to create an static html page to show the documentation in a user-friendly way.
In order to build the html documentation just go to the project root in the command line and type
```
mkdocs build
```
Once the build is finished, a new folder called 'site' will appear in the project root containing the html guide. In order to open it, just open 'index.html' file with a browser.

## Documentation
All the .md files are allocated in the 'docs' folder, although is recommended to build the documentation and consult it from the html files.

## Use
To run any of the functions from the module it is necessary to import it in the python console or in the script you want it to use the functions, then it is possible to use them.

## Test
In order to get the tests run for a specific module and to get the coverage, run from the project root in the command line
```
pytest --cov <module to test>
```
To run the complete set of tests for all the modules and get the coverage, run from the project root in the command line
```
pytest --cov
```
**Important:**
Although it could see there is usage of some files in testing, there is not, everything is mocked up.

## Authors
* **Luis Crespo** - *Initial work*
