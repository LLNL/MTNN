About
==============================================
# Purpose 
MTNN is a PyTorch framework to develop and test the application of multigrid algorithms to various neural network architectures.

## Project Layout

* **Source** is in `MTNN/`
* **Documentation** is in `MTNN/docs`
* **Dependencies** for environment set-up is in `MTNN/environment`
* **Configuration files** to run MTNN are in `MTNN/config`
  
* **Test hyper-parameters and directory path for outputs** are in `MTNN/mtnnconstants.py`
* **Unit tests** are in `MTNN/tests`
    * Positive test cases (where tests should *not* fail) are in `MTNN/tests/positives`
    * Negative test cases (where tests should fail) are in `MTNN/tests/negative`
* **Example scripts** using MTNN modules & framework are in `MTNN/scripts/examples/`

# Workflow 
* To run a script with a YAML configuration file: 
```bash
python run.py <relative path to script> <relative path to YAML configuration file>
```

# Developer
## Environment Set-up
* How to set-up environment and package dependencies on LC machines.
* The following was tested on the folRlowing LC systems: Pascal. 

Additional Confluence documentation on [how to set-up PyTorch on LC](https://lc.llnl.gov/confluence/display/LC/PyTorch+in+LC)

### Recommended: With Conda 
**The following assumes Bash shell and that Anaconda is already installed on the system.** 
* To-do: Make this into a shell script/makefile

Pre-steps: Install and Update your Conda repository

```bash
conda update --all
```

1. Log-in to an LC machine and HTTPS Git clone this repository. You'll be prompted to authenticate with your Active Directory Official Username (OUN) and RSA-OTP (One Time Password).

    ```bash
    git clone https://lc.llnl.gov/bitbucket/scm/mtnn/mtnnpython.git
    ```

    Checkout this branch: dev/mao6

    ```bash
    cd mtnnpython/
    git checkout dev/mao6
    ```

2. Create a Conda environment with  Python 3.7.4 and pip pre-installed.  In this example, `mtnnenv` is the name of the environment, but any name of your choosing can be given.

    ```bash
    conda create -n mtnnenv pip python=3.7.4
    ```

3. Activate your conda environment

    ```bash
    conda activate mtnnenv
    ```

4. a. **Developer Mode** Install Python dependencies with `setup.py`. Setup.py adds the source code to the (virtualenv) Python path. 
Develop mode creates a symbolic link from the deployment directory `/home/user/.conda/envs/mtnnenv/lib/python3.7/site-packages/` 
to this source code directory `/home/user/mtnnpython/` instead of copying the source code directly. You can call `python setup.py develop` using the Makefile target `init`.
    
    ```bash
    make init
    ```

    To clean up distribution packages symbolic links:

    ```bash
    python setup.py develop --uninstall
    ```

    b. **Safest method**  `Setup.py` might pull the latest (incomptaible) package versions. So it's safe to use `pip install` will search the local project directory for dependencies. 

    **Development mode:**

    ```bash
    pip install -e .
    ```

    c. **Alternative** Navigate to to `mtnnpython/MTNN/environment/` and install version-specific Python dependencies in `requirements.txt` with `pip` in your Conda environment

    ```bash
    cd MTNN/environment
    pip install -r requirements.txt
    ```
  

5. Run an example script in `scripts/`

    ```bash
    python scripts/examples/hello_model.py
    ```


### With Virtualenv

```bash
    cd mtnnpython/mtnn
    python3 -m venv mtnn
    source mtnn/bin/activate
    cd environment/
    pip install -r requirements.txt
```

### How to export dependencies to requirements.txt

* Do a clean install of your conda/virtualenv (see Environment Set-up/With Conda steps 1-3)
* Test run code (e.g. examples/) and ensure all dependencies are imported 
* `$pip freeze > requirements.txt`


## Model Configuration 
The neural network architecture can be specified by a YAML configuration file. These are stored in `MTNN/config`.

### With Yaml Files

* TODO: Make a template example yaml file
### Generating YAML files 
 * TODO

## Testing

* In `mtnnconstants.py`, edit the `TEST_CONFIG_*` parameters to modify the range of parameters you want to
 create for testing yaml configuration files
* Run `config_generator.py`
    * This will generate test configuration files to `MTNN/tests/config/positive by default`
* Run tests in `MTNN/tests`

# Generating Documentation 
Documentation can be generated from commented source code using Doxygen. 

* Install [Doxygen](http://www.doxygen.nl/manual/install.html)
* In the commandline run `doxygen -g` to create an initial Doxyfile configuration file or use existing `mtnnpython/Doxyfile`
* Modify the `Doxyfile` configuration file  
* Run with Make:
     ```bash 
    make docs 
     ```

# Debugging 

## With MTNN Model logs
* For a MTNN Model object, set `Model.debug` to `True` to enable logging
* Logs will be stored in  `source folder/logs/`

## Visualization
### With ONYXX 
 * TODO 
### With Tensorboard
1. Initialize or set Model.Tensorboard to True
1.5 Remove Previous logs:

    ```bash
    !rm -rf ./runs/
    ```

2. In the commandline terminal:

    ```bash
    tensorboard --logdir=<full path to log directory> --port 6006
    ```

    * Defaults to `port 6006`
    * Don't use quotation marks in the directory path; Tensorboard will not raise any errors
3. In your browser:  `http://localhost:6006` or CTRL + click on link in the commandline terminal


