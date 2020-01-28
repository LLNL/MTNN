MTNN PyTorch 
==============================================
# Purpose 
MTNN is a framework to develop and test the applicaiton of multigrid algorithms to various neural network architectures. 

## Project Layout

* **Source** is in `MTNN/`
* **Documentation** is in `MTNN/docs`
* **Dependencies** for environment set-up is in `MTNN/environment`
* **Configuration files** to run MTNN are in `MTNN/config`
  
* **Test hyper-parameters and directory path for outputs** are in `MTNN/mtnnconstants.py`
* **Unit tests** are in `MTNN/tests`
    * Positive test cases (where tests should *not* fail) are in `MTNN/tests/positives`
    * Negative test cases (where tests should fail) are in `MTNN/tests/negative`
* **Example scripts** using MTNN modules & framework are in `examples/`



# Environment Set-up
How to set-up environment and package dependencies on LC machines.
The following was tested on Pascal. 

Additional Confluence documentation on [how to set-up PyTorch on LC](https://lc.llnl.gov/confluence/display/LC/PyTorch+in+LC)

## With Conda 
**The following assumes Bash shell and that Anaconda is already installed on the system.** 
* To-do: Make this into a shell script/makefile

1. Log-in to an LC machine and HTTPS Git clone this repository. You'll be prompted to authenticate with your Active Directory Official Username (OUN) and RSA-OTP (One Time Password ).
    ```
    git clone https://lc.llnl.gov/bitbucket/scm/mtnn/mtnnpython.git
    ```
    
    Checkout this branch: Prolongation 

    ```
    git checkout prolongation
    ```

2. Create a Conda environment with  Python 3.7.4 and pip pre-installed.  In this example, `mtnnenv` is the name of the environment, but any name of your choosing can be given. 
    ```
    conda create -n mtnnenv pip python=3.7.4
    ```
3. Activate your conda environment 
    ```
    conda activate mtnnenv
    ```


4. a. Navigate to to `mtnnpython/MTNN/environment/` 
    ```
    cd mtnnpython/MTNN/environment
    ```
    b. Install Python dependencies in `requirements.txt` with `pip` in your Conda environment

    ```
    pip install -r requirements.txt
    ```

5. b. Build source code 
    ```
    cd ..
    python setup.py develop
    ```

6. Run an example script
    ```
    cd /examples
    python hello_model.py
    ```
## With Virtualenv
```
    cd mtnnpython/mtnn
    python3 -m venv mtnn
    source mtnn/bin/activate
    cd environment/
    pip install -r requirements.txt
```


# Configuration 
The neural network architecture can be specified by a YAML configuration file. These are stored in `MTNN/config`.

## With Yaml Files
   * To-do: Make a template example yaml file


# Testing
## Testing Set-up
* In `mtnnconstants.py`, edit the `TEST_CONFIG_*` parameters to modify the range of parameters you want to
 create for testing yaml configuration files
* Run `config_generator.py`
    * This will generate test configuration files to `MTNN/tests/config/positive by default`
* Run tests in `MTNN/tests`

# Generating  Sphinx Documentation 
 * To-do

# Debugging 
## With MTNN Model logs
* For a MTNN Model object, set `Model.debug` to `True` to enable logging
* Logs will be stored in  `source folder/logs/`

## Visualization With  Tensorboard
1. Initialize or set Model.Tensorboard to True
1.5 Remove Previous logs:
    ```
    !rm -rf ./runs/
    ```
2. In the commandline terminal:
    ```
    tensorboard --logdir=<full path to log directory> --port 6006
    ```
    * Defaults to port 6006
    * Don't use quotation marks in the directory path; Tensorboard will not raise any errors
3. In your browser:  http://localhost:6006 or CTRL + click on link in the commandline terminal


