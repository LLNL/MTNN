.PHONY: help setup init req clean-tests clean-logsrun  test 

TESTPATH = ./tests/

help:
	@echo "Usage: make [TARGET]"
	@echo "Targets:" 
	@echo "setup		Build developer environment: creates conda environment and run in developer mode" 
	@echo "init 		Install package dependencies and run in developer mode" 
	@echo "req      	Write source code dependencies to requirements.txt"
	@echo "test			Run Pytests in test/"
	@echo "run			Run script with config file"
	@echo "clean-tests      Clean tests/ folder of generated tests cases"
	@echo "clean-logs       Clean log files from scripts/experiments/examples/runs/logs/"


init:
	@echo "init"
	@echo "     installing package dependencies"
	python setup.py develop

req:
	@echo "requirements" 
	@echo "		write source code dependencies to ./requirements.txt"
# TODO: Activate conda in subshell to collect clean requirements
#	conda create -y -n mtnn_dependencies pip python=3.7.4
#	eval "$(conda shell.bash hook)"
#	conda activate mtnn_dependencies
	pip freeze >> test.txt
	@echo "Requirements gathered:"
	cat test.txt
#	conda deactivate
#	conda remove --name mtnn_dependencies --all

###########################################################
# Developer 
##########################################################
baseline: 
	python MTNN/run.py MTNN/scripts/examples/hello_model.py MTNN/config/hello_model.yaml --debug --log

run: 
	python MTNN/run.py MTNN/scripts/experiments/find_overfit.py MTNN/config/mnist_mgbase.yaml --debug

# Clean 
clean-tests: 
	@echo " clean-tests" 
	@echo "		remove YAML files from tests/config/positives/ "

	 find ./tests/config/positive -maxdepth 1 -type f -name "*.yaml" -exec rm -f {} \;

clean-logs:
	@echo " clean-logs"
	@echo "     clear runs/ directory of logs, checkpoints, tensorboard event files, etc."

	#find /scripts/examples/runs/logs/ -type f -name "*.txt" -exec rm -rf {} \; # Generated debugging logs
	find . -type d -name runs* -exec rm -rf {} \; # Model run logs


# Tests
test-config::
	pytest tests/test_config.py

test_model: 
	pytest tests/test_model.py




