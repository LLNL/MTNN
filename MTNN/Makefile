.PHONY: init req

TESTPATH = ./tests/

help:
	@echo "Please use \'make <target>' where <target> is one of these options" 
	@echo "init 		Install package dependencies and run in developer mode" 
	@echo "req      	Write source code dependencies to requirements.txt"
	@echo "clean-tests      Clean tests/ folder of generated tests cases"
	@echo "clean-logs       Clean log files from scripts/experiments/examples/runs/logs/"

init:
	@echo "init"
	@echo "     installing package dependencies"
	python ../setup.py develop

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

clean-tests: 
	@echo " clean-tests" 
	@echo "		remove YAML files from tests/config/positives/ "

	 find ./tests/config/positive -maxdepth 1 -type f -name "*.yaml" -exec rm -f {} \;

clean-logs:
	@echo " clean-logs"
	@echo "     remove *.txt files from scripts/examples/runs/logs/"

	find ../scripts/examples/runs/logs/ -type f -name "*.txt" -exec rm -f {} \;

