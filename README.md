The MTNN PyTorch Stuff
=========================

The example is in `examples/` and the source is in `MTNN/`.


### How to install dependencies with Virtualenv
```
    cd mtnnpython/mtnn
    python3 -m venv mtnn
    source mtnn/bin/activate
    cd environment/
    pip install -r requirements.txt
```

### How to run Tensorboard
1. Initialize or set Model.Tensorboard to True
1.5 Remove Previous logs:
`!rm -rf ./runs/`
2. In the commandline terminal:
    `tensorboard --logdir=<full path to log directory> --port 6006`
    * Defaults to port 6006
    * Don't use quotation marks in the directory path; Tensorboard will not raise any errors
3. In your browser:  http://localhost:6006 or CTRL + click on link in the commandline terminal