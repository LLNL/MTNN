import sys

int_reader = lambda x : int(x)
int_reader.format_string = "int"

float_reader = lambda x : float(x)
float_reader.format_string = "float"

string_reader = lambda x : x
string_reader.format_string = "string"

bool_reader = lambda x : x.lower() in ("yes", "true", "t", "1")
bool_reader.format_string = "True/False"

ensure_trailing_reader = lambda tr : lambda x : x.rstrip(tr) + tr
ensure_trailing_reader.format_string = "string"

class array_reader:
    def __init__(self, element_reader):
        self.element_reader = element_reader
        el_fmt = element_reader.format_string
        self.format_string = el_fmt + "," + el_fmt + ",..."

    def __call__(self, x):
        return [self.element_reader(z) for z in x.split(',')]

class ArgReader:
    """ Class for simple parsing of command-line arguments.

    Handles command-line arguments of the form "arg1=val1 arg2=val2"
    """
    
    def __init__(self, additional_readers = None):
        """
        @param additional_readers A dict of (name : (reader_fn, help_string)) . See the
        self.reader_fns definition for examples.

        """
        self.reader_fns = {
            "num_cycles"   : (int_reader, "Number of multilevel training cycles to execute."),
            "num_levels"   : (int_reader, "Number of levels in the multilevel hierarchy."),
            "smooth_iters" : (int_reader,
                             "If using a NextKLoader subsetloader, number of minibatches in each smoothing pass."),
            "conv_ch"      : (array_reader(int_reader), "In a CNN, number of output channels in each convolutional level."),
            "conv_kernel_width" : (array_reader(int_reader),
                                   "In a CNN, width of square kernel in each convolutional level."),
            "conv_stride"  : (array_reader(int_reader), "In a CNN, stride of each convolutional layer."),
            "fc_width"     : (array_reader(int_reader),
                          "Width of each fully-connected layer. In a CNN, FC layers are assumed to come after convolutional layers, and first FC layer must have correct width to match against last convolutional output."),
            "momentum"     : (float_reader, "Momentum in SGD optimizer."),
            "learning_rate": (float_reader, "Learning rate in SGD optimizer."),
            "weight_decay" : (float_reader, "Weight decay in SGD optimizer."),
            "tau_corrector": (string_reader,
                              "Type of tau corrector to use. Options are 'none', 'wholeset', and 'minibatch'"),
            "weighted_projection": (bool_reader,
            "In the PariwiseOpsBuilder, weight restriction by vector norms. See PairwiseOpsBuilder for more information. (Default=True)"),
            "rand_seed"    : (int_reader, "Set the pseudorandom number generator seed. (Default=0)")}

        if additional_readers is not None:
            self.reader_fns.update(additional_readers)

    def print_help_and_exit(self, name):
        print("Command line format: python {} [option=value] [option=value]...".format(name))
        print("Options include the following.")
        for opt_name, val in self.reader_fns.items():
            print("{: >35}: {}".format("{}=({})".format(opt_name, val[0].format_string), val[1]))
        exit(1)

    def read_args(self, args):
        params_dict = {"weighted_projection": True, "rand_seed": 0}
        try:
            for a in args[1:]:
                tokens = a.split('=')
                params_dict[tokens[0]] = self.reader_fns[tokens[0]][0](tokens[1])
        except Exception as e:
            print(str(e) + "\n")
            self.print_help_and_exit(args[0])
        return params_dict
