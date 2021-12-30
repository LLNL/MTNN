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
    
    def __init__(self, reader_fns = None):
        """
        @param reader_fns A dict of (name : (reader_fn, help_string)) . See the
        self.reader_fns definition for examples.

        """
        self.reader_fns = reader_fns

    def print_help_and_exit(self, name):
        print("Command line format: python {} [option=value] [option=value]...".format(name))
        print("Options include the following.")
        for opt_name, val in self.reader_fns.items():
            print("{: >35}: {}".format("{}=({})".format(opt_name, val[0].format_string), val[1]))
        exit(1)

    def read_args(self, args):
        try:
            for a in args[1:]:
                tokens = a.split('=')
                self.params_dict[tokens[0]] = self.reader_fns[tokens[0]][0](tokens[1])
        except Exception as e:
            print("{} for argument '{}' with value '{}': {}\n".format(e.__class__.__name__, tokens[0], tokens[1], str(e)))
            self.print_help_and_exit(args[0])
        return self.params_dict

    
class MTNNArgReader(ArgReader):
    def __init__(self):
        super().__init__({
            "num_cycles"   : (int_reader, "Number of multilevel training cycles to execute."),
            "num_levels"   : (int_reader, "Number of levels in the multilevel hierarchy."),
            "smooth_iters" : (int_reader,
                             "If using a NextKLoader subsetloader, number of minibatches in each smoothing pass. (Default=4)"),
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
            "rand_seed"    : (int_reader, "Set the pseudorandom number generator seed. (Default=0)"),
            "log_filename" : (string_reader, "Filename to print log to (Default=./logs/mtnn.txt)")
            })
        
        self.params_dict = {"smooth_iters" : 4,
                            "weighted_projection": True,
                            "rand_seed": 0,
                            "log_filename" : "./logs/mtnn.txt"}


class VisualizationArgReader(ArgReader):
    def __init__(self):
        super().__init__({
            "file_names" : (array_reader(string_reader), "Array of data files"),
            "label_names" : (array_reader(string_reader), "Array of key names for the legend"),
            "work_units" : (array_reader(float_reader),
                            "Array of work unit multipliers associated with each data file. WUs are a ballpark relative measure of computational efforts associated with each Vcycle"),
            "save_prefix" : (string_reader,
                             "Filename prefix to use, files will be [save_prefix]_l2.pdf and [save_prefix]_linf.pdf"),
            "hier_levels" : (array_reader(int_reader), "Which hierarchy levels to display"),
            "smoothing_window" : (int_reader,
                                  "Size of moving average window. Not specifying or set to 1 implies no smoothing."),
            "start_reading_at" : (int_reader, "Cycle num to begin reading at. Avoids large values at startup.")
            })
        
        self.params_dict = {"smoothing_window" : 1,
                            "start_reading_at" : 0}
