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
        """additional_readers is a dict of (name, reader_fn) pairs. See the
        self.reader_fns definition for examples.

        """
        self.reader_fns = { "num_cycles" : int_reader,
                            "num_levels": int_reader,
                            "smooth_iters": int_reader,
                            "conv_ch" : array_reader(int_reader),
                            "conv_kernel_width" : array_reader(int_reader),
                            "conv_stride" : array_reader(int_reader),
                            "fc_width" : array_reader(int_reader),
                            "loader_sizes" : array_reader(int_reader),
                            "momentum": float_reader,
                            "learning_rate": float_reader,
                            "weight_decay": float_reader,
                            "tau_corrector": string_reader,
                            "weighted_projection": bool_reader,
                            "rand_seed" : int_reader}
        if additional_readers is not None:
            self.reader_fns.update(additional_readers)

        self.format_string = "python " + sys.argv[0] + " \\\n" + "\n".join(["   {}=[{}] \\".format(k, v.format_string) for k,v in self.reader_fns.items()])

    def read_args(self, args):
        params_dict = {}
        try:
            for a in args[1:]:
                tokens = a.split('=')
                params_dict[tokens[0]] = self.reader_fns[tokens[0]](tokens[1])
        except Exception as e:
            exit(str(e) + "\n\nCommand line format: " + self.format_string)
        return params_dict
