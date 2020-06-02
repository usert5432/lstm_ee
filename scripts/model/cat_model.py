"""Pretty Print training configuration and model structure"""

import argparse
from lstm_ee.utils.io import load_model

def parse_cmdargs():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser("Pretty print model")
    parser.add_argument(
        'outdir',
        metavar = 'OUTDIR',
        type    = str,
        help    = 'Directory with saved models'
    )

    return parser.parse_args()

def main():
    # pylint: disable=missing-function-docstring
    cmdargs = parse_cmdargs()
    args, model = load_model(cmdargs.outdir, compile = False)

    print(args.config.pprint())
    #print(model.get_config())
    print(model.summary())

if __name__ == '__main__':
    main()
