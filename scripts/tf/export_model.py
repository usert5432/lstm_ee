"""Extract tensorflow graph from a `keras` model and export it to a file.

Notes
-----
This is tested only with tensorflow v1. Need to update for TF v2.
"""

import argparse
import json
import os

from  keras import backend as K

import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph

from lstm_ee.utils.io import load_model

# TODO: verify if this is really needed
K.set_learning_phase(0)

def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser("Convert keras model to TF graph")

    parser.add_argument(
        'outdir',
        help    = 'Directory with saved models',
        metavar = 'OUTDIR',
        type    = str,
    )

    parser.add_argument(
        '-t', '--text',
        action = 'store_true',
        dest   = 'text_also',
        help   = 'Save ascii version also',
    )

    parser.add_argument(
        '-r', '--raw',
        action = 'store_true',
        dest   = 'raw_also',
        help   = 'Save raw unoptimized model also',
    )

    return parser

def freeze_session(session, output_names):
    """Freeze tensorflow graph"""
    graph = session.graph

    with graph.as_default():
        input_graph_def = graph.as_graph_def()

        for node in input_graph_def.node:
            node.device = ""

        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names
        )

        return frozen_graph

def get_optimized_graph(frozen_graph, inputs, outputs):
    """Optimize tensorflow graph for evaluation"""

    trans_list = [
        'strip_unused_nodes',
        'remove_nodes(op=Identity, op=CheckNumerics)',
        'fold_constants(ignore_errors=true)',
        'fold_batch_norms',
        'fold_old_batch_norms',
    ]

    return TransformGraph(frozen_graph, inputs, outputs, trans_list)

def save_graph(graph, root, name, text_also = False):
    """Save tensorflow graph"""
    tf.train.write_graph(graph, root, name, as_text=False)

    if text_also:
        tf.train.write_graph(graph, root, name + ".txt", as_text=True)

def get_tf_opname_for_layer(model, layer_name, output_op = False):
    """Return tf i/o node name for a layer of keras model"""
    try:
        layer = model.get_layer(layer_name)
    except ValueError:
        return None

    if output_op:
        return layer.output.op.name
    else:
        return layer.input.op.name

def create_tf_config(args, model):
    """
    Create evaluation configuration that holds input variables and graph nodes
    """
    config_tf = {}

    config_tf['vars_slice'] = args.vars_input_slice
    config_tf['vars_png2d'] = args.vars_input_png2d
    config_tf['vars_png3d'] = args.vars_input_png3d

    config_tf.update({
        x : get_tf_opname_for_layer(model, x, False) \
            for x in [ 'input_slice', 'input_png2d', 'input_png3d' ]
    })

    config_tf.update({
        x : get_tf_opname_for_layer(model, x, True) \
            for x in [ 'target_primary', 'target_total' ]
    })

    return config_tf

def save_config(config_tf, outdir_tf):
    """Create evaluation configuration"""
    with open(os.path.join(outdir_tf, "config.json"), "wt") as f:
        json.dump(config_tf, f, indent = 4, sort_keys = True)

def main():
    # pylint: disable=missing-function-docstring
    parser  = create_parser()
    cmdargs = parser.parse_args()

    args, model = load_model(cmdargs.outdir, compile = False)

    config_tf = create_tf_config(args, model)

    inputs  = [node.op.name for node in model.inputs]
    outputs = [node.op.name for node in model.outputs]

    outdir_tf = os.path.join(cmdargs.outdir, "tf")
    os.makedirs(outdir_tf, exist_ok = True)

    save_config(config_tf, outdir_tf)

    frozen_graph = freeze_session(K.get_session(), outputs)
    if cmdargs.raw_also:
        save_graph(frozen_graph, outdir_tf, "model_raw.pb", cmdargs.text_also)

    optimized_graph = get_optimized_graph(frozen_graph, inputs, outputs)
    save_graph(optimized_graph, outdir_tf, "model.pb", cmdargs.text_also)


if __name__ == '__main__':
    main()

