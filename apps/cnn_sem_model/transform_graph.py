# pylint: disable=g-bad-file-header
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Removes parts of a graph that are only needed for training.

There are several common transformations that can be applied to GraphDefs
created to train a model, that help reduce the amount of computation needed when
the network is used only for inference. These include:

 - Removing training-only operations like checkpoint saving.

 - Stripping out parts of the graph that are never reached.

 - Removing debug operations like CheckNumerics.

 - Folding batch normalization ops into the pre-calculated weights.

 - Fusing common operations into unified versions.

This script takes either a frozen binary GraphDef file (where the weight
variables have been converted into constants by the freeze_graph script), or a
text GraphDef proto file (the weight variables are stored in a separate
checkpoint file), and outputs a new GraphDef with the optimizations applied.

If the input graph is a text graph file, make sure to include the node that
restores the variable weights in output_names. That node is usually named
"restore_all".

An example of command-line usage is:

  mxp_python ../python/cnn_sem_model/transform_graph.py 
--input dlsem_test_model.pb
--output opt_graph.pb 
--input_names input,truth,phase 
--output_names output,cost 
--frozen_graph=True
--transform_graph=True


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

from google.protobuf import text_format
import tensorflow as tf

from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_io
from tensorflow.python.platform import app
from tensorflow.python.platform import gfile
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.framework import graph_util


FLAGS = None


def main(unused_args):
  if not gfile.Exists(FLAGS.input):
    print("Input graph file '" + FLAGS.input + "' does not exist!")
    return -1

  input_graph_def = graph_pb2.GraphDef()
  with gfile.Open(FLAGS.input, "rb") as f:
    data = f.read()
    if FLAGS.frozen_graph:
      input_graph_def.ParseFromString(data)
    else:
      text_format.Merge(data.decode("utf-8"), input_graph_def)

  input_graph_def = graph_util.convert_variables_to_constants(
    tf.Session(), 
    input_graph_def,
    FLAGS.output_names.split(","),
  )

  '''
  input_graph_def = optimize_for_inference_lib.optimize_for_inference(
      input_graph_def,
      FLAGS.input_names.split(","),
      FLAGS.output_names.split(","), FLAGS.placeholder_type_enum)
  '''

  transforms = ["strip_unused_nodes", 
                #"fold_constants(ignore_errors=True)",
                "fold_batch_norms",
                "fold_old_batch_norms",
                #"round_weights(num_steps=256)"
               ]
  input_graph_def = TransformGraph(
      input_graph_def,
      FLAGS.input_names.split(","),
      FLAGS.output_names.split(","), 
      transforms)

  if FLAGS.frozen_graph:
    f = gfile.FastGFile(FLAGS.output, "w")
    f.write(input_graph_def.SerializeToString())
  else:
    graph_io.write_graph(input_graph_def,
                         os.path.dirname(FLAGS.output),
                         os.path.basename(FLAGS.output))
  tf.train.write_graph(input_graph_def, '.', FLAGS.output + "txt")
  return 0


def parse_args():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--input",
      type=str,
      default="",
      help="TensorFlow \'GraphDef\' file to load.")
  parser.add_argument(
      "--output",
      type=str,
      default="",
      help="File to save the output graph to.")
  parser.add_argument(
      "--input_names",
      type=str,
      default="",
      help="Input node names, comma separated.")
  parser.add_argument(
      "--output_names",
      type=str,
      default="",
      help="Output node names, comma separated.")
  parser.add_argument(
      "--frozen_graph",
      nargs="?",
      const=True,
      type="bool",
      default=True,
      help="""\
      If true, the input graph is a binary frozen GraphDef
      file; if false, it is a text GraphDef proto file.\
      """)
  parser.add_argument(
      "--placeholder_type_enum",
      type=int,
      default=dtypes.float32.as_datatype_enum,
      help="The AttrValue enum to use for placeholders.")
  return parser.parse_known_args()


if __name__ == "__main__":
  '''
  Example command line: 
  go to mxp/data folder. then run command line:
  mxp_python ../python/cnn_sem_model/transform_graph.py --input dlsem_test_model.pb --output opt_graph.pb --input_names input,truth,phase --output_names output,cost --frozen_graph

  about what transform we can try here, please read:
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/graph_transforms/README.md
  '''
  FLAGS, unparsed = parse_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
    