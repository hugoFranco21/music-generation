from tensorflow.python.framework import convert_to_constants
import os
from training import get_model
import tensorflow as tf
import tensorflow.compat.v1 as tf1

def print_checkpoint(save_path):
    latest = tf.train.latest_checkpoint(save_path)
    reader = tf.train.load_checkpoint(latest)
    shapes = reader.get_variable_to_shape_map()
    dtypes = reader.get_variable_to_dtype_map()
    print(f"Checkpoint at '{save_path}':")
    for key in shapes:
        print(f"  (key='{key}', shape={shapes[key]}, dtype={dtypes[key].name}, "
            f"value={reader.get_tensor(key)})")

def export_to_frozen_pb(model: tf.keras.models.Model, path: str) -> None:
    """
    Creates a frozen graph from a keras model.

    Turns the weights of a model into constants and saves the resulting graph into a protobuf file.

    Args:
        model: tf.keras.Model to convert into a frozen graph
        path: Path to save the profobuf file
    """
    inference_func = tf.function(lambda input: model(input))

    concrete_func = inference_func.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
    output_func = convert_to_constants.convert_variables_to_constants_v2(concrete_func)

    graph_def = output_func.graph.as_graph_def()
    graph_def.node[-1].name = 'output'

    with open(os.path.join(path, 'note_rnn.pb'), 'wb') as freezed_pb:
        freezed_pb.write(graph_def.SerializeToString())

def convert_tf2_to_tf1(checkpoint_dir, output_prefix):
    """Converts a TF2 checkpoint to TF1.

    The checkpoint must be saved using a 
    `tf.train.Checkpoint(var_list={name: variable})`

    To load the converted checkpoint with `tf.compat.v1.Saver`:
    ```
    saver = tf.compat.v1.train.Saver(var_list={name: variable}) 

    # An alternative, if the variable names match the keys:
    saver = tf.compat.v1.train.Saver(var_list=[variables]) 
    saver.restore(sess, output_path)

    ```
    """
    vars = {}
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print(latest)
    reader = tf.train.load_checkpoint(latest)
    print(reader)
    dtypes = reader.get_variable_to_dtype_map()
    print(dtypes)
    for key in dtypes.keys():
        # Get the "name" from the 
        print(key)
        if key.startswith('var_list/'):
            var_name = key.split('/')[1]
            # TF2 checkpoint keys use '/', so if they appear in the user-defined name,
            # they are escaped to '.S'.
            var_name = var_name.replace('.S', '/')
            vars[var_name] = tf.Variable(reader.get_tensor(key))

    return tf1.train.Saver(var_list=vars).save(sess=None, save_path=output_prefix)
    
converted_path = convert_tf2_to_tf1('training-checkpoints/',
                                    'model/')
print("\n[Converted]")
print_checkpoint(converted_path)

"""
model = get_model('model/note_rnn', 0.001)
export_to_frozen_pb(model, 'model/')
"""