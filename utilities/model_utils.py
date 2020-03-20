import tensorflow as tf
from utilities.util import flatten
from typing import Tuple, Union, List

def extract_layers(network: tf.keras.Model, unfold_tds: bool = False) -> List[tf.keras.layers.Layer]:
    """Recursively extract layers from a potentially nested list of Sequentials of unknown depth."""
    if not hasattr(network, "layers"):
        return [network]

    layers = []
    for l in network.layers:
        if isinstance(l, tf.keras.Model) or isinstance(l, tf.keras.Sequential):
            layers.append(extract_layers(l))
        elif isinstance(l, tf.keras.layers.TimeDistributed) and unfold_tds:
            if isinstance(l.layer, tf.keras.Model) or isinstance(l.layer, tf.keras.Sequential):
                layers.append(extract_layers(l.layer))
            else:
                layers.append(l.layer)
        else:
            layers.append(l)

    return flatten(layers)


def get_layers_by_names(network: tf.keras.Model, layer_names: List[str]):
    """Get a list of layers identified by their names from a network."""
    layers = extract_layers(network)
    all_layer_names = [l.name for l in layers]

    assert all(ln in all_layer_names for ln in layer_names), "Cannot find layer name in network extraction."

    return [layers[all_layer_names.index(layer_name)] for layer_name in layer_names]


def build_sub_model_to(network: tf.keras.Model, tos: Union[List[str], List[tf.keras.Model]], include_original=False):
    """Build a sub model of a given network that has (multiple) outputs at layer activations defined by a list of layer
    names."""
    layers = get_layers_by_names(network, tos) if isinstance(tos[0], str) else tos
    outputs = []

    # probe layers to check if model can be build to them
    for layer in layers:
        success = False
        layer_input_id = 0
        while not success:
            success = True
            try:
                tf.keras.Model(inputs=[network.input], outputs=[layer.get_output_at(layer_input_id)])
            except ValueError as ve:
                if len(ve.args) > 0 and ve.args[0].split(" ")[0] == "Graph":
                    layer_input_id += 1
                    success = False
                else:
                    raise ValueError(f"Cannot use layer {layer.name}. Error: {ve.args}")
            else:
                outputs.append([layer.get_output_at(layer_input_id)])

    if include_original:
        outputs = outputs + network.outputs

    return tf.keras.Model(inputs=[network.input], outputs=outputs)