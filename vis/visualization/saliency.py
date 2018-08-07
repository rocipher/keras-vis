from __future__ import absolute_import

import numpy as np
from scipy.ndimage.interpolation import zoom

from keras.layers.convolutional import _Conv
from keras.layers.pooling import _Pooling1D, _Pooling2D, _Pooling3D
from keras.layers.wrappers import Wrapper
from keras import backend as K

from ..losses import ActivationMaximization
from ..optimizer import Optimizer
from ..backprop_modifiers import get
from ..utils import utils


def _find_penultimate_layer(model, layer_idx, penultimate_layer_idx):
    if penultimate_layer_idx is None:
        for idx, layer in utils.reverse_enumerate(
                model.layers[:layer_idx - 1]):
            if isinstance(layer, Wrapper):
                layer = layer.layer
            if isinstance(layer, (_Conv, _Pooling1D, _Pooling2D, _Pooling3D)):
                penultimate_layer_idx = idx
                break

    if penultimate_layer_idx is None:
        raise ValueError('Unable to determine penultimate `Conv` or `Pooling` '
                         'layer for layer_idx: {}'.format(layer_idx))

    # Handle negative indexing otherwise the next check can fail.
    if layer_idx < 0:
        layer_idx = len(model.layers) + layer_idx
    if penultimate_layer_idx > layer_idx:
        raise ValueError(
            '`penultimate_layer_idx` needs to be before `layer_idx`')

    return model.layers[penultimate_layer_idx]


def _normalize_grads_for_saliency(grads):
    channel_idx = 1 if K.image_data_format() == 'channels_first' else -1
    grads = np.max(grads, axis=channel_idx)
    grads = utils.normalize(grads)[0]
    return grads


def visualize_saliency_with_losses(input_tensor,
                                   losses,
                                   seed_input,
                                   wrt_tensor=None,
                                   grad_modifier='absolute'):
    opt = Optimizer(
        input_tensor, losses, output_tensors=wrt_tensor, norm_grads=False)
    results = opt.minimize(
        seed_inputs=seed_input,
        max_iter=1,
        grad_modifier=grad_modifier,
        verbose=False)

    if len(input_tensor) > 1:
        grads = []
        for result in results:
            _, g, _ = result
            grads.append(_normalize_grads_for_saliency(g))
    else:
        _, grads, _ = results
        grads = _normalize_grads_for_saliency(grads)
    return grads


def visualize_saliency(model,
                       layer_idx,
                       filter_indices,
                       seed_input,
                       wrt_tensor=None,
                       backprop_modifier=None,
                       grad_modifier='absolute'):
    if backprop_modifier is not None:
        modifier_fn = get(backprop_modifier)
        model = modifier_fn(model)

    losses = [(ActivationMaximization(model.layers[layer_idx], filter_indices),
               -1)]
    return visualize_saliency_with_losses(model.inputs, losses, seed_input,
                                          wrt_tensor, grad_modifier)


def _normalize_grads_for_cam(input_tensor, output_dims, grads,
                             penultimate_output_value):
    grads = grads / (np.max(grads) + K.epsilon())

    channel_idx = 1 if K.image_data_format() == 'channels_first' else -1
    other_axis = np.delete(np.arange(len(grads.shape)), channel_idx)
    weights = np.mean(grads, axis=tuple(other_axis))

    heatmap = np.zeros(shape=output_dims, dtype=K.floatx())
    for i, w in enumerate(weights):
        if channel_idx == -1:
            heatmap += w * penultimate_output_value[0, ..., i]
        else:
            heatmap += w * penultimate_output_value[0, i, ...]

    heatmap = np.maximum(heatmap, 0)

    input_dims = utils.get_img_shape(input_tensor)[2:]

    zoom_factor = [
        i / (j * 1.0) for i, j in iter(zip(input_dims, output_dims))
    ]
    heatmap = zoom(heatmap, zoom_factor)
    return utils.normalize(heatmap)


def visualize_cam_with_losses(input_tensor,
                              losses,
                              seed_input,
                              penultimate_layer,
                              grad_modifier=None):
    penultimate_output = penultimate_layer.output
    opt = Optimizer(
        input_tensor,
        losses,
        output_tensors=penultimate_output,
        norm_grads=False)
    results = opt.minimize(
        seed_input, max_iter=1, grad_modifier=grad_modifier, verbose=False)

    output_dims = utils.get_img_shape(penultimate_output)[2:]
    _, grads, penultimate_output_value = results

    if len(input_tensor) > 1:
        grads_list = []
        for tensor in input_tensor:
            grads_list.append(
                _normalize_grads_for_cam(tensor, output_dims, grads,
                                         penultimate_output_value))
        grads = grads_list
    else:
        grads = _normalize_grads_for_cam(input_tensor[0], output_dims, grads,
                                         penultimate_output_value)
    return grads


def visualize_cam(model,
                  layer_idx,
                  filter_indices,
                  seed_input,
                  penultimate_layer_idx=None,
                  backprop_modifier=None,
                  grad_modifier=None):
    if backprop_modifier is not None:
        modifier_fn = get(backprop_modifier)
        model = modifier_fn(model)

    penultimate_layer = _find_penultimate_layer(model, layer_idx,
                                                penultimate_layer_idx)

    losses = [(ActivationMaximization(model.layers[layer_idx], filter_indices),
               -1)]
    return visualize_cam_with_losses(model.inputs, losses, seed_input,
                                     penultimate_layer, grad_modifier)
