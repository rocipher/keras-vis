from __future__ import absolute_import

import numpy as np
from keras import backend as K

from .callbacks import Print
from .grad_modifiers import get
from .utils import utils


class Optimizer(object):
    def _to_list(self, value):
        # vis.utils.utils.listify() inconvenient if value is None.
        if value is None:
            return []
        if isinstance(value, list):
            return value
        elif value is None:
            return []
        else:
            return [value]

    def __init__(self,
                 input_tensors,
                 losses,
                 input_ranges=[(0, 255)],
                 output_tensors=None,
                 norm_grads=True):
        self.input_tensors = self._to_list(input_tensors)
        self.input_ranges = self._to_list(input_ranges)
        if len(self.input_tensors) != len(self.input_ranges):
            if len(self.input_ranges) == 1:
                for i in range(1, len(self.input_tensors)):
                    self.input_ranges.append(self.input_ranges[0])
            else:
                raise ValueError(('The length of input_tensors '
                                  'and input_ranges must be the same.'))
        self.loss_names = []
        self.loss_functions = []
        if output_tensors is None or input_tensors is output_tensors:
            self.output_tensors = []
            for tensor in self.input_tensors:
                self.output_tensors.append(K.identity(tensor))
            self.output_tensors_is_input_tensors = True
        else:
            self.output_tensors = self._to_list(output_tensors)
            self.output_tensors_is_input_tensors = False

        overall_loss = 0
        for loss, weight in losses:
            if weight == 0:
                print("Don't build loss function with 0 weight.")
            loss_fn = weight * loss.build_loss()
            overall_loss = overall_loss + loss_fn
            self.loss_names.append(loss.name)
            self.loss_functions.append(loss_fn)

        if self.output_tensors_is_input_tensors:
            grads = K.gradients(overall_loss, self.input_tensors)
        else:
            grads = K.gradients(overall_loss, self.output_tensors)
        if norm_grads:
            for i in range(len(grads)):
                grads[i] = K.l2_normalize(grads[i])

        self.compute_fn = K.function(
            self.input_tensors + [K.learning_phase()],
            self.loss_functions + [overall_loss] + grads + self.output_tensors)

    def minimize(self,
                 seed_inputs=None,
                 max_iter=200,
                 input_modifiers=None,
                 grad_modifier=None,
                 callbacks=None,
                 verbose=True):
        seed_inputs = self._to_list(seed_inputs)
        for i in range(len(seed_inputs), len(self.input_tensors)):
            seed_inputs.append(None)
        for i in range(len(seed_inputs)):
            seed_inputs[i] = self._get_seed_input(i, seed_inputs[i])

        input_modifiers = input_modifiers or []
        grad_modifier = (lambda x: x) \
            if grad_modifier is None else get(grad_modifier)

        callbacks = callbacks or []
        if verbose:
            callbacks.append(Print())

        caches = []
        for i in range(len(seed_inputs)):
            caches.append(None)
        best_loss = float('inf')
        best_inputs = []

        grads = None
        output_values = []

        for i in range(max_iter):
            # Apply modifiers `pre` step
            for modifier in input_modifiers:
                for i in range(len(seed_inputs)):
                    seed_inputs[i] = modifier.pre(seed_inputs[i])

            # 0 learning phase for 'test'
            computed_values = self.compute_fn(seed_inputs + [0])
            cursor = len(self.loss_names)
            losses = computed_values[:cursor]
            named_losses = list(zip(self.loss_names, losses))
            overall_loss = computed_values[cursor]
            cursor += 1
            grads = computed_values[cursor:cursor + len(self.output_tensors)]
            cursor += len(self.output_tensors)
            output_values = computed_values[cursor:]

            # TODO: theano grads shape is inconsistent for some reason.
            # Patch for now and investigate later.
            for i, (grad, output_value) in enumerate(
                    zip(grads, output_values)):
                if grad.shape != output_value.shape:
                    grads[i] = np.reshape(grad, output_value.shape)

            # Apply grad modifier.
            grads = grad_modifier(grads)

            # Trigger callbacks
            for c in callbacks:
                c.callback(i, named_losses, overall_loss, grads, output_values)

            # Gradient descent update.
            # It only makes sense to do this if wrt_tensor is input_tensor.
            # Otherwise shapes wont match for the update.
            if self.output_tensors_is_input_tensors:
                for i in range(len(seed_inputs)):
                    step, caches[i] = self._rmsprop(grads[i], caches[i])
                    seed_inputs[i] = seed_inputs[i] + step

            # Apply modifiers `post` step
            for modifier in input_modifiers:
                for i in reversed(range(len(seed_inputs))):
                    seed_inputs[i] = modifier.post(seed_inputs[i])

            if overall_loss < best_loss:
                best_loss = overall_loss.copy()
                best_inputs = list(seed_inputs)

        # Trigger on_end
        for c in callbacks:
            c.on_end()

        results = []
        for best_input, input_range, grad, output_value in zip(
                best_inputs, self.input_ranges, grads, output_values):
            results.append((utils.deprocess_input(best_input[0], input_range),
                            grad, output_value))
        if len(results) == 1:
            return results[0]
        return results

    def _rmsprop(self, grads, cache=None, decay_rate=0.95):
        if cache is None:
            cache = np.zeros_like(grads)
        cache = decay_rate * cache + (1 - decay_rate) * grads**2
        step = -grads / np.sqrt(cache + K.epsilon())
        return step, cache

    def _get_seed_input(self, i, seed_input):
        """Creates a random `seed_input` if None. Otherwise:
            - Ensures batch_size dim on provided `seed_input`.
            - Shuffle axis according to expected `image_data_format`.
        """
        desired_shape = (1, ) + K.int_shape(self.input_tensors[i])[1:]
        if seed_input is None:
            return utils.random_array(
                desired_shape,
                mean=np.mean(self.input_ranges[i]),
                std=0.05 * (self.input_ranges[i][1] - self.input_ranges[i][0]))

        # Add batch dim if needed.
        if len(seed_input.shape) != len(desired_shape):
            seed_input = np.expand_dims(seed_input, 0)

        # Only possible if channel idx is out of place.
        if seed_input.shape != desired_shape:
            seed_input = np.moveaxis(seed_input, -1, 1)
        return seed_input.astype(K.floatx())
