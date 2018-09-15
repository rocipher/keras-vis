from __future__ import absolute_import

import numpy as np
from keras import backend as K

from .callbacks import Print
from .grad_modifiers import get
from .utils import utils


class Optimizer(object):

    def __init__(self, input_tensors, losses, input_ranges=(0, 255), wrt_tensor=None, norm_grads=True):
        """Creates an optimizer that minimizes weighted loss function.

        Args:
            input_tensors: An input tensor or list of input tensor.
                The shape of an input tensor is `(samples, channels, image_dims...)` if `image_data_format=
                channels_first`, Or it's `(samples, image_dims..., channels)` if `image_data_format=channels_last`.
            losses: List of ([Loss](vis.losses#Loss), weight) tuples.
            input_ranges: A `(min, max)` tuple or a list of `(min, max)` tuple corresponding to `input_tensors` input range.
                This is used to rescale the final optimized input to the given range. (Default value=(0, 255))
                If a single range is provided it will be assumed for all input_tensors.
            wrt_tensor: Short for, with respect to. This instructs the optimizer that the aggregate loss from `losses`
                should be minimized with respect to `wrt_tensor`.
                `wrt_tensor` can be any tensor that is part of the model graph. Default value is set to None
                which means that loss will simply be minimized with respect to `input_tensors`.
            norm_grads: True to normalize gradients. Normalization avoids very small or large gradients and ensures
                a smooth gradient gradient descent process. If you want the actual gradient
                (for example, visualizing attention), set this to false.
        """
        self.input_tensors = utils.listify(input_tensors)
        self.input_ranges = utils.listify(input_ranges)
        if len(self.input_tensors) != len(self.input_ranges):
            if len(self.input_ranges) == 1:
                self.input_ranges = self.input_ranges * len(self.input_tensors)
            else:
                raise ValueError(('The length of input_ranges must be the same as input_tensors or just 1.'))
        self.loss_names = []
        self.loss_functions = []
        if wrt_tensor is None:
            wrt_tensor = self.input_tensors[0]
        if wrt_tensor in self.input_tensors:
            self.wrt_tensor_index = self.input_tensors.index(wrt_tensor)
            self.wrt_tensor = K.identity(wrt_tensor)
        else:
            self.wrt_tensor_index = None
            self.wrt_tensor = wrt_tensor

        overall_loss = 0
        for loss, weight in losses:
            # Perf optimization. Don't build loss function with 0 weight.
            if weight != 0:
                loss_fn = weight * loss.build_loss()
                overall_loss = overall_loss + loss_fn
                self.loss_names.append(loss.name)
                self.loss_functions.append(loss_fn)

        # Compute gradient of overall with respect to `wrt` tensor.
        grads = K.gradients(overall_loss, self.wrt_tensor)[0]
        if norm_grads:
            grads = K.l2_normalize(grads)

        # The main function to compute various quantities in optimization loop.
        self.compute_fn = K.function(
            self.input_tensors + [K.learning_phase()],
            self.loss_functions + [overall_loss, grads, self.wrt_tensor])

    def _rmsprop(self, grads, cache=None, decay_rate=0.95):
        """Uses RMSProp to compute step from gradients.

        Args:
            grads: numpy array of gradients.
            cache: numpy array of same shape as `grads` as RMSProp cache
            decay_rate: How fast to decay cache

        Returns:
            A tuple of
                step: numpy array of the same shape as `grads` giving the step.
                    Note that this does not yet take the learning rate into account.
                cache: Updated RMSProp cache.
        """
        if cache is None:
            cache = np.zeros_like(grads)
        cache = decay_rate * cache + (1 - decay_rate) * grads ** 2
        step = -grads / np.sqrt(cache + K.epsilon())
        return step, cache

    def _get_seed_inputs(self, seed_inputs):
        """Creates a random `seed_input` if None. Otherwise:
            - Ensures batch_size dim on provided `seed_input`.
            - Shuffle axis according to expected `image_data_format`.
        """

        seed_inputs = utils.listify(seed_inputs)
        # Make seed_inputs as same length as input_tensors
        for _ in range(len(seed_inputs), len(self.input_tensors)):
            seed_inputs.append(None)

        for i in range(len(self.seed_inputs)):
            desired_shape = (1, ) + K.int_shape(self.input_tensors[i])[1:]
            min_val, max_val = input_ranges[i]
            seed_input = seed_inputs[i]
            if seed_input is None:
                seed_input = utils.random_array(desired_shape, mean=np.mean((min_val, max_val)), std=0.05 * (max_val - min_val))

            # Add batch dim if needed.
            if len(seed_input.shape) != len(desired_shape):
                seed_input = np.expand_dims(seed_input, 0)

            # Only possible if channel idx is out of place.
            if seed_input.shape != desired_shape:
                seed_input = np.moveaxis(seed_input, -1, 1)

            seed_inputs[i] = seed_input.astype(K.floatx())

        return seed_inputs

    def minimize(self, seed_inputs=None, max_iter=200,
                 input_modifiers=None, grad_modifier=None,
                 callbacks=None, verbose=True):
        """Performs gradient descent on the input image with respect to defined losses.

        Args:
            seed_inputs: A numpy array or list of numpy array.
                The shape of an N-dim numpy array is `(samples, channels, image_dims...)` if `image_data_format=
                channels_first`, Or it's `(samples, image_dims..., channels)` if `image_data_format=channels_last`.
                Seeded with random noise if set to None. (Default value = None)
            max_iter: The maximum number of gradient descent iterations. (Default value = 200)
            input_modifiers: A list of InputModifier, or a list of tuple of target input tensor and InputModifier.
                [InputModifier](vis.input_modifiers#inputmodifier) instances specifying
                how to make `pre` and `post` changes to the optimized input during the optimization process.
                `pre` is applied in list order while `post` is applied in reverse order. For example,
                `input_modifiers = [f, g]` means that `pre_input = g(f(inp))` and `post_input = f(g(inp))`
            grad_modifier: gradient modifier to use. See [grad_modifiers](vis.grad_modifiers.md). If you don't
                specify anything, gradients are unchanged. (Default value = None)
            callbacks: A list of of OptimizerCallback, or a list of tuple of target tensor and OptimizerCallback.
                [OptimizerCallback](vis.callbacks#optimizercallback) instances to trigger.
                target tensor is usualy wrt_tensor, but when wrt_tensors is None, it is input_tensors.
                That is, When optimizer is used ActivationMaximization.
            verbose: Logs individual losses at the end of every gradient descent iteration.
                Very useful to estimate loss weight factor(s). (Default value = True)

        Returns:
            The tuple of `(optimized input, grads with respect to wrt, wrt_value)` after gradient descent iterations.
        """
        seed_inputs = self._get_seed_inputs(seed_inputs)

        input_modifiers = input_modifiers or []
        grad_modifier = (lambda x: x) if grad_modifier is None else get(grad_modifier)

        callbacks = callbacks or []
        if verbose:
            callbacks.append(Print())

        # Make caches as same length as seed_inputs (i.e., input_tensors)
        cache = None

        best_loss = float('inf')
        best_inputs = None

        grads = None
        wrt_value = None

        for i in range(max_iter):
            # Apply modifiers `pre` step
            for modifier in input_modifiers:
                if type(modifier) is tuple:
                    target, modifier = modifier
                    j = self.input_tensors.index(target)
                    seed_inputs[j] = modifier.pre(seed_inputs[j])
                else:
                    for j in range(len(seed_inputs)):
                        seed_inputs[j] = modifier.pre(seed_inputs[j])

            # 0 learning phase for 'test'
            computed_values = self.compute_fn(seed_inputs + [0])
            losses = computed_values[:len(self.loss_names)]
            named_losses = list(zip(self.loss_names, losses))
            overall_loss, grads, wrt_value = computed_values[len(self.loss_names):]

            # TODO: theano grads shape is inconsistent for some reason. Patch for now and investigate later.
            if grads.shape != wrt_value.shape:
                grads = np.reshape(grads, wrt_value.shape)

            # Apply grad modifier.
            grads = grad_modifier(grads)

            # Trigger callbacks
            for c in callbacks:
                c.callback(i, named_losses, overall_loss, grads, wrt_value)

            # Gradient descent update. It only makes sense to do this if wrt_tensor is input_tensor.
            # Otherwise shapes wont match for the update.
            if self.wrt_tensor_index is not None:
                step, cache = self._rmsprop(grads, cache)
                seed_inputs[self.wrt_tensor_index] += step

            # Apply modifiers `post` step
            for modifier in reversed(input_modifiers):
                if type(modifier) is tuple:
                    target, modifier = modifier
                    j = self.input_tensors.index(target)
                    seed_inputs[j] = modifier.poset(seed_inputs[j])
                else:
                    for j in reversed(range(len(seed_inputs))):
                        seed_inputs[j] = modifier.post(seed_inputs[j])

            if overall_loss < best_loss:
                best_loss = overall_loss.copy()
                best_inputs = np.array(seed_inputs).copy()

        # Trigger on_end
        for c in callbacks:
            c.on_end()

        if self.wrt_tensor_index is None:
            deprocessed_input = None
        else:
            deprocessed_input = utils.deprocess_input(best_inputs[self.wrt_tensor_index][0],
                                                      self.input_ranges[self.wrt_tensor_index])
        return deprocessed_input, grads, wrt_value
