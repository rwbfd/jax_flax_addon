from typing import (Any, Callable, Optional, Tuple)

from jax import lax
from jax.nn import initializers
import jax.numpy as jnp
from jax import vmap, partial
from flax.linen.module import Module, compact

PRNGKey = Any
Array = Any
Shape = Tuple[int]
Dtype = Any  # this could be a real type?

_no_init = lambda rng, shape: ()


def _absolute_dims(rank, dims):
    return tuple([rank + dim if dim < 0 else dim for dim in dims])


class _GhostBatchNormImpl(Module):
    """BatchNorm Module.
    Attributes:
      use_running_average: if true, the statistics stored in batch_stats
        will be used instead of computing the batch statistics on the input.
      axis: the feature or non-batch axis of the input.
      momentum: decay rate for the exponential moving average of
        the batch statistics.
      epsilon: a small float added to variance to avoid dividing by zero.
      dtype: the dtype of the computation (default: float32).
      bias:  if True, bias (beta) is added.
      scale: if True, multiply by scale (gamma).
        When the next layer is linear (also e.g. nn.relu), this can be disabled
        since the scaling will be done by the next layer.
      bias_init: initializer for bias, by default, zero.
      scale_init: initializer for scale, by default, one.
      axis_name: the axis name used to combine batch statistics from multiple
        devices. See `jax.pmap` for a description of axis names (default: None).
      axis_index_groups: groups of axis indices within that named axis
        representing subsets of devices to reduce over (default: None). For
        example, `[[0, 1], [2, 3]]` would independently batch-normalize over
        the examples on the first two and last two devices. See `jax.lax.psum`
        for more details.
    """
    use_running_average: bool = False
    axis: int = -1
    momentum: float = 0.99
    epsilon: float = 1e-5
    dtype: Dtype = jnp.float32
    axis_name: Optional[str] = None
    axis_index_groups: Any = None

    @compact
    def __call__(self, x):
        """Normalizes the input using batch statistics.
        Args:
          x: the input to be normalized.
        Returns:
          Normalized inputs (the same shape as inputs).
        """
        x = jnp.asarray(x, jnp.float32)
        axis = self.axis if isinstance(self.axis, tuple) else (self.axis,)
        axis = _absolute_dims(x.ndim, axis)
        feature_shape = tuple(d if i in axis else 1 for i, d in enumerate(x.shape))
        reduced_feature_shape = tuple(d for i, d in enumerate(x.shape) if i in axis)
        reduction_axis = tuple(i for i in range(x.ndim) if i not in axis)
        initializing = not self.has_variable('batch_stats', 'mean')

        ra_mean = self.variable('batch_stats', 'mean',
                                lambda s: jnp.zeros(s, jnp.float32),
                                reduced_feature_shape)
        ra_var = self.variable('batch_stats', 'var',
                               lambda s: jnp.ones(s, jnp.float32),
                               reduced_feature_shape)
        if self.use_running_average:
            mean, var = ra_mean.value, ra_var.value
        else:
            mean = jnp.mean(x, axis=reduction_axis, keepdims=False)
            mean2 = jnp.mean(lax.square(x), axis=reduction_axis, keepdims=False)
            if self.axis_name is not None and not initializing:
                concatenated_mean = jnp.concatenate([mean, mean2])
                mean, mean2 = jnp.split(
                    lax.pmean(
                        concatenated_mean,
                        axis_name=self.axis_name,
                        axis_index_groups=self.axis_index_groups), 2)
            var = mean2 - lax.square(mean)

            if not initializing:
                ra_mean.value = self.momentum * ra_mean.value + (1 - self.momentum) * mean
                ra_var.value = self.momentum * ra_var.value + (1 - self.momentum) * var

        y = x - mean.reshape(feature_shape)
        mul = lax.rsqrt(var + self.epsilon)
        y = y * mul
        return jnp.asarray(y, self.dtype)


class GhostBatchNorm(Module):
    """GhostBatchNorm Module.
    Attributes:
      use_running_average: if true, the statistics stored in batch_stats
        will be used instead of computing the batch statistics on the input.
      axis: the feature or non-batch axis of the input.
      momentum: decay rate for the exponential moving average of
        the batch statistics.
      epsilon: a small float added to variance to avoid dividing by zero.
      dtype: the dtype of the computation (default: float32).
      bias:  if True, bias (beta) is added.
      scale: if True, multiply by scale (gamma).
        When the next layer is linear (also e.g. nn.relu), this can be disabled
        since the scaling will be done by the next layer.
      bias_init: initializer for bias, by default, zero.
      scale_init: initializer for scale, by default, one.
      axis_name: the axis name used to combine batch statistics from multiple
        devices. See `jax.pmap` for a description of axis names (default: None).
      axis_index_groups: groups of axis indices within that named axis
        representing subsets of devices to reduce over (default: None). For
        example, `[[0, 1], [2, 3]]` would independently batch-normalize over
        the examples on the first two and last two devices. See `jax.lax.psum`
        for more details.
    """
    divider: int = 4
    use_running_average: bool = True
    axis: int = -1
    momentum: float = 0.99
    epsilon: float = 1e-5
    dtype: Dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
    axis_name: Optional[str] = None
    axis_index_groups: Any = None

    @compact
    def __call__(self, x):
        """Normalizes the input using batch statistics.
        Args:
          x: the input to be normalized.
        Returns:
          Normalized inputs (the same shape as inputs).
        """
        x = jnp.asarray(x, jnp.float32)
        original_shape = x.shape
        axis = self.axis if isinstance(self.axis, tuple) else (self.axis,)
        axis = _absolute_dims(x.ndim, axis)
        feature_shape = tuple(d if i in axis else 1 for i, d in enumerate(x.shape))
        reduced_feature_shape = tuple(d for i, d in enumerate(x.shape) if i in axis)
        reduction_axis = tuple(i for i in range(x.ndim) if i not in axis)

        batch_dim = x.shape[0]
        new_shape = [self.divider, int(batch_dim / self.divider)] + list(x.shape[1:])

        x = jnp.reshape(x, new_shape)
        y = vmap(partial(_GhostBatchNormImpl,
                         use_running_average=self.use_running_average,
                         axis=axis,
                         momentum=self.momentum,
                         epsilon=self.epsilon,
                         axis_name=self.axis_name,
                         axis_index_groups=self.axis_index_groups),
                 (0))(x)

        y = jnp.reshape(y, original_shape)
        if self.use_scale:
            scale = self.param('scale',
                               self.scale_init,
                               reduced_feature_shape).reshape(feature_shape)
            y = y * scale
        if self.use_bias:
            bias = self.param('bias',
                              self.bias_init,
                              reduced_feature_shape).reshape(feature_shape)
            y = y + bias
        return jnp.asarray(y, self.dtype)
