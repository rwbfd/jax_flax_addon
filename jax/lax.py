# coding = 'utf-8'
import jax
import jax.numpy as jnp


@jax.partial(jax.jit, static_argnums=(1, 2,))
def swapaxes(input, dim_a, dim_b):
    return jnp.swapaxes(input, dim_a, dim_b)
