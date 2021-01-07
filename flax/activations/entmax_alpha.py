import jax.numpy as jnp
from jax import grad, jit, value_and_grad
from jax import vmap, pmap
from jax import random
import jax
from jax import lax
from jax import custom_jvp


def p_tau(z, tau, alpha=1.5):
    return jnp.clip((alpha - 1) * z - tau, a_min=0) ** (1 / (alpha - 1))


def get_tau(tau, tau_max, tau_min, z_value):
    return lax.cond(z_value < 1,
                    lambda _: (tau, tau_min),
                    lambda _: (tau_max, tau),
                    operand=None
                    )


def body(kwargs, x):
    tau_min = kwargs['tau_min']
    tau_max = kwargs['tau_max']
    z = kwargs['z']
    alpha = kwargs['alpha']

    tau = (tau_min + tau_max) / 2
    z_value = p_tau(z, tau, alpha).sum()
    taus = get_tau(tau, tau_max, tau_min, z_value)
    tau_max, tau_min = taus[0], taus[1]
    return {'tau_min': tau_min, 'tau_max': tau_max, 'z': z, 'alpha': alpha}, None


def map_row(z_input, alpha, T):
    z = (alpha - 1) * z_input

    tau_min, tau_max = jnp.min(z) - 1, jnp.max(z) - z.shape[0] ** (1 - alpha)
    result, _ = lax.scan(body, {'tau_min': tau_min, 'tau_max': tau_max, 'z': z, 'alpha': alpha}, xs=None,
                         length=T)
    tau = (result['tau_max'] + result['tau_min']) / 2
    result = p_tau(z, tau, alpha)
    return result / result.sum()


def _entmax(input, axis=-1, alpha=1.5, T=20):
    result = vmap(jax.partial(map_row, alpha=alpha, T=T), axis)(input)
    return result


@jax.partial(custom_jvp, nondiff_argnums=(1, 2, 3,))
def entmax(input, axis=-1, alpha=1.5, T=10):
    return _entmax(input, axis, alpha, T)


def _entmax_jvp_impl(axis, alpha, T, primals, tangents):
    input = primals[0]
    Y = entmax(input, axis, alpha, T)
    gppr = Y ** (2 - alpha)
    grad_output = tangents[0]
    dX = grad_output * gppr
    q = dX.sum(axis=axis) / gppr.sum(axis=axis)
    q = jnp.expand_dims(q, axis=axis)
    dX -= q * gppr
    return Y, dX


@entmax.defjvp
def entmax_jvp(axis, alpha, T, primals, tangents):
    return _entmax_jvp_impl(axis, alpha, T, primals, tangents)
