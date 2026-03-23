# Portions of this code are adapted from Google's learned_optimization repository
# (https://github.com/google/learned_optimization), which is licensed under the
# Apache License, Version 2.0. You may obtain a copy of the License at:
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Original copyright (c) 2021 Google LLC.
# Modifications copyright (c) 2025 Abhinav Moudgil.

"""
Self-contained Celo2 optax implementation.

Paper: https://arxiv.org/abs/2602.19142
"""

import collections
import functools

import flax
import haiku as hk
import jax
from jax import lax
import jax.numpy as jnp
import numpy as onp
import optax

ArrayTree = dict[str, jnp.ndarray] | tuple | list


# =============================================================================
# Shared utilities
# =============================================================================

def factored_dims(shape) -> tuple[int, int] | None:
    """Whether to use a factored second moment estimator."""
    if len(shape) < 2:
        return None
    sorted_dims = onp.argsort(shape)
    return int(sorted_dims[-2]), int(sorted_dims[-1])


def _safe_rsqrt(x):
    return lax.rsqrt(jnp.maximum(x, 1e-9))


def _second_moment_normalizer(x, axis, eps=1e-9):
    rms = jnp.mean(jnp.square(x), axis=axis, keepdims=True)
    rsqrt = jax.lax.rsqrt(eps + rms)
    return x * rsqrt


def orthogonalize_via_newton_schulz(
    x: jax.Array,
    ns_coeffs: jax.Array,
    ns_steps: int = 5,
    eps: float = 1e-8,
) -> jax.Array:
    """Newton-Schulz orthogonalization."""
    if x.ndim < 2:
        raise ValueError(f'Input must have >= 2 dims, got {x.shape}')
    if ns_coeffs.shape != (3,):
        raise ValueError(f'ns_coeffs must have shape (3,), got {ns_coeffs}')

    def newton_schulz_iterator(x: jax.Array, coeffs: jax.Array) -> jax.Array:
        x_mT = jnp.swapaxes(x, -2, -1)
        a = x @ x_mT
        b = coeffs[1] * a + coeffs[2] * a @ a
        return coeffs[0] * x + b @ x

    transposed = False
    if x.shape[-2] > x.shape[-1]:
        x = jnp.swapaxes(x, -2, -1)
        transposed = True
    x /= (jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + eps)
    ns_coeffs = ns_coeffs.astype(x.dtype)
    x = jax.lax.fori_loop(
        0, ns_steps,
        lambda _, x: newton_schulz_iterator(x, ns_coeffs), x,
    )
    if transposed:
        x = jnp.swapaxes(x, -2, -1)
    return x


def load_checkpoint(path: str, **config) -> dict:
    """Load meta-trained Celo2 parameters from a local checkpoint file.

    Args:
        path: Path to a flax-serialized checkpoint file.
        **config: Configuration kwargs forwarded to Celo2Transformation
            (needed to reconstruct the correct parameter tree structure).
            Defaults match celo2-base; override as needed for other variants.

    Returns:
        Theta dict suitable for passing to scale_by_celo2().
    """
    ref = Celo2Transformation(**config).init_meta_params(jax.random.PRNGKey(0))
    with open(path, 'rb') as f:
        return flax.serialization.from_bytes(ref, f.read())


def vec_bias_correction(m: ArrayTree, betas: jnp.ndarray, iteration: jnp.ndarray) -> ArrayTree:
    """Apply bias correction to momentum values across multiple betas."""
    betas = jnp.asarray(betas)
    return jax.vmap(
        lambda m_beta, beta: optax.tree.bias_correction(m_beta, beta, iteration),
        in_axes=(-1, 0),
        out_axes=-1,
    )(m, betas)


# =============================================================================
# Accumulator primitives (from learned_optimization.learned_optimizers.common)
# =============================================================================

MomAccumulator = collections.namedtuple("MomAccumulator", ["m", "t"])
RMSAccumulator = collections.namedtuple("RMSAccumulator", ["rms", "t"])
_InitUpdate = collections.namedtuple("_InitUpdate", ["init", "update"])


@flax.struct.dataclass
class FactoredAccum:
    v_col: jnp.ndarray
    v_row: jnp.ndarray
    v_diag: jnp.ndarray


def rolling_mom(decay: float) -> _InitUpdate:
    """Accumulator to keep track of momentum."""
    def init_fn(p: ArrayTree) -> MomAccumulator:
        return MomAccumulator(
            m=jax.tree_util.tree_map(jnp.zeros_like, p),
            t=jnp.asarray(0, dtype=jnp.int32),
        )

    def update_fn(state: MomAccumulator, grad: ArrayTree) -> MomAccumulator:
        m = jax.tree_util.tree_map(
            lambda a, b: decay * a + (1 - decay) * b, state.m, grad,
        )
        return MomAccumulator(m=m, t=state.t + 1)

    return _InitUpdate(init_fn, update_fn)


def rolling_rms(decay: float) -> _InitUpdate:
    """Accumulator to keep track of second moment accumulators."""
    def init_fn(p: ArrayTree) -> RMSAccumulator:
        return RMSAccumulator(
            rms=jax.tree_util.tree_map(jnp.zeros_like, p),
            t=jnp.asarray(0, dtype=jnp.int32),
        )

    def update_fn(state: RMSAccumulator, grad: ArrayTree) -> RMSAccumulator:
        clip_decay = jnp.clip(decay, 0.0, 1.0)
        rms = jax.tree_util.tree_map(
            lambda a, b: clip_decay * a + (1 - clip_decay) * (b * b),
            state.rms, grad,
        )
        return RMSAccumulator(rms=rms, t=state.t + 1)

    return _InitUpdate(init_fn, update_fn)


def factored_rolling(decay_rate: float, epsilon: float = 1e-30) -> _InitUpdate:
    """Gradient statistics accumulator based on factored gradients (AdaFactor-style)."""

    def init_fn(params: ArrayTree) -> FactoredAccum:
        def _init_one(param):
            shape = param.shape
            f_dims = factored_dims(shape)
            if f_dims is not None:
                d1, d0 = f_dims
                vr_shape = onp.delete(shape, d0)
                vc_shape = onp.delete(shape, d1)
                v_row = jnp.zeros(vr_shape, dtype=jnp.float32)
                v_col = jnp.zeros(vc_shape, dtype=jnp.float32)
                return v_row, v_col, jnp.asarray([], dtype=jnp.float32)
            else:
                v = jnp.zeros(param.shape, dtype=jnp.float32)
                return (
                    jnp.asarray([], dtype=jnp.float32),
                    jnp.asarray([], dtype=jnp.float32),
                    v,
                )

        leaves, tree = jax.tree_util.tree_flatten(params)
        v_rows, v_cols, v_fulls = zip(*[_init_one(l) for l in leaves])
        return FactoredAccum(
            v_row=jax.tree_util.tree_unflatten(tree, v_rows),
            v_col=jax.tree_util.tree_unflatten(tree, v_cols),
            v_diag=jax.tree_util.tree_unflatten(tree, v_fulls),
        )

    def update_fn(state: FactoredAccum, grad: ArrayTree) -> tuple[FactoredAccum, ArrayTree]:
        def update_one(v_col, v_row, v_full, g):
            clip_decay_rate = jnp.clip(decay_rate, 0.0, 1.0)
            mixing_rate = 1.0 - clip_decay_rate
            grad_sqr = g * g + epsilon
            f_dims = factored_dims(g.shape)

            if f_dims is not None:
                d1, d0 = f_dims
                new_v_row = clip_decay_rate * v_row + mixing_rate * jnp.mean(grad_sqr, axis=d0)
                new_v_col = clip_decay_rate * v_col + mixing_rate * jnp.mean(grad_sqr, axis=d1)
                reduced_d1 = d1 - 1 if d1 > d0 else d1
                row_col_mean = jnp.mean(new_v_row, axis=reduced_d1, keepdims=True)
                row_factor = _safe_rsqrt(new_v_row / (row_col_mean + 1e-9))
                col_factor = _safe_rsqrt(new_v_col)
                y = (
                    g
                    * jnp.expand_dims(row_factor, axis=d0)
                    * jnp.expand_dims(col_factor, axis=d1)
                )
                return new_v_col, new_v_row, jnp.asarray([], jnp.float32), y
            else:
                new_v = clip_decay_rate * v_full + mixing_rate * grad_sqr
                y = g * _safe_rsqrt(new_v + 1e-9)
                return jnp.asarray([], jnp.float32), jnp.asarray([], jnp.float32), new_v, y

        f_v_col, tree = jax.tree_util.tree_flatten(state.v_col)
        f_v_row = jax.tree_util.tree_leaves(state.v_row)
        f_v = jax.tree_util.tree_leaves(state.v_diag)
        f_g = jax.tree_util.tree_leaves(grad)
        assert len(f_g) == len(f_v_col)
        assert len(f_g) == len(f_v)
        assert len(f_g) == len(f_v_row)
        f_v_col, f_v_row, f_v, outs = zip(
            *[update_one(*args) for args in zip(f_v_col, f_v_row, f_v, f_g)]
        )

        next_state = FactoredAccum(
            v_col=jax.tree_util.tree_unflatten(tree, f_v_col),
            v_row=jax.tree_util.tree_unflatten(tree, f_v_row),
            v_diag=jax.tree_util.tree_unflatten(tree, f_v),
        )
        return next_state, jax.tree_util.tree_unflatten(tree, outs)

    return _InitUpdate(init_fn, update_fn)


def _vmap_accumulator(accumulator, decays: jnp.ndarray) -> _InitUpdate:
    """Vmaps an accumulator fn to run on multiple decays."""
    def init_fn(p):
        return jax.vmap(lambda d: accumulator(d).init(p), out_axes=-1)(decays)

    def update(state, grads):
        return jax.vmap(
            lambda s, d: accumulator(d).update(s, grads),
            in_axes=-1, out_axes=-1,
        )(state, decays)

    return _InitUpdate(init=init_fn, update=update)


def vec_rolling_mom(decays: jnp.ndarray) -> _InitUpdate:
    """Vectorized accumulator for multiple momentum decays."""
    return _vmap_accumulator(rolling_mom, decays)


def vec_rolling_rms(decays: jnp.ndarray) -> _InitUpdate:
    """Vectorized accumulator for multiple second moment decays."""
    return _vmap_accumulator(rolling_rms, decays)


def vec_factored_rolling(decays: jnp.ndarray) -> _InitUpdate:
    """Vectorized accumulator for factored accumulators."""
    return _vmap_accumulator(factored_rolling, decays)


# =============================================================================
# Celo2 optax transformation
# =============================================================================

@flax.struct.dataclass
class Celo2State:
    """Optax state for Celo2."""
    rms_rolling: ArrayTree
    mom_rolling: ArrayTree
    fac_rolling: ArrayTree
    iteration: jnp.ndarray


class Celo2Transformation:
    """
    Self-contained optax-compatible Celo2 optimizer with MLP forward pass.

    Default args refer to Celo2 optimizer.
    For Celo2-base, set orthogonalize=False.
    """

    def __init__(
        self,
        theta=None,
        ff_hidden_size=8,
        ff_hidden_layers=2,
        initial_momentum_decays=(0.9, 0.99, 0.999),
        initial_rms_decays=(0.95,),
        initial_adafactor_decays=(0.9, 0.99, 0.999),
        exp_mult=0.0,
        rmsmult=1.0,
        with_g=True,
        with_m=True,
        with_rms=True,
        with_rms_norm_g=True,
        with_rsqrt_rms=True,
        with_p=True,
        with_fac_norm_g=True,
        with_fac_rms=True,
        with_fac_rsqrt=True,
        with_grad_clip_feat=True,
        with_fac_mom_mult=True,
        with_rms_only_norm_g=True,
        param_scale_mult=False,
        precondition_output=False,
        normalize_input=True,
        normalize_output=True,
        aggregate_mag=False,
        bias_correction=False,
        mlp_activation="relu",
        orthogonalize=True,
        ns_coeffs=(3.4445, -4.7750, 2.0315),
        ns_iters=5,
        ns_eps=1e-8,
    ):
        self.theta = theta
        self.ff_hidden_size = ff_hidden_size
        self.ff_hidden_layers = ff_hidden_layers
        self.initial_momentum_decays = initial_momentum_decays
        self.initial_rms_decays = initial_rms_decays
        self.initial_adafactor_decays = initial_adafactor_decays
        self.exp_mult = exp_mult
        self.rmsmult = rmsmult

        self.with_g = with_g
        self.with_m = with_m
        self.with_rms = with_rms
        self.with_rms_norm_g = with_rms_norm_g
        self.with_rsqrt_rms = with_rsqrt_rms
        self.with_p = with_p
        self.with_fac_norm_g = with_fac_norm_g
        self.with_fac_rms = with_fac_rms
        self.with_fac_rsqrt = with_fac_rsqrt
        self.with_grad_clip_feat = with_grad_clip_feat
        self.with_fac_mom_mult = with_fac_mom_mult
        self.with_rms_only_norm_g = with_rms_only_norm_g

        self.param_scale_mult = param_scale_mult
        self.precondition_output = precondition_output
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        self.aggregate_mag = aggregate_mag
        self.bias_correction = bias_correction
        self.mlp_activation = mlp_activation

        self.orthogonalize = orthogonalize
        self.ns_coeffs = jnp.asarray(ns_coeffs)
        self.ns_iters = ns_iters
        self.ns_eps = ns_eps

        if self.mlp_activation == 'relu':
            self.act_fn = jax.nn.relu
        elif self.mlp_activation == 'tanh':
            self.act_fn = jax.nn.tanh
        else:
            raise ValueError(f"Invalid MLP activation: {self.mlp_activation}")

        self.ff_mod = hk.without_apply_rng(hk.transform(self._ff_mod))

    def _ff_mod(self, p, g, m, rms, fac_g, fac_vec_col, fac_vec_row):
        """MLP for 2D+ parameters - WITH adafactor features."""
        if len(p.shape) == 0:
            p = jnp.expand_dims(p, 0)
            g = jnp.expand_dims(g, 0)
            m = jnp.expand_dims(m, 0)
            rms = jnp.expand_dims(rms, 0)
            fac_g = jnp.expand_dims(fac_g, 0)
            fac_vec_col = jnp.expand_dims(fac_vec_col, 0)
            fac_vec_row = jnp.expand_dims(fac_vec_row, 0)
            did_reshape = True
        else:
            did_reshape = False

        inps = []

        if self.with_g:
            inps.append(jnp.expand_dims(g, axis=-1))
        if self.with_grad_clip_feat:
            inps.append(jnp.expand_dims(jnp.clip(g, -0.1, 0.1), axis=-1))
        if self.with_p:
            inps.append(jnp.expand_dims(p, axis=-1))
        if self.with_m:
            inps.append(m)
        if self.with_rms:
            inps.append(rms)

        rsqrt = lax.rsqrt(rms + 1e-8)
        if self.with_rms_norm_g:
            inps.append(m * rsqrt)
        if self.with_rsqrt_rms:
            inps.append(rsqrt)
        if self.with_fac_norm_g:
            inps.append(fac_g)
        if self.with_rms_only_norm_g:
            inps.append(jnp.expand_dims(g, axis=-1) * rsqrt)

        factored_dim = factored_dims(g.shape)
        if factored_dim is not None:
            d1, d0 = factored_dim

            to_tile = [1] * (1 + len(g.shape))
            to_tile[d0] = g.shape[d0]
            row_feat = jnp.tile(jnp.expand_dims(fac_vec_row, axis=d0), to_tile)

            to_tile = [1] * (1 + len(g.shape))
            to_tile[d1] = g.shape[d1]
            col_feat = jnp.tile(jnp.expand_dims(fac_vec_col, axis=d1), to_tile)

            if self.with_fac_rms:
                inps.append(row_feat)
                inps.append(col_feat)
            if self.with_fac_rsqrt:
                inps.append(lax.rsqrt(row_feat + 1e-8))
                inps.append(lax.rsqrt(col_feat + 1e-8))

            if self.with_fac_mom_mult:
                reduced_d1 = d1 - 1 if d1 > d0 else d1
                row_col_mean = jnp.mean(fac_vec_row, axis=reduced_d1, keepdims=True)
                row_factor = _safe_rsqrt(fac_vec_row / (row_col_mean + 1e-9))
                col_factor = _safe_rsqrt(fac_vec_col)
                fac_mom_mult = (
                    m
                    * jnp.expand_dims(row_factor, axis=d0)
                    * jnp.expand_dims(col_factor, axis=d1)
                )
                inps.append(fac_mom_mult)

        last_size = sum([i.shape[-1] for i in inps])
        weights, biases = [], []

        for wi, w in enumerate([self.ff_hidden_size] * self.ff_hidden_layers + [3]):
            stddev = 1.0 / onp.sqrt(last_size)
            w_init = hk.initializers.TruncatedNormal(stddev=stddev)
            if wi == 0:
                w1 = []
                for ii, i in enumerate(inps):
                    w1.append(hk.get_parameter(
                        f"w{wi}__{ii}",
                        shape=(i.shape[-1], w), dtype=jnp.float32, init=w_init,
                    ))
                weights.append(w1)
            else:
                weights.append(hk.get_parameter(
                    f"w{wi}",
                    shape=(last_size, w), dtype=jnp.float32, init=w_init,
                ))
            biases.append(hk.get_parameter(
                f"b{wi}", shape=(w,), dtype=jnp.float32, init=jnp.zeros,
            ))
            last_size = w

        axis = list(range(len(p.shape)))[-2:]
        if self.normalize_input:
            inp_stack = [_second_moment_normalizer(i, axis=axis) for i in inps]
        else:
            inp_stack = inps

        o = inp_stack
        for wi, (w, b) in enumerate(zip(weights, biases)):
            if wi == 0:
                o_tmp = jnp.zeros(o[0].shape[:-1] + w[0].shape[1:])
                for oi, oo in enumerate(o):
                    o_tmp = o_tmp + oo @ w[oi]
            else:
                o_tmp = o @ w
            o = o_tmp + jnp.broadcast_to(b, list(o_tmp.shape[0:-1]) + [o_tmp.shape[-1]])
            if wi != len(weights) - 1:
                o = self.act_fn(o)

        direction = o[..., 0]
        magnitude_param = o[..., 1]
        if self.aggregate_mag:
            magnitude_param = jnp.mean(magnitude_param)

        mag_param = jnp.exp(magnitude_param * self.exp_mult)
        param_scale = jnp.sqrt(jnp.mean(jnp.square(p)) + 1e-9)

        if self.param_scale_mult:
            step = direction * (param_scale * mag_param)
        else:
            step = direction * mag_param

        if self.orthogonalize and len(step.shape) >= 2:
            step = orthogonalize_via_newton_schulz(
                step, self.ns_coeffs, self.ns_iters, self.ns_eps,
            )

        if self.normalize_output:
            step = _second_moment_normalizer(step, axis=axis)

        step = step * self.rmsmult

        step = step.reshape(p.shape)

        if self.precondition_output:
            norms = rms[..., -1]
            step = step * lax.rsqrt(norms + 1e-6)

        if did_reshape:
            step = jnp.squeeze(step, 0)

        return step

    def accumulators_for_decays(self):
        mom_decay = jnp.asarray(self.initial_momentum_decays)
        rms_decay = jnp.asarray(self.initial_rms_decays)
        adafactor_decay = jnp.asarray(self.initial_adafactor_decays)
        mom_roll = vec_rolling_mom(mom_decay)
        rms_roll = vec_rolling_rms(rms_decay)
        fac_vec_roll = vec_factored_rolling(adafactor_decay)
        return mom_roll, rms_roll, fac_vec_roll

    def init_meta_params(self, key) -> dict:
        """Initialize the learned MLP parameters (theta)."""
        r, c = 10, 10
        p = jnp.ones([r, c])
        g = jnp.ones([r, c])
        m = jnp.ones([r, c, len(self.initial_momentum_decays)])
        rms = jnp.ones([r, c, len(self.initial_rms_decays)])
        fac_g = jnp.ones([r, c, len(self.initial_adafactor_decays)])
        fac_vec_row = jnp.ones([r, len(self.initial_adafactor_decays)])
        fac_vec_col = jnp.ones([c, len(self.initial_adafactor_decays)])

        key1, key = jax.random.split(key)
        theta = self.ff_mod.init(key1, p, g, m, rms, fac_g, fac_vec_col, fac_vec_row)
        return {"ff_mod_stack": theta}

    def init(self, params: ArrayTree) -> Celo2State:
        """Initialize optimizer state (accumulators)."""
        mom_roll, rms_roll, fac_roll = self.accumulators_for_decays()
        return Celo2State(
            rms_rolling=rms_roll.init(params),
            mom_rolling=mom_roll.init(params),
            fac_rolling=fac_roll.init(params),
            iteration=jnp.asarray(0, dtype=jnp.int32),
        )

    def update(self, grads: ArrayTree, state: Celo2State, params: ArrayTree = None) -> tuple:
        """Returns (step, new_state). Step is the raw MLP update direction."""
        iteration = optax.safe_increment(state.iteration)
        grads = jax.tree_util.tree_map(
            lambda x: jnp.clip(x, -1000.0, 1000.0), grads,
        )

        mom_roll, rms_roll, fac_roll = self.accumulators_for_decays()
        next_mom_rolling = mom_roll.update(state.mom_rolling, grads)
        next_rms_rolling = rms_roll.update(state.rms_rolling, grads)
        next_fac_rolling, fac_g = fac_roll.update(state.fac_rolling, grads)

        if self.bias_correction:
            m = vec_bias_correction(
                next_mom_rolling.m, self.initial_momentum_decays, iteration,
            )
            rms = vec_bias_correction(
                next_rms_rolling.rms, self.initial_rms_decays, iteration,
            )
            v_col = vec_bias_correction(
                next_fac_rolling.v_col, self.initial_adafactor_decays, iteration,
            )
            v_row = vec_bias_correction(
                next_fac_rolling.v_row, self.initial_adafactor_decays, iteration,
            )
        else:
            m = next_mom_rolling.m
            rms = next_rms_rolling.rms
            v_col = next_fac_rolling.v_col
            v_row = next_fac_rolling.v_row

        apply_partial = functools.partial(
            self.ff_mod.apply, self.theta["ff_mod_stack"],
        )
        step = jax.tree_util.tree_map(
            apply_partial, params, grads, m, rms, fac_g, v_col, v_row,
        )

        new_state = Celo2State(
            mom_rolling=next_mom_rolling,
            rms_rolling=next_rms_rolling,
            fac_rolling=next_fac_rolling,
            iteration=iteration,
        )
        return step, new_state


# =============================================================================
# Convenience function: wrap transformations as optax.GradientTransformation
# =============================================================================

def scale_by_celo2(theta: dict, **config) -> optax.GradientTransformation:
    """Create an optax GradientTransformation for Celo2.

    Args:
        theta: Meta parameters from Celo2Transformation.init_meta_params().
        **config: Configuration kwargs forwarded to Celo2Transformation.
    """
    transformation = Celo2Transformation(theta=theta, **config)

    def init_fn(params):
        return transformation.init(params)

    def update_fn(updates, state, params=None, **kwargs):
        return transformation.update(updates, state, params)

    return optax.GradientTransformation(init_fn, update_fn)
