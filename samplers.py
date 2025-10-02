# samplers.py

import torch
from tqdm.auto import trange
from comfy.samplers import KSAMPLER, KSAMPLER_NAMES, SAMPLER_NAMES, SchedulerHandler, SCHEDULER_HANDLERS, SCHEDULER_NAMES
import comfy.k_diffusion.sampling


# ==============================
# LowStep Sampler (sa_ode_lowstep)
# ==============================
def sample_sa_ode_lowstep(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    use_taylor_restoration=False,
):
    extra_args = {} if extra_args is None else extra_args
    num_steps = len(sigmas) - 1
    velocity_buffer = []

    for i in trange(num_steps, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_batch = sigma.expand(x.size(0))
        x0 = model(x, sigma_batch, **extra_args)

        if sigma_next == 0:
            if use_taylor_restoration and len(velocity_buffer) >= 3:
                v0, v1, v2 = velocity_buffer[-1], velocity_buffer[-2], velocity_buffer[-3]
                dt = -sigma
                dt2 = dt * dt
                dt3 = dt2 * dt
                x = x0 + v0 * dt + v1 * (dt2 * 0.5) + v2 * (dt3 / 6.0)
            else:
                x = x0
        else:
            sigma_safe = sigma.clamp(min=1e-8)
            velocity = (x - x0) / sigma_safe
            dt = sigma_next - sigma
            x = x + velocity * dt
            velocity_buffer.append(velocity)
            if len(velocity_buffer) > 3:
                velocity_buffer.pop(0)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': x0})

    return x

# ==============================
# LowStep Scheduler
# ==============================

def lowstep_sigma_scheduler(model_sampling, steps: int, shift: float = 3.0) -> torch.Tensor:
    if steps <= 0:
        return torch.FloatTensor([1.0, 0.0])
    t = torch.linspace(0, 1, steps + 1, dtype=torch.float32)
    sigmas = 1.0 - t
    sigmas = shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
    sigmas[-1] = 0.0
    return sigmas


# ==============================
# Stable SA-ODE Sampler (sa_ode_stable)
# ==============================
def sample_sa_ode_stable(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    disable=None,
    solver_order=3,
    use_adaptive_order=True,
    use_velocity_smoothing=False,
    convergence_threshold=0.15,
    smoothing_factor=0.7,
):
    extra_args = {} if extra_args is None else extra_args
    velocity_buffer = []
    smoothed_velocity = None
    num_inference_steps = len(sigmas) - 1

    def get_adaptive_order(sigma_val):
        if not use_adaptive_order:
            return solver_order
        if num_inference_steps <= 8:
            return min(2, solver_order)
        if sigma_val > 0.7:
            return min(2, solver_order)
        elif sigma_val > convergence_threshold:
            return solver_order
        else:
            return max(1, solver_order - 1)

    def compute_multistep_velocity(order):
        if len(velocity_buffer) < order:
            order = len(velocity_buffer)
        if order >= 3:
            v = (23/12) * velocity_buffer[-1] - (16/12) * velocity_buffer[-2] + (5/12) * velocity_buffer[-3]
        elif order >= 2:
            v = 1.5 * velocity_buffer[-1] - 0.5 * velocity_buffer[-2]
        else:
            v = velocity_buffer[-1]
        return v

    for i in trange(len(sigmas) - 1, disable=disable):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]

        if sigma_next == 0:
            sigma_batch = sigma.expand(x.size(0))
            x0 = model(x, sigma_batch, **extra_args)
            x = x0
            if callback is not None:
                callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': x0})
            break

        sigma_batch = sigma.expand(x.size(0))
        x0 = model(x, sigma_batch, **extra_args)
        sigma_safe = sigma.clamp(min=1e-8)
        velocity_raw = (x - x0) / sigma_safe.view(-1, 1, 1, 1)

        velocity_buffer.append(velocity_raw)
        if len(velocity_buffer) > solver_order + 1:
            velocity_buffer.pop(0)

        sigma_val = sigma.item()
        current_order = get_adaptive_order(sigma_val)

        if len(velocity_buffer) >= 2:
            velocity = compute_multistep_velocity(current_order)
        else:
            velocity = velocity_raw

        if use_velocity_smoothing and num_inference_steps > 8:
            if 1e-3 <= sigma_val < convergence_threshold:
                dynamic_alpha = smoothing_factor * (sigma_val / convergence_threshold)
                if smoothed_velocity is None:
                    smoothed_velocity = velocity
                else:
                    smoothed_velocity = dynamic_alpha * smoothed_velocity + (1 - dynamic_alpha) * velocity
                velocity = smoothed_velocity
            else:
                smoothed_velocity = velocity
        else:
            smoothed_velocity = velocity

        dt = sigma_next - sigma

        if num_inference_steps > 8 and sigma_val < convergence_threshold and sigma_val > 1e-3:
            damping = 0.5 + 0.5 * (sigma_val / convergence_threshold)
            dt = dt * damping

        x = x + velocity * dt

        if num_inference_steps > 8 and sigma_val < 0.05 and sigma_val > 1e-3 and len(velocity_buffer) >= 3:
            avg_v = sum(velocity_buffer[-3:]) / 3
            stabilized = x - velocity * dt + avg_v * dt
            blend = sigma_val / 0.05
            x = blend * x + (1 - blend) * stabilized

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigma, 'sigma_hat': sigma, 'denoised': x0})

    return x