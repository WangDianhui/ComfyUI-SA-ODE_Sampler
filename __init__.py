# __init__.py

from .samplers import (
    sample_sa_ode_lowstep,
    sample_sa_ode_stable,
    lowstep_sigma_scheduler,
)
from comfy.samplers import (
    KSAMPLER, KSAMPLER_NAMES, SAMPLER_NAMES,
    SchedulerHandler, SCHEDULER_HANDLERS, SCHEDULER_NAMES
)
import comfy.k_diffusion.sampling


# ==============================
# 注册采样函数到 comfy.k_diffusion.sampling
# ==============================
comfy.k_diffusion.sampling.sample_sa_ode_lowstep = sample_sa_ode_lowstep
comfy.k_diffusion.sampling.sample_sa_ode_stable = sample_sa_ode_stable


# ==============================
# 注册采样器名称
# ==============================
def safe_append(lst, item):
    if item not in lst:
        lst.append(item)

safe_append(KSAMPLER_NAMES, "sa_ode_lowstep")
safe_append(SAMPLER_NAMES, "sa_ode_lowstep")

safe_append(KSAMPLER_NAMES, "sa_ode_stable")
safe_append(SAMPLER_NAMES, "sa_ode_stable")


# ==============================
# 注册 LowStep 调度器
# ==============================
LOWSTEP_SCHEDULER_NAME = "lowstep"
if LOWSTEP_SCHEDULER_NAME not in SCHEDULER_HANDLERS:
    handler = SchedulerHandler(handler=lowstep_sigma_scheduler, use_ms=True)
    SCHEDULER_HANDLERS[LOWSTEP_SCHEDULER_NAME] = handler
    SCHEDULER_NAMES.append(LOWSTEP_SCHEDULER_NAME)


# ==============================
# 节点类定义
# ==============================
class SAODELowStepSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "use_taylor_restoration": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom"
    FUNCTION = "get_sampler"

    def get_sampler(self, use_taylor_restoration):
        def wrapper(model, x, sigmas, extra_args=None, callback=None, disable=None):
            return sample_sa_ode_lowstep(
                model=model,
                x=x,
                sigmas=sigmas,
                extra_args=extra_args,
                callback=callback,
                disable=disable,
                use_taylor_restoration=use_taylor_restoration,
            )
        return (KSAMPLER(wrapper),)


class LowStepSigmaScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 6, "min": 1, "max": 20}),
                "shift": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"
    FUNCTION = "get_sigmas"

    def get_sigmas(self, model, steps, shift):
        model_sampling = model.get_model_object("model_sampling")
        device = model_sampling.sigmas.device if hasattr(model_sampling.sigmas, 'device') else 'cpu'
        sigmas = lowstep_sigma_scheduler(model_sampling, steps, shift)
        return (sigmas.to(device),)


class SAODEStableSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "solver_order": ("INT", {"default": 3, "min": 1, "max": 5}),
                "use_adaptive_order": ("BOOLEAN", {"default": True}),
                "use_velocity_smoothing": ("BOOLEAN", {"default": False}),
                "convergence_threshold": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 0.5, "step": 0.01}),
                "smoothing_factor": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom"
    FUNCTION = "get_sampler"

    def get_sampler(self, solver_order, use_adaptive_order, use_velocity_smoothing,
                    convergence_threshold, smoothing_factor):
        def wrapper(model, x, sigmas, extra_args=None, callback=None, disable=None):
            return sample_sa_ode_stable(
                model=model,
                x=x,
                sigmas=sigmas,
                extra_args=extra_args,
                callback=callback,
                disable=disable,
                solver_order=solver_order,
                use_adaptive_order=use_adaptive_order,
                use_velocity_smoothing=use_velocity_smoothing,
                convergence_threshold=convergence_threshold,
                smoothing_factor=smoothing_factor,
            )
        return (KSAMPLER(wrapper),)


# ==============================
# NODES MAPPING
# ==============================
NODE_CLASS_MAPPINGS = {
    "SAODELowStepSampler": SAODELowStepSampler,
    "LowStepSigmaScheduler": LowStepSigmaScheduler,
    "SAODEStableSampler": SAODEStableSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SAODELowStepSampler": "SA-ODE LowStep Sampler",
    "LowStepSigmaScheduler": "LowStep Sigma Scheduler",
    "SAODEStableSampler": "SA-ODE Stable Sampler",
}