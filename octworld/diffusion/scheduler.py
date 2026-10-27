import torch
from typing_extensions import Literal


def _sigma_to_alpha_sigma_t(sigma):
    """Linear flow schedule: alpha_t = 1 - sigma, sigma_t = sigma."""
    return 1 - sigma, sigma


def uni_p_bh_update(sample, order, *,
                    model_outputs, sigmas, step_index, solver_type):
    """UniPC predictor (B(h) version), predict_x0=True.

    Pure function: all scheduler state is passed explicitly so the math stays
    independent of the class. ``model_outputs`` is the history of converted x0
    predictions (most recent last).
    """
    m0 = model_outputs[-1]
    x = sample
    device = sample.device

    sigma_t = sigmas[step_index + 1].to(device)
    sigma_s0 = sigmas[step_index].to(device)
    alpha_t, sigma_t = _sigma_to_alpha_sigma_t(sigma_t)
    alpha_s0, sigma_s0 = _sigma_to_alpha_sigma_t(sigma_s0)

    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
    h = lambda_t - lambda_s0

    rks = []
    D1s = []
    for i in range(1, order):
        si = step_index - i
        mi = model_outputs[-(i + 1)]
        sigma_si = sigmas[si].to(device)
        alpha_si, sigma_si = _sigma_to_alpha_sigma_t(sigma_si)
        lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
        rk = (lambda_si - lambda_s0) / h
        rks.append(rk)
        D1s.append((mi - m0) / rk)
    rks.append(1.0)
    rks = torch.tensor(rks, device=device)

    R = []
    b = []
    hh = -h  # predict_x0
    h_phi_1 = torch.expm1(hh)
    h_phi_k = h_phi_1 / hh - 1
    factorial_i = 1
    B_h = hh if solver_type == "bh1" else torch.expm1(hh)
    for i in range(1, order + 1):
        R.append(torch.pow(rks, i - 1))
        b.append(h_phi_k * factorial_i / B_h)
        factorial_i *= i + 1
        h_phi_k = h_phi_k / hh - 1 / factorial_i
    R = torch.stack(R)
    b = torch.tensor(b, device=device)

    if len(D1s) > 0:
        D1s = torch.stack(D1s, dim=1)  # (B, K, ...)
        if order == 2:
            rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
        else:
            rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1]).to(device).to(x.dtype)
    else:
        D1s = None

    x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
    if D1s is not None:
        pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
    else:
        pred_res = 0
    x_t = x_t_ - alpha_t * B_h * pred_res
    return x_t.to(x.dtype)


def uni_c_bh_update(this_model_output, last_sample, this_sample, order, *,
                    model_outputs, sigmas, step_index, solver_type):
    """UniPC corrector (B(h) version), predict_x0=True.

    Pure function: all scheduler state is passed explicitly. ``this_model_output``
    is the converted x0 prediction at the current step.
    """
    m0 = model_outputs[-1]
    x = last_sample
    model_t = this_model_output
    device = this_sample.device

    sigma_t = sigmas[step_index].to(device)
    sigma_s0 = sigmas[step_index - 1].to(device)
    alpha_t, sigma_t = _sigma_to_alpha_sigma_t(sigma_t)
    alpha_s0, sigma_s0 = _sigma_to_alpha_sigma_t(sigma_s0)

    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
    h = lambda_t - lambda_s0

    rks = []
    D1s = []
    for i in range(1, order):
        si = step_index - (i + 1)
        mi = model_outputs[-(i + 1)]
        sigma_si = sigmas[si].to(device)
        alpha_si, sigma_si = _sigma_to_alpha_sigma_t(sigma_si)
        lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
        rk = (lambda_si - lambda_s0) / h
        rks.append(rk)
        D1s.append((mi - m0) / rk)
    rks.append(1.0)
    rks = torch.tensor(rks, device=device)

    R = []
    b = []
    hh = -h  # predict_x0
    h_phi_1 = torch.expm1(hh)
    h_phi_k = h_phi_1 / hh - 1
    factorial_i = 1
    B_h = hh if solver_type == "bh1" else torch.expm1(hh)
    for i in range(1, order + 1):
        R.append(torch.pow(rks, i - 1))
        b.append(h_phi_k * factorial_i / B_h)
        factorial_i *= i + 1
        h_phi_k = h_phi_k / hh - 1 / factorial_i
    R = torch.stack(R)
    b = torch.tensor(b, device=device)

    if len(D1s) > 0:
        D1s = torch.stack(D1s, dim=1)
    else:
        D1s = None

    if order == 1:
        rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
    else:
        rhos_c = torch.linalg.solve(R, b).to(device).to(x.dtype)

    x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
    if D1s is not None:
        corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
    else:
        corr_res = 0
    D1_t = model_t - m0
    x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
    return x_t.to(x.dtype)


class FlowMatchScheduler():

    def __init__(self, 
        template: Literal["Wan"] = "Wan",
        sample_mode : Literal["euler", "unipc"] = "euler",
        solver_order: int = 2,
        solver_type: Literal["bh1", "bh2"] = "bh2",
        lower_order_final: bool = True,
        ):
        self.sample_mode = sample_mode
        self.num_train_timesteps = 1000
        # The sigma/timestep schedule depends on BOTH the template and the
        # sample mode: UniPC needs sigma_max < 1 (see set_timesteps_unipc).
        self.set_timesteps_fn = {
            ("Wan", "euler"): self.set_timesteps_wan,
            ("Wan", "unipc"): self.set_timesteps_unipc,
        }.get((template, sample_mode), self.set_timesteps_wan)

        self.step_fn = {
            "euler": self.step_euler,
            "unipc": self.step_unipc,
        }.get(sample_mode, self.step_euler)

        self.sigmas = None
        self.timesteps = None

        # ── UniPC config + state (only used when sample_mode == "unipc") ──
        # We keep the predict_x0 / flow_prediction branch only; sigmas use the
        # linear flow convention alpha_t = 1 - sigma, sigma_t = sigma.
        self.solver_order = solver_order
        self.solver_type = solver_type
        self.lower_order_final = lower_order_final
    
    def set_timesteps_wan(self, num_inference_steps=100, denoising_strength=1.0, shift=None):
        sigma_min = 0.0
        sigma_max = 1.0
        shift = 5 if shift is None else shift
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * self.num_train_timesteps
        return sigmas, timesteps

    def set_timesteps_unipc(self, num_inference_steps=100, denoising_strength=1.0, shift=None):
        """UniPC sigma schedule (matches Wan's FlowUniPCMultistepScheduler).

        Differs from the euler schedule in two ways:
        - ``sigma_max`` is ``1 - 1/num_train_timesteps`` (≈ 0.999), not 1.0.
          UniPC works in the log-domain (``lambda = log(alpha) - log(sigma)``);
          ``sigma = 1`` would give ``alpha = 0`` and ``log(0) = -inf``.
        - ``sigmas`` has a trailing 0 appended (the sampling-endpoint sigma=0):
          the UniPC predictor reads ``sigmas[step_index + 1]`` on the last step,
          so this 0 is a real target value, not padding. unipc is inference-only,
          so ``sigmas`` being one element longer than ``timesteps`` is fine.
        Shift is applied identically to the euler schedule.
        """
        shift = 5 if shift is None else shift
        sigma_max = 1.0 - 1.0 / self.num_train_timesteps
        sigma_min = 0.0
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        timesteps = sigmas * self.num_train_timesteps
        sigmas = torch.cat([sigmas, torch.zeros(1)])
        # Start each schedule with a fresh multistep state.
        self._reset_unipc_state()
        return sigmas, timesteps

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, **kwargs):
        assert not (training and self.sample_mode == "unipc"), (
            "unipc is inference-only; use sample_mode='euler' for training."
        )
        self.sigmas, self.timesteps = self.set_timesteps_fn(
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            **kwargs,
        )
        if training:
            self.set_training_weight()
            self.training = True
        else:
            self.training = False

    def step_euler(self, model_output, timestep, sample, to_final=False):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    # ──────────────────────────────────────────────────────────────────
    # UniPC multistep solver (2nd-order by default).
    #
    # Ported from Wan 2.1's FlowUniPCMultistepScheduler, keeping only the
    # predict_x0=True / prediction_type="flow_prediction" branch. The model
    # output is the flow velocity v = eps - x0; with the linear flow schedule
    # (alpha_t = 1 - sigma, sigma_t = sigma) the x0 prediction is simply
    # x0 = sample - sigma * v, so no extra conversion is needed.
    # ──────────────────────────────────────────────────────────────────
    def _reset_unipc_state(self):
        self.model_outputs = [None] * self.solver_order
        self.last_sample = None
        self.lower_order_nums = 0
        self.this_order = None

    def step_unipc(self, model_output, timestep, sample):
        """One UniPC multistep update. Same call signature as ``step_euler``."""
        if self.sigmas is None:
            raise RuntimeError(
                "UniPC sigmas are not set; call set_timesteps(...) first."
            )
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.timesteps.device)
        step_index = int(torch.argmin((self.timesteps - timestep).abs()))
        # Auto-reset the multistep state on the first denoising step so the
        # scheduler is safe to reuse across sampling runs without an explicit
        # reset (e.g. AR rollout: set_timesteps + sampling once per chunk).
        if step_index == 0:
            self._reset_unipc_state()

        use_corrector = (
            step_index > 0
            and self.last_sample is not None
        )

        # flow velocity -> predicted x0 (predict_x0 branch).
        sigma_t = self.sigmas[step_index].to(sample)
        model_output_convert = sample - sigma_t * model_output
        if use_corrector:
            sample = uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
                model_outputs=self.model_outputs,
                sigmas=self.sigmas,
                step_index=step_index,
                solver_type=self.solver_type,
            )

        for i in range(self.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output_convert

        if self.lower_order_final:
            this_order = min(self.solver_order,
                             len(self.timesteps) - step_index)
        else:
            this_order = self.solver_order
        self.this_order = min(this_order, self.lower_order_nums + 1)
        assert self.this_order > 0

        self.last_sample = sample
        prev_sample = uni_p_bh_update(
            sample=sample,
            order=self.this_order,
            model_outputs=self.model_outputs,
            sigmas=self.sigmas,
            step_index=step_index,
            solver_type=self.solver_type,
        )

        if self.lower_order_nums < self.solver_order:
            self.lower_order_nums += 1

        return prev_sample

    def set_training_weight(self):
        assert self.sample_mode != "unipc", "training is not supported in unipc mode"
        steps = 1000
        x = self.timesteps
        y = torch.exp(-2 * ((x - steps / 2) / steps) ** 2)
        y_shifted = y - y.min()
        bsmntw_weighing = y_shifted * (steps / y_shifted.sum())
        if len(self.timesteps) != 1000:
            # This is an empirical formula.
            bsmntw_weighing = bsmntw_weighing * (len(self.timesteps) / steps)
            bsmntw_weighing = bsmntw_weighing + bsmntw_weighing[1]
        self.linear_timesteps_weights = bsmntw_weighing

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        """Look up training weight for given timestep(s) by nearest match."""
        assert self.sample_mode != "unipc", "training is not supported in unipc mode"
        self.linear_timesteps_weights = self.linear_timesteps_weights.to(timestep.device)
        self.timesteps = self.timesteps.to(timestep.device)
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(1) - timestep.unsqueeze(0)).abs(), dim=0
        )
        return self.linear_timesteps_weights[timestep_id]

    def add_noise(self, original_samples, noise, timestep):
        """x_noisy = (1 - sigma) * x0 + sigma * noise"""
        assert self.sample_mode != "unipc", "training is not supported in unipc mode"
        timestep_id = torch.argmin(
            (self.timesteps.to(timestep.device).unsqueeze(1) - timestep.unsqueeze(0)).abs(), dim=0
        )
        sigma = self.sigmas.to(timestep.device)[timestep_id]
        while sigma.dim() < original_samples.dim():
            sigma = sigma.unsqueeze(-1)
        return (1 - sigma) * original_samples + sigma * noise

    def training_target(self, sample, noise, timestep):
        """velocity target: eps - x0"""
        assert self.sample_mode != "unipc", "training is not supported in unipc mode"
        return noise - sample