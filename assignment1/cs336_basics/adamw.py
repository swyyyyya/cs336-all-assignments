import torch


class AdamW(torch.optim.Optimizer):
    """AdamW optimizer (Loshchilov & Hutter, 2019), implemented from scratch.

    The update rule follows the decoupled-weight-decay formulation: the
    parameter is first shrunk towards zero by the weight decay, then updated
    with bias-corrected moment estimates.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]
                # Initialize the moment estimates on the first step.
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                # Decoupled weight decay, applied to the parameter before the
                # gradient-based update (t starts at 1 in the algorithm).
                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                # Update the first and second moment estimates.
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias-corrected step: alpha_t * m_hat / (sqrt(v_hat) + eps).
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = group["lr"] / bias_correction1
                bias_correction2_sqrt = bias_correction2**0.5
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(
                    group["eps"]
                )
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
