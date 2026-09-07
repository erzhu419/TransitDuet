"""Exact weighted KL projection for finite-action policy targets."""

from __future__ import annotations

import torch


def joint_weighted_kl_projection(
        base_logits, action_costs, feasible_actions, sample_weights, limits,
        *, tolerance=1e-8, max_iterations=200, support_floor=1e-12):
    """Project categorical rows onto global weighted expected-cost limits.

    The returned distribution minimizes the sample-weighted mean
    ``KL(projected || base)``. ``action_costs`` has shape ``[C, B, A]`` and
    each of the ``C`` constraints is an expectation over the same normalized
    sample weights.
    """
    if base_logits.ndim != 2:
        raise ValueError("base_logits must have shape [batch, actions]")
    batch, actions = base_logits.shape
    if action_costs.ndim != 3 or tuple(action_costs.shape[1:]) != (
            batch, actions):
        raise ValueError("action_costs must have shape [constraints, batch, actions]")
    constraints = int(action_costs.shape[0])
    if constraints < 1:
        raise ValueError("at least one projection constraint is required")
    if feasible_actions.shape != base_logits.shape:
        raise ValueError("feasible_actions must match base_logits")
    if sample_weights.ndim != 1 or sample_weights.shape[0] != batch:
        raise ValueError("sample_weights must have shape [batch]")
    if limits.ndim != 1 or limits.shape[0] != constraints:
        raise ValueError("limits must have shape [constraints]")
    if batch < 1 or actions < 2:
        raise ValueError("projection requires nonempty categorical rows")
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("projection tolerance and iterations must be positive")
    if not 0.0 < support_floor < 1.0:
        raise ValueError("projection support_floor must lie in (0, 1)")

    device = base_logits.device
    dtype = torch.float64
    logits = base_logits.detach().to(dtype=dtype)
    costs = action_costs.detach().to(device=device, dtype=dtype)
    feasible = feasible_actions.detach().to(device=device, dtype=torch.bool)
    weights = sample_weights.detach().to(device=device, dtype=dtype)
    bounds = limits.detach().to(device=device, dtype=dtype)
    if not bool(feasible.any(dim=-1).all().item()):
        raise ValueError("every projection row needs a feasible action")
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError("base_logits must be finite")
    if not bool(torch.isfinite(costs).all().item()):
        raise ValueError("action_costs must be finite")
    if not bool(torch.isfinite(weights).all().item()) or bool(
            (weights < 0.0).any().item()):
        raise ValueError("sample_weights must be finite and nonnegative")
    if not bool(torch.isfinite(bounds).all().item()) or bool(
            (bounds < 0.0).any().item()):
        raise ValueError("projection limits must be finite and nonnegative")
    weight_sum = weights.sum()
    if not bool((weight_sum > 0.0).item()):
        raise ValueError("projection sample weights must have positive mass")
    weights = weights / weight_sum

    negative_infinity = torch.tensor(
        float("-inf"), dtype=dtype, device=device)
    masked_logits = torch.where(feasible, logits, negative_infinity)
    base_probabilities = torch.softmax(masked_logits, dim=-1)
    base_probabilities = torch.where(
        feasible,
        base_probabilities.clamp_min(float(support_floor)),
        torch.zeros_like(base_probabilities),
    )
    base_probabilities = (
        base_probabilities / base_probabilities.sum(dim=-1, keepdim=True))
    log_base = torch.where(
        feasible, base_probabilities.log(), negative_infinity)

    def projection_state(multipliers):
        tilted_logits = log_base - torch.einsum(
            "c,cba->ba", multipliers, costs)
        tilted_logits = torch.where(
            feasible, tilted_logits, negative_infinity)
        log_partition = torch.logsumexp(tilted_logits, dim=-1)
        probabilities = torch.where(
            feasible,
            torch.exp(tilted_logits - log_partition.unsqueeze(-1)),
            torch.zeros_like(tilted_logits),
        )
        per_state_cost = torch.einsum(
            "ba,cba->cb", probabilities, costs)
        expected_cost = torch.einsum("b,cb->c", weights, per_state_cost)
        second_moment = torch.einsum(
            "ba,cba,dba->bcd", probabilities, costs, costs)
        covariance = second_moment - torch.einsum(
            "cb,db->bcd", per_state_cost, per_state_cost)
        hessian = torch.einsum("b,bcd->cd", weights, covariance)
        objective = (
            torch.dot(weights, log_partition)
            + torch.dot(multipliers, bounds))
        return probabilities, expected_cost, hessian, objective

    multipliers = torch.zeros(constraints, dtype=dtype, device=device)
    converged = False
    iterations = 0
    kkt_residual = torch.tensor(float("inf"), dtype=dtype, device=device)
    for iteration in range(1, int(max_iterations) + 1):
        iterations = iteration
        probabilities, expected_cost, hessian, objective = projection_state(
            multipliers)
        gradient = bounds - expected_cost
        active_multiplier = multipliers > float(tolerance)
        residuals = torch.where(
            active_multiplier, gradient.abs(), (-gradient).clamp_min(0.0))
        kkt_residual = residuals.max()
        if bool((kkt_residual <= float(tolerance)).item()):
            converged = True
            break

        active = active_multiplier | (gradient < 0.0)
        active_indices = torch.nonzero(active, as_tuple=False).reshape(-1)
        if not active_indices.numel():
            converged = True
            break
        active_hessian = hessian.index_select(
            0, active_indices).index_select(1, active_indices)
        active_gradient = gradient.index_select(0, active_indices)
        ridge = max(
            1e-12,
            abs(float(torch.trace(active_hessian).item())) * 1e-10,
        )
        regularized = active_hessian + ridge * torch.eye(
            active_indices.numel(), dtype=dtype, device=device)
        try:
            active_direction = torch.linalg.solve(
                regularized, -active_gradient)
        except RuntimeError:
            active_direction = -active_gradient
        direction = torch.zeros_like(multipliers)
        direction[active_indices] = active_direction
        directional_derivative = torch.dot(gradient, direction)
        if (not bool(torch.isfinite(direction).all().item())
                or float(directional_derivative.item()) >= 0.0):
            direction.zero_()
            direction[active_indices] = -active_gradient
            directional_derivative = torch.dot(gradient, direction)

        accepted = False
        step = 1.0
        for _ in range(50):
            candidate = torch.clamp(multipliers + step * direction, min=0.0)
            delta = candidate - multipliers
            _, _, _, candidate_objective = projection_state(candidate)
            armijo = objective + 1e-4 * torch.dot(gradient, delta)
            if bool((candidate_objective <= armijo).item()):
                multipliers = candidate
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break

    probabilities, expected_cost, _, _ = projection_state(multipliers)
    log_probabilities = torch.where(
        feasible, probabilities.clamp_min(1e-300).log(),
        torch.zeros_like(probabilities))
    finite_log_base = torch.where(
        feasible, log_base, torch.zeros_like(log_base))
    kl_by_state = torch.sum(
        probabilities * (log_probabilities - finite_log_base), dim=-1)
    entropy_by_state = -torch.sum(
        probabilities * log_probabilities, dim=-1)
    return {
        "base_probabilities": base_probabilities,
        "probabilities": probabilities,
        "converged": bool(converged),
        "iterations": int(iterations),
        "multipliers": multipliers,
        "expected_costs": expected_cost,
        "constraint_residuals": expected_cost - bounds,
        "kkt_residual": kkt_residual,
        "weighted_kl": torch.dot(weights, kl_by_state),
        "weighted_entropy": torch.dot(weights, entropy_by_state),
    }
