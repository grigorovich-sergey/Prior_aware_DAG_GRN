"""Train a prior-aware gene-regulatory quantum circuit.

The trainer owns observations, fixed initial-gene angles, edge-angle
optimization, loss evaluation, history, configuration, and result saving.
Backend construction, transpilation, Qiskit parameter order, and count
alignment remain in :mod:`quantum_execution`.

The real observations-table schema is intentionally not implemented yet.
``read_observed_frequencies`` therefore generates a deterministic synthetic
distribution with incomplete support.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
from numbers import Integral, Real
from pathlib import Path
import random
from typing import Any

import yaml
from tqdm.auto import tqdm


SUPPORTED_EXECUTION_MODES = frozenset(
    {"aer_noiseless", "aer_noisy", "ibm_hardware"}
)
SUPPORTED_INITIALIZATIONS = frozenset({"zero", "small_random"})

__all__ = [
    "LossEvaluation",
    "ProjectedSampleDistribution",
    "SPSATrainingResult",
    "TrainingHistoryRecord",
    "calculate_initial_gene_angles",
    "evaluate_sampled_counts",
    "initialize_edge_angles",
    "jensen_shannon_divergence",
    "load_effective_config",
    "merge_config",
    "probability_overlap",
    "project_sampled_counts_to_observed_support",
    "read_observed_frequencies",
    "run_training",
    "summarize_edge_sign_concordance",
    "train_with_spsa",
]


@dataclass(frozen=True)
class ProjectedSampleDistribution:
    """Sample output conditioned on the observed bitstring support."""

    matching_counts: dict[str, int]
    matching_probabilities: dict[str, float]
    total_shots: int
    matching_shots: int
    excluded_shots: int
    matching_fraction: float

    @property
    def nonmatching_fraction(self) -> float:
        return self.excluded_shots / self.total_shots


@dataclass(frozen=True)
class LossEvaluation:
    """Loss components and support diagnostics for one finite-shot sample."""

    total_loss: float
    distribution_loss: float
    support_penalty: float
    nonmatching_fraction: float
    agreement_score: float
    projection: ProjectedSampleDistribution


@dataclass(frozen=True)
class TrainingHistoryRecord:
    """Clean central-model metrics at iteration zero or after an update."""

    iteration: int
    total_loss: float
    distribution_loss: float
    support_penalty: float
    nonmatching_fraction: float
    agreement_score: float
    total_shots: int
    matching_shots: int
    excluded_shots: int
    learning_rate: float | None
    perturbation: float | None
    edge_angles: tuple[float, ...]
    central_job_id: str | None


@dataclass(frozen=True)
class SPSATrainingResult:
    """Optimized edge values plus complete training diagnostics."""

    initial_edge_angles: tuple[float, ...]
    optimized_edge_angles: tuple[float, ...]
    history: tuple[TrainingHistoryRecord, ...]
    perturbation_steps: tuple[dict[str, Any], ...]
    final_counts: dict[str, int]
    final_probabilities: dict[str, float]
    final_sampler_metadata: Mapping[str, Any]
    final_execution_metadata: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_bitstring_length(bitstring_length: int) -> int:
    if not isinstance(bitstring_length, Integral) or isinstance(
        bitstring_length, bool
    ):
        raise TypeError("bitstring_length must be an integer.")
    if bitstring_length <= 0:
        raise ValueError("bitstring_length must be greater than zero.")
    return int(bitstring_length)


def _validate_seed(seed: int | None, parameter_name: str = "seed") -> int | None:
    if seed is None:
        return None
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise TypeError(f"{parameter_name} must be an integer or None.")
    if seed < 0:
        raise ValueError(f"{parameter_name} must be nonnegative.")
    return int(seed)


def _validate_observed_frequencies(
    observed_frequencies: Mapping[str, float],
) -> tuple[dict[str, float], int]:
    if not isinstance(observed_frequencies, Mapping):
        raise TypeError("observed_frequencies must be a mapping.")
    if not observed_frequencies:
        raise ValueError("observed_frequencies must not be empty.")

    normalized: dict[str, float] = {}
    width: int | None = None
    for bitstring, frequency in observed_frequencies.items():
        if not isinstance(bitstring, str):
            raise TypeError("Every observed-frequency key must be a string.")
        if not bitstring or set(bitstring) - {"0", "1"}:
            raise ValueError("Every observed-frequency key must be a bitstring.")
        if width is None:
            width = len(bitstring)
        elif len(bitstring) != width:
            raise ValueError("All observed bitstrings must have the same length.")
        if not isinstance(frequency, Real) or isinstance(frequency, bool):
            raise TypeError("Every observed frequency must be a real number.")
        value = float(frequency)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("Every observed frequency must be finite and positive.")
        normalized[bitstring] = value

    total = math.fsum(normalized.values())
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"Observed frequencies must sum to 1; received {total:.17g}.")
    assert width is not None
    return ({key: value / total for key, value in normalized.items()}, width)


def _validate_probability_distribution(
    probabilities: Mapping[str, float],
) -> tuple[dict[str, float], int]:
    """Validate a normalized distribution while permitting zero entries."""
    if not isinstance(probabilities, Mapping):
        raise TypeError("probabilities must be a mapping.")
    if not probabilities:
        raise ValueError("probabilities must not be empty.")
    normalized: dict[str, float] = {}
    width: int | None = None
    for bitstring, probability in probabilities.items():
        if not isinstance(bitstring, str):
            raise TypeError("Every probability key must be a string.")
        if not bitstring or set(bitstring) - {"0", "1"}:
            raise ValueError("Every probability key must be a bitstring.")
        if width is None:
            width = len(bitstring)
        elif len(bitstring) != width:
            raise ValueError("All probability bitstrings must have the same length.")
        if not isinstance(probability, Real) or isinstance(probability, bool):
            raise TypeError("Every probability must be a real number.")
        value = float(probability)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("Every probability must be finite and nonnegative.")
        normalized[bitstring] = value
    total = math.fsum(normalized.values())
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"Probabilities must sum to 1; received {total:.17g}.")
    assert width is not None
    return ({key: value / total for key, value in normalized.items()}, width)


def read_observed_frequencies(
    observations_file: str | Path | None = None,
    *,
    bitstring_length: int,
    number_of_observed_bitstrings: int | None = None,
    seed: int | None = 0,
) -> dict[str, float]:
    """Generate deterministic synthetic observations with incomplete support.

    A non-null ``observations_file`` raises ``NotImplementedError`` until the
    real table schema is agreed. ``seed=None`` still uses seed zero so this
    temporary data source never becomes silently nondeterministic.
    """
    if observations_file is not None:
        if not isinstance(observations_file, (str, Path)):
            raise TypeError("observations_file must be a path or None.")
        raise NotImplementedError(
            "Observation-file loading is deferred until its table schema is defined."
        )

    width = _validate_bitstring_length(bitstring_length)
    normalized_seed = _validate_seed(seed)
    effective_seed = 0 if normalized_seed is None else normalized_seed
    total_outcomes = 1 << width

    if number_of_observed_bitstrings is None:
        support_size = min(8, max(1, total_outcomes // 2))
    else:
        if not isinstance(number_of_observed_bitstrings, Integral) or isinstance(
            number_of_observed_bitstrings, bool
        ):
            raise TypeError("number_of_observed_bitstrings must be an integer or None.")
        support_size = int(number_of_observed_bitstrings)
    if support_size <= 0 or support_size >= total_outcomes:
        raise ValueError(
            "number_of_observed_bitstrings must be positive and smaller than "
            "the number of possible bitstrings."
        )

    rng = random.Random(effective_seed)
    selected: set[int] = set()
    while len(selected) < support_size:
        selected.add(rng.getrandbits(width))
    bitstrings = [format(value, f"0{width}b") for value in sorted(selected)]
    weights = [rng.uniform(0.1, 1.0) for _ in bitstrings]
    weight_total = math.fsum(weights)
    frequencies = {
        bitstring: weight / weight_total
        for bitstring, weight in zip(bitstrings, weights)
    }
    last = bitstrings[-1]
    frequencies[last] = 1.0 - math.fsum(
        value for key, value in frequencies.items() if key != last
    )
    validated, _ = _validate_observed_frequencies(frequencies)
    return validated


def project_sampled_counts_to_observed_support(
    sampled_counts: Mapping[str, int],
    observed_frequencies: Mapping[str, float],
) -> ProjectedSampleDistribution:
    """Project counts onto observed support, retaining leakage diagnostics.

    In the zero-match case the conditional probabilities are all zero. The
    objective evaluator handles that case explicitly as maximal JSD.
    """
    observed, width = _validate_observed_frequencies(observed_frequencies)
    if not isinstance(sampled_counts, Mapping):
        raise TypeError("sampled_counts must be a mapping.")
    if not sampled_counts:
        raise ValueError("sampled_counts must not be empty.")

    counts: dict[str, int] = {}
    for bitstring, count in sampled_counts.items():
        if not isinstance(bitstring, str):
            raise TypeError("Every sampled-count key must be a string.")
        if len(bitstring) != width or set(bitstring) - {"0", "1"}:
            raise ValueError(
                "Every sampled-count key must match the observed bitstring width."
            )
        if not isinstance(count, Integral) or isinstance(count, bool):
            raise TypeError("Every sampled count must be an integer.")
        if count < 0:
            raise ValueError("Every sampled count must be nonnegative.")
        counts[bitstring] = int(count)

    total_shots = sum(counts.values())
    if total_shots <= 0:
        raise ValueError("sampled_counts must contain at least one shot.")
    matching_counts = {key: counts.get(key, 0) for key in observed}
    matching_shots = sum(matching_counts.values())
    matching_probabilities = (
        {key: count / matching_shots for key, count in matching_counts.items()}
        if matching_shots
        else {key: 0.0 for key in observed}
    )
    return ProjectedSampleDistribution(
        matching_counts=matching_counts,
        matching_probabilities=matching_probabilities,
        total_shots=total_shots,
        matching_shots=matching_shots,
        excluded_shots=total_shots - matching_shots,
        matching_fraction=matching_shots / total_shots,
    )


def _validate_logarithm_base(logarithm_base: float) -> float:
    if not isinstance(logarithm_base, Real) or isinstance(logarithm_base, bool):
        raise TypeError("logarithm_base must be a real number.")
    base = float(logarithm_base)
    if not math.isfinite(base) or base <= 1.0:
        raise ValueError("logarithm_base must be finite and greater than one.")
    return base


def jensen_shannon_divergence(
    first: Mapping[str, float],
    second: Mapping[str, float],
    *,
    logarithm_base: float = 2.0,
) -> float:
    """Calculate JSD for two normalized distributions on identical support."""
    first_validated, first_width = _validate_observed_frequencies(first)
    if set(first_validated) != set(second):
        raise ValueError("Both distributions must have identical support.")
    second_validated, second_width = _validate_probability_distribution(second)
    if first_width != second_width:
        raise ValueError("Both distributions must have equal bitstring width.")
    base = _validate_logarithm_base(logarithm_base)
    denominator = math.log(base)
    divergence = 0.0
    for key in first_validated:
        p = first_validated[key]
        q = second_validated[key]
        midpoint = 0.5 * (p + q)
        divergence += 0.5 * p * math.log(p / midpoint) / denominator
        if q > 0.0:
            divergence += 0.5 * q * math.log(q / midpoint) / denominator
    return max(0.0, divergence)


def probability_overlap(
    observed_frequencies: Mapping[str, float],
    sampled_probabilities: Mapping[str, float],
) -> float:
    """Return sum(min(Pobs, Psampl)), equal to one minus TV distance."""
    observed, _ = _validate_observed_frequencies(observed_frequencies)
    if set(observed) != set(sampled_probabilities):
        raise ValueError("Both distributions must have identical support.")
    sampled, _ = _validate_probability_distribution(sampled_probabilities)
    return math.fsum(min(observed[key], sampled[key]) for key in observed)


def evaluate_sampled_counts(
    sampled_counts: Mapping[str, int],
    observed_frequencies: Mapping[str, float],
    *,
    logarithm_base: float = 2.0,
    nonmatching_penalty_weight: float = 1.0,
) -> LossEvaluation:
    """Evaluate conditional JSD plus a penalty for sampled support leakage."""
    observed, _ = _validate_observed_frequencies(observed_frequencies)
    base = _validate_logarithm_base(logarithm_base)
    if not isinstance(nonmatching_penalty_weight, Real) or isinstance(
        nonmatching_penalty_weight, bool
    ):
        raise TypeError("nonmatching_penalty_weight must be a real number.")
    penalty_weight = float(nonmatching_penalty_weight)
    if not math.isfinite(penalty_weight) or penalty_weight < 0.0:
        raise ValueError("nonmatching_penalty_weight must be finite and nonnegative.")

    projection = project_sampled_counts_to_observed_support(
        sampled_counts, observed
    )
    leakage = projection.nonmatching_fraction
    if projection.matching_shots == 0:
        distribution_loss = math.log(2.0) / math.log(base)
        agreement = 0.0
    else:
        distribution_loss = jensen_shannon_divergence(
            observed,
            projection.matching_probabilities,
            logarithm_base=base,
        )
        agreement = probability_overlap(
            observed, projection.matching_probabilities
        )
    support_penalty = penalty_weight * leakage
    return LossEvaluation(
        total_loss=distribution_loss + support_penalty,
        distribution_loss=distribution_loss,
        support_penalty=support_penalty,
        nonmatching_fraction=leakage,
        agreement_score=agreement,
        projection=projection,
    )


def calculate_initial_gene_angles(
    observed_frequencies: Mapping[str, float],
) -> tuple[float, ...]:
    """Calculate fixed gene angles as 2*asin(sqrt(activation rate))."""
    observed, width = _validate_observed_frequencies(observed_frequencies)
    activation_rates = [
        math.fsum(
            probability
            for bitstring, probability in observed.items()
            if bitstring[index] == "1"
        )
        for index in range(width)
    ]
    return tuple(2.0 * math.asin(math.sqrt(rate)) for rate in activation_rates)


def _validate_bounds(bounds: Sequence[float]) -> tuple[float, float]:
    if isinstance(bounds, (str, bytes)) or len(bounds) != 2:
        raise ValueError("bounds must contain exactly [lower, upper].")
    lower, upper = bounds
    if any(
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for value in (lower, upper)
    ):
        raise ValueError("bounds must contain two finite real numbers.")
    normalized = (float(lower), float(upper))
    if normalized[0] >= normalized[1]:
        raise ValueError("The lower bound must be smaller than the upper bound.")
    return normalized


def _project_bounds(
    values: Sequence[float], bounds: Sequence[float]
) -> tuple[float, ...]:
    lower, upper = _validate_bounds(bounds)
    return tuple(min(upper, max(lower, float(value))) for value in values)


def initialize_edge_angles(
    number_of_edges: int,
    *,
    method: str = "zero",
    small_random_half_width: float = 0.05,
    seed: int | None = 0,
    bounds: Sequence[float] = (-math.pi / 2.0, math.pi / 2.0),
) -> tuple[float, ...]:
    """Initialize edge angles with zeros or seeded values near zero."""
    if not isinstance(number_of_edges, Integral) or isinstance(number_of_edges, bool):
        raise TypeError("number_of_edges must be an integer.")
    if number_of_edges < 0:
        raise ValueError("number_of_edges must be nonnegative.")
    if method not in SUPPORTED_INITIALIZATIONS:
        raise ValueError("method must be 'zero' or 'small_random'.")
    if not isinstance(small_random_half_width, Real) or isinstance(
        small_random_half_width, bool
    ):
        raise TypeError("small_random_half_width must be a real number.")
    half_width = float(small_random_half_width)
    if not math.isfinite(half_width) or half_width < 0.0:
        raise ValueError("small_random_half_width must be finite and nonnegative.")
    normalized_seed = _validate_seed(seed)
    if method == "zero":
        values = (0.0,) * int(number_of_edges)
    else:
        rng = random.Random(0 if normalized_seed is None else normalized_seed)
        values = tuple(
            rng.uniform(-half_width, half_width)
            for _ in range(int(number_of_edges))
        )
    return _project_bounds(values, bounds)


def summarize_edge_sign_concordance(
    scheduled_edges: Sequence[Sequence[str]],
    known_relationship_signs: Sequence[str | int],
    trained_edge_angles: Sequence[float],
    *,
    zero_threshold: float,
) -> dict[str, Any]:
    """Compare trained edge-angle signs with known regulatory signs.

    Activation expects a positive angle and repression a negative angle.
    Angles whose absolute value is at or below ``zero_threshold`` receive half
    credit. Mixed and unknown relationships are reported but not scored.
    """
    threshold = _require_finite(
        zero_threshold,
        "biological_validation.edge_angle_zero_threshold",
        minimum=0.0,
    )
    edges = tuple(tuple(edge) for edge in scheduled_edges)
    signs = tuple(known_relationship_signs)
    angles = tuple(trained_edge_angles)
    if not (len(edges) == len(signs) == len(angles)):
        raise ValueError(
            "scheduled_edges, known_relationship_signs, and "
            "trained_edge_angles must have equal lengths."
        )

    label_to_sign = {"activation": 1, "repression": -1}
    records: list[dict[str, Any]] = []
    correct = near_zero = wrong = excluded = 0

    for index, (edge, known_sign, angle) in enumerate(
        zip(edges, signs, angles)
    ):
        if len(edge) != 2 or any(not isinstance(gene, str) for gene in edge):
            raise ValueError(
                f"scheduled_edges[{index}] must contain two gene strings."
            )
        if not isinstance(angle, Real) or isinstance(angle, bool):
            raise TypeError(f"trained_edge_angles[{index}] must be real.")
        numeric_angle = float(angle)
        if not math.isfinite(numeric_angle):
            raise ValueError(f"trained_edge_angles[{index}] must be finite.")

        if isinstance(known_sign, str):
            relationship = known_sign.strip().lower()
            expected_sign = label_to_sign.get(relationship)
        elif isinstance(known_sign, Integral) and not isinstance(known_sign, bool):
            numeric_sign = int(known_sign)
            if numeric_sign not in {-1, 0, 1}:
                raise ValueError(
                    f"known_relationship_signs[{index}] must be -1, 0, or 1."
                )
            expected_sign = numeric_sign or None
            relationship = {
                1: "activation",
                -1: "repression",
                0: "unknown",
            }[numeric_sign]
        else:
            raise TypeError(
                f"known_relationship_signs[{index}] must be a string or integer."
            )

        if numeric_angle > threshold:
            angle_sign = 1
            angle_class = "positive"
        elif numeric_angle < -threshold:
            angle_sign = -1
            angle_class = "negative"
        else:
            angle_sign = 0
            angle_class = "near_zero"

        if expected_sign is None:
            classification = "not_evaluable"
            score = None
            excluded += 1
        elif angle_sign == 0:
            classification = "near_zero"
            score = 0.5
            near_zero += 1
        elif angle_sign == expected_sign:
            classification = "correct"
            score = 1.0
            correct += 1
        else:
            classification = "wrong"
            score = 0.0
            wrong += 1

        source, target = edge
        records.append(
            {
                "edge": f"{source}->{target}",
                "source": source,
                "target": target,
                "known_relationship": relationship,
                "expected_angle_sign": expected_sign,
                "trained_angle": numeric_angle,
                "trained_angle_class": angle_class,
                "classification": classification,
                "score": score,
            }
        )

    evaluated = correct + near_zero + wrong
    return {
        "status": "evaluated",
        "zero_threshold_radians": threshold,
        "total_edges": len(edges),
        "evaluated_edges": evaluated,
        "excluded_mixed_or_unknown_edges": excluded,
        "correct_sign_edges": correct,
        "near_zero_edges": near_zero,
        "wrong_sign_edges": wrong,
        "strict_sign_accuracy": None if not evaluated else correct / evaluated,
        "half_credit_score": (
            None if not evaluated else (correct + 0.5 * near_zero) / evaluated
        ),
        "edges": records,
    }


def _history_record(
    iteration: int,
    evaluation: LossEvaluation,
    edge_angles: Sequence[float],
    central_job_id: str | None,
    learning_rate: float | None,
    perturbation: float | None,
) -> TrainingHistoryRecord:
    projection = evaluation.projection
    return TrainingHistoryRecord(
        iteration=iteration,
        total_loss=evaluation.total_loss,
        distribution_loss=evaluation.distribution_loss,
        support_penalty=evaluation.support_penalty,
        nonmatching_fraction=evaluation.nonmatching_fraction,
        agreement_score=evaluation.agreement_score,
        total_shots=projection.total_shots,
        matching_shots=projection.matching_shots,
        excluded_shots=projection.excluded_shots,
        learning_rate=learning_rate,
        perturbation=perturbation,
        edge_angles=tuple(float(value) for value in edge_angles),
        central_job_id=central_job_id,
    )


def train_with_spsa(
    executor: Any,
    prepared: Any,
    observed_frequencies: Mapping[str, float],
    initial_gene_angles: Sequence[float],
    initial_edge_angles: Sequence[float],
    *,
    iterations: int,
    shots: int,
    logarithm_base: float,
    nonmatching_penalty_weight: float,
    learning_rate: float,
    stability_offset: float,
    learning_rate_exponent: float,
    perturbation: float,
    perturbation_exponent: float,
    bounds: Sequence[float],
    seed: int | None,
    result_timeout: float | None = None,
) -> SPSATrainingResult:
    """Optimize edge angles with local first-order SPSA.

    The positive and negative perturbations share one batched job. A separate
    central-model job supplies clean history at iteration zero and after every
    update. Both perturbations and updates are projected onto ``bounds``.
    """
    from quantum_execution import CircuitParameterSet, execute_parameter_sets

    if not isinstance(iterations, Integral) or isinstance(iterations, bool):
        raise TypeError("iterations must be an integer.")
    if iterations < 0:
        raise ValueError("iterations must be nonnegative.")
    if not isinstance(shots, Integral) or isinstance(shots, bool) or shots <= 0:
        raise ValueError("shots must be a positive integer.")
    schedule_values = {
        "learning_rate": learning_rate,
        "stability_offset": stability_offset,
        "learning_rate_exponent": learning_rate_exponent,
        "perturbation": perturbation,
        "perturbation_exponent": perturbation_exponent,
    }
    for name, value in schedule_values.items():
        if not isinstance(value, Real) or isinstance(value, bool):
            raise TypeError(f"{name} must be a real number.")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite.")
    if learning_rate <= 0.0 or perturbation <= 0.0:
        raise ValueError("learning_rate and perturbation must be positive.")
    if stability_offset < 0.0:
        raise ValueError("stability_offset must be nonnegative.")
    if learning_rate_exponent <= 0.0 or perturbation_exponent <= 0.0:
        raise ValueError("Schedule exponents must be positive.")

    normalized_seed = _validate_seed(seed)
    rng = random.Random(0 if normalized_seed is None else normalized_seed)
    initial_values = tuple(float(value) for value in initial_gene_angles)
    edges = _project_bounds(initial_edge_angles, bounds)
    if iterations and not edges:
        raise ValueError("SPSA requires at least one trainable edge angle.")

    def evaluate_central(
        iteration: int,
        values: tuple[float, ...],
        current_learning_rate: float | None,
        current_perturbation: float | None,
    ) -> tuple[TrainingHistoryRecord, Any, Any]:
        batch = execute_parameter_sets(
            executor,
            prepared,
            [CircuitParameterSet(initial_values, values, f"central_{iteration}")],
            shots=int(shots),
            result_timeout=result_timeout,
        )
        sample = batch.samples[0]
        evaluation = evaluate_sampled_counts(
            sample.aligned_counts,
            observed_frequencies,
            logarithm_base=logarithm_base,
            nonmatching_penalty_weight=nonmatching_penalty_weight,
        )
        return (
            _history_record(
                iteration,
                evaluation,
                values,
                batch.job_id,
                current_learning_rate,
                current_perturbation,
            ),
            sample,
            batch,
        )

    first_record, final_sample, final_batch = evaluate_central(
        0, edges, None, None
    )
    history = [first_record]
    perturbation_steps: list[dict[str, Any]] = []

    progress = tqdm(
        range(1, int(iterations) + 1),
        desc="Training",
        unit="iteration",
        dynamic_ncols=True,
    )

    for iteration in progress:
        ak = float(learning_rate) / (
            float(stability_offset) + iteration
        ) ** float(learning_rate_exponent)
        ck = float(perturbation) / iteration ** float(perturbation_exponent)
        delta = tuple(1.0 if rng.random() < 0.5 else -1.0 for _ in edges)
        plus = _project_bounds(
            (value + ck * sign for value, sign in zip(edges, delta)), bounds
        )
        minus = _project_bounds(
            (value - ck * sign for value, sign in zip(edges, delta)), bounds
        )
        perturbation_batch = execute_parameter_sets(
            executor,
            prepared,
            [
                CircuitParameterSet(initial_values, plus, f"plus_{iteration}"),
                CircuitParameterSet(initial_values, minus, f"minus_{iteration}"),
            ],
            shots=int(shots),
            result_timeout=result_timeout,
        )
        plus_evaluation = evaluate_sampled_counts(
            perturbation_batch.samples[0].aligned_counts,
            observed_frequencies,
            logarithm_base=logarithm_base,
            nonmatching_penalty_weight=nonmatching_penalty_weight,
        )
        minus_evaluation = evaluate_sampled_counts(
            perturbation_batch.samples[1].aligned_counts,
            observed_frequencies,
            logarithm_base=logarithm_base,
            nonmatching_penalty_weight=nonmatching_penalty_weight,
        )
        loss_difference = plus_evaluation.total_loss - minus_evaluation.total_loss
        gradient = tuple(loss_difference * sign / (2.0 * ck) for sign in delta)
        edges = _project_bounds(
            (value - ak * component for value, component in zip(edges, gradient)),
            bounds,
        )
        record, final_sample, final_batch = evaluate_central(
            iteration, edges, ak, ck
        )
        history.append(record)
        progress.set_postfix(
            loss=f"{record.total_loss:.5f}",
            nonmatching=f"{record.nonmatching_fraction:.3f}",
            agreement=f"{record.agreement_score:.3f}",
        )
        perturbation_steps.append(
            {
                "iteration": iteration,
                "job_id": perturbation_batch.job_id,
                "learning_rate": ak,
                "perturbation": ck,
                "delta": list(delta),
                "plus_edge_angles": list(plus),
                "minus_edge_angles": list(minus),
                "plus_loss": plus_evaluation.total_loss,
                "minus_loss": minus_evaluation.total_loss,
                "gradient_estimate": list(gradient),
                "updated_edge_angles": list(edges),
                "execution_metadata": dict(perturbation_batch.execution_metadata),
            }
        )

    return SPSATrainingResult(
        initial_edge_angles=tuple(float(value) for value in initial_edge_angles),
        optimized_edge_angles=edges,
        history=tuple(history),
        perturbation_steps=tuple(perturbation_steps),
        final_counts=dict(final_sample.aligned_counts),
        final_probabilities=dict(final_sample.probabilities),
        final_sampler_metadata=dict(final_sample.sampler_metadata),
        final_execution_metadata=dict(final_batch.execution_metadata),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a prior-aware gene-regulatory quantum circuit."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional YAML file whose values recursively override "
            "configs/default_model.yaml."
        ),
    )
    return parser.parse_args()


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration file '{path}' must contain a mapping.")
    return config


def merge_config(
    base: Mapping[str, Any], overrides: Mapping[str, Any]
) -> dict[str, Any]:
    """Recursively merge overrides into a deep copy of the defaults."""
    merged = deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_effective_config(override_path: str | Path | None) -> dict[str, Any]:
    default_path = Path(__file__).resolve().parents[1] / "configs" / "default_model.yaml"
    config = load_config(default_path)
    if override_path is not None:
        config = merge_config(config, load_config(override_path))
    return config


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration section '{key}' must be a mapping.")
    return value


def _require_finite(
    value: Any, name: str, *, minimum: float | None = None, strict: bool = False
) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise ValueError(f"{name} must be a real number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and (
        normalized <= minimum if strict else normalized < minimum
    ):
        relation = "greater than" if strict else "at least"
        raise ValueError(f"{name} must be {relation} {minimum}.")
    return normalized


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the relationships needed before backend construction."""
    input_config = _require_mapping(config, "input")
    observations = _require_mapping(config, "observations")
    circuit = _require_mapping(config, "circuit")
    execution = _require_mapping(config, "execution")
    transpilation = _require_mapping(config, "transpilation")
    loss = _require_mapping(config, "loss")
    biological_validation = _require_mapping(config, "biological_validation")
    optimizer = _require_mapping(config, "optimizer")
    output = _require_mapping(config, "output")

    if not isinstance(input_config.get("graph_file"), (str, Path)):
        raise ValueError("input.graph_file must be a path.")
    if input_config.get("observations_file") is not None:
        raise NotImplementedError(
            "input.observations_file must remain null until its schema is defined."
        )
    support_size = observations.get("number_of_observed_bitstrings")
    if support_size is not None and (
        not isinstance(support_size, Integral)
        or isinstance(support_size, bool)
        or support_size <= 0
    ):
        raise ValueError(
            "observations.number_of_observed_bitstrings must be positive or null."
        )
    _validate_seed(observations.get("seed"), "observations.seed")

    for name in ("rotation_axis", "initial_rotation_axis", "controlled_rotation_axis"):
        value = circuit.get(name)
        if name == "rotation_axis" and value not in {"x", "y"}:
            raise ValueError("circuit.rotation_axis must be 'x' or 'y'.")
        if name != "rotation_axis" and value not in {None, "x", "y"}:
            raise ValueError(f"circuit.{name} must be null, 'x', or 'y'.")

    mode = execution.get("mode")
    if mode not in SUPPORTED_EXECUTION_MODES:
        raise ValueError("execution.mode is not supported.")
    backend_name = execution.get("backend_name")
    if mode == "aer_noiseless" and backend_name is not None:
        raise ValueError("execution.backend_name must be null for aer_noiseless.")
    if mode != "aer_noiseless" and (
        not isinstance(backend_name, str) or not backend_name
    ):
        raise ValueError("execution.backend_name is required for this mode.")
    shots = execution.get("shots")
    if not isinstance(shots, Integral) or isinstance(shots, bool) or shots <= 0:
        raise ValueError("execution.shots must be a positive integer.")
    timeout = execution.get("result_timeout")
    if timeout is not None:
        _require_finite(timeout, "execution.result_timeout", minimum=0.0, strict=True)
    _validate_seed(execution.get("seed_simulator"), "execution.seed_simulator")
    if mode == "ibm_hardware" and execution.get("seed_simulator") is not None:
        raise ValueError("execution.seed_simulator must be null for ibm_hardware.")
    if not isinstance(execution.get("aer_method"), str) or not execution.get("aer_method"):
        raise ValueError("execution.aer_method must be a nonempty string.")
    for name in ("aer_backend_options", "sampler_options"):
        if not isinstance(execution.get(name), Mapping):
            raise ValueError(f"execution.{name} must be a mapping.")
    use_session = execution.get("use_runtime_session")
    if not isinstance(use_session, bool):
        raise ValueError("execution.use_runtime_session must be Boolean.")
    if mode != "ibm_hardware" and use_session:
        raise ValueError("Runtime sessions are only valid for ibm_hardware.")
    if mode != "ibm_hardware" and execution.get("session_max_time") is not None:
        raise ValueError("execution.session_max_time is only valid for ibm_hardware.")

    level = transpilation.get("optimization_level")
    if not isinstance(level, Integral) or isinstance(level, bool) or level not in {0, 1, 2, 3}:
        raise ValueError("transpilation.optimization_level must be 0, 1, 2, or 3.")
    _validate_seed(transpilation.get("seed_transpiler"), "transpilation.seed_transpiler")
    if not isinstance(transpilation.get("options"), Mapping):
        raise ValueError("transpilation.options must be a mapping.")

    _validate_logarithm_base(loss.get("logarithm_base"))
    _require_finite(
        loss.get("nonmatching_penalty_weight"),
        "loss.nonmatching_penalty_weight",
        minimum=0.0,
    )
    _require_finite(
        biological_validation.get("edge_angle_zero_threshold"),
        "biological_validation.edge_angle_zero_threshold",
        minimum=0.0,
    )
    if optimizer.get("method") != "local_spsa":
        raise ValueError("optimizer.method must be 'local_spsa'.")
    iterations = optimizer.get("iterations")
    if not isinstance(iterations, Integral) or isinstance(iterations, bool) or iterations < 0:
        raise ValueError("optimizer.iterations must be a nonnegative integer.")
    _validate_seed(optimizer.get("seed"), "optimizer.seed")
    _validate_bounds(optimizer.get("bounds"))
    initialization = optimizer.get("initialization")
    if not isinstance(initialization, Mapping):
        raise ValueError("optimizer.initialization must be a mapping.")
    if initialization.get("method") not in SUPPORTED_INITIALIZATIONS:
        raise ValueError("optimizer.initialization.method must be zero or small_random.")
    _require_finite(
        initialization.get("small_random_half_width"),
        "optimizer.initialization.small_random_half_width",
        minimum=0.0,
    )
    schedule = optimizer.get("schedule")
    if not isinstance(schedule, Mapping):
        raise ValueError("optimizer.schedule must be a mapping.")
    for name in (
        "learning_rate",
        "learning_rate_exponent",
        "perturbation",
        "perturbation_exponent",
    ):
        _require_finite(schedule.get(name), f"optimizer.schedule.{name}", minimum=0.0, strict=True)
    _require_finite(
        schedule.get("stability_offset"),
        "optimizer.schedule.stability_offset",
        minimum=0.0,
    )

    history_file = output.get("history_file")
    if history_file is not None and not isinstance(history_file, (str, Path)):
        raise ValueError("output.history_file must be a path or null.")
    metadata_file = output.get("metadata_file")
    if not isinstance(metadata_file, (str, Path)) or not str(metadata_file):
        raise ValueError("output.metadata_file must be a nonempty path.")


def save_history_csv(
    history: Sequence[TrainingHistoryRecord], file_path: str | Path
) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "iteration",
        "total_loss",
        "distribution_loss",
        "support_penalty",
        "nonmatching_fraction",
        "agreement_score",
        "total_shots",
        "matching_shots",
        "excluded_shots",
        "learning_rate",
        "perturbation",
        "edge_angles",
        "central_job_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in history:
            row = asdict(record)
            row["edge_angles"] = json.dumps(row["edge_angles"])
            writer.writerow(row)


def _json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_compatible(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_compatible(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def save_run_metadata(metadata: Mapping[str, Any], file_path: str | Path) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_compatible(metadata), file, indent=2)
        file.write("\n")


def run_training(config: Mapping[str, Any]) -> tuple[SPSATrainingResult, dict[str, Any]]:
    """Build, prepare once, train, and return results plus run metadata."""
    from quantum_circuit import (
        build_parameterized_circuit,
        load_gene_graph,
        metadata_to_serializable,
    )
    from quantum_execution import (
        close_quantum_executor,
        create_quantum_executor,
        prepare_execution,
    )

    validate_config(config)
    started_at = _utc_now()
    input_config = config["input"]
    observations_config = config["observations"]
    circuit_config = config["circuit"]
    execution = config["execution"]
    transpilation = config["transpilation"]
    loss = config["loss"]
    biological_validation = config["biological_validation"]
    optimizer = config["optimizer"]

    graph_path = resolve_project_path(input_config["graph_file"])
    graph = load_gene_graph(graph_path)
    circuit, circuit_metadata = build_parameterized_circuit(
        graph,
        rotation_axis=circuit_config["rotation_axis"],
        initial_rotation_axis=circuit_config["initial_rotation_axis"],
        controlled_rotation_axis=circuit_config["controlled_rotation_axis"],
        add_measurements=True,
    )
    observed = read_observed_frequencies(
        input_config["observations_file"],
        bitstring_length=len(circuit_metadata["ordered_genes"]),
        number_of_observed_bitstrings=observations_config[
            "number_of_observed_bitstrings"
        ],
        seed=observations_config["seed"],
    )
    initial_gene_angles = calculate_initial_gene_angles(observed)
    initialization = optimizer["initialization"]
    initial_edge_angles = initialize_edge_angles(
        len(circuit_metadata["scheduled_edges"]),
        method=initialization["method"],
        small_random_half_width=initialization["small_random_half_width"],
        seed=optimizer["seed"],
        bounds=optimizer["bounds"],
    )

    executor = create_quantum_executor(
        execution["mode"],
        backend_name=execution["backend_name"],
        aer_method=execution["aer_method"],
        seed_simulator=execution["seed_simulator"],
        aer_backend_options=execution["aer_backend_options"],
        sampler_options=execution["sampler_options"],
        use_runtime_session=execution["use_runtime_session"],
        session_max_time=execution["session_max_time"],
    )
    try:
        prepared = prepare_execution(
            executor,
            circuit,
            circuit_metadata,
            optimization_level=transpilation["optimization_level"],
            seed_transpiler=transpilation["seed_transpiler"],
            transpiler_options=transpilation["options"],
        )
        schedule = optimizer["schedule"]
        result = train_with_spsa(
            executor,
            prepared,
            observed,
            initial_gene_angles,
            initial_edge_angles,
            iterations=optimizer["iterations"],
            shots=execution["shots"],
            logarithm_base=loss["logarithm_base"],
            nonmatching_penalty_weight=loss["nonmatching_penalty_weight"],
            learning_rate=schedule["learning_rate"],
            stability_offset=schedule["stability_offset"],
            learning_rate_exponent=schedule["learning_rate_exponent"],
            perturbation=schedule["perturbation"],
            perturbation_exponent=schedule["perturbation_exponent"],
            bounds=optimizer["bounds"],
            seed=optimizer["seed"],
            result_timeout=execution["result_timeout"],
        )
    finally:
        close_quantum_executor(executor)

    metadata = {
        "status": "completed",
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "effective_config": deepcopy(dict(config)),
        "random_seeds": {
            "observations_configured": observations_config["seed"],
            "observations_effective": (
                0 if observations_config["seed"] is None else observations_config["seed"]
            ),
            "optimizer_configured": optimizer["seed"],
            "optimizer_effective": 0 if optimizer["seed"] is None else optimizer["seed"],
            "simulator": execution["seed_simulator"],
            "transpiler": transpilation["seed_transpiler"],
        },
        "resolved_paths": {"graph_file": str(graph_path)},
        "observed_frequencies": observed,
        "circuit": metadata_to_serializable(circuit_metadata),
        "initial_gene_angles": list(initial_gene_angles),
        "gene_to_initial_angle": dict(
            zip(circuit_metadata["ordered_genes"], initial_gene_angles)
        ),
        "initial_edge_angles": list(result.initial_edge_angles),
        "optimized_edge_angles": list(result.optimized_edge_angles),
        "edge_to_initial_angle": {
            f"{source}->{target}": angle
            for (source, target), angle in zip(
                circuit_metadata["scheduled_edges"], result.initial_edge_angles
            )
        },
        "edge_to_optimized_angle": {
            f"{source}->{target}": angle
            for (source, target), angle in zip(
                circuit_metadata["scheduled_edges"], result.optimized_edge_angles
            )
        },
        "training_summary": {
            "iterations_completed": len(result.history) - 1,
            "initial_total_loss": result.history[0].total_loss,
            "final_total_loss": result.history[-1].total_loss,
            "initial_nonmatching_fraction": (
                result.history[0].nonmatching_fraction
            ),
            "final_nonmatching_fraction": (
                result.history[-1].nonmatching_fraction
            ),
            "initial_agreement_score": (
                result.history[0].agreement_score
            ),
            "final_agreement_score": (
                result.history[-1].agreement_score
            ),
            "central_jobs_submitted": len(result.history),
            "perturbation_jobs_submitted": len(result.perturbation_steps),
        },
        "execution_provenance": dict(prepared.provenance),
        "final_counts": result.final_counts,
        "final_probabilities": result.final_probabilities,
        "final_sampler_metadata": dict(result.final_sampler_metadata),
        "final_execution_metadata": dict(result.final_execution_metadata),
    }
    final_evaluation = evaluate_sampled_counts(
        result.final_counts,
        observed,
        logarithm_base=loss["logarithm_base"],
        nonmatching_penalty_weight=loss["nonmatching_penalty_weight"],
    )
    metadata["final_loss_evaluation"] = asdict(final_evaluation)

    controlled_axis = circuit_metadata["controlled_rotation_axis"]
#    if controlled_axis == "y":
    known_signs = [
        graph.edges[source, target].get(
            "sign",
            graph.edges[source, target].get("regulatory_sign", 0),
        )
        for source, target in circuit_metadata["scheduled_edges"]
    ]
    sign_concordance = summarize_edge_sign_concordance(
        circuit_metadata["scheduled_edges"],
        known_signs,
        result.optimized_edge_angles,
        zero_threshold=biological_validation[
            "edge_angle_zero_threshold"
        ],
    )
    sign_concordance["controlled_rotation_axis"] = controlled_axis
    sign_concordance["interpretation"] = (
        "Limited angle-sign concordance sanity check; it is not proof "
        "that a biological regulatory mechanism was reconstructed."
    )
#    else:
#        sign_concordance = {
#            "status": "not_evaluated",
#            "controlled_rotation_axis": controlled_axis,
#            "reason": (
#                "The sign of a CRX angle is not directly an activation or "
#                "repression direction in computational-basis measurements."
#            ),
#        }
    metadata["biological_sign_concordance"] = sign_concordance
    return result, metadata


def main() -> int:
    args = parse_args()
    config = load_effective_config(args.config)
    validate_config(config)
    result, metadata = run_training(config)
    metadata["configuration_sources"] = {
        "default": str(
            Path(__file__).resolve().parents[1] / "configs" / "default_model.yaml"
        ),
        "override": (
            None
            if args.config is None
            else str(Path(args.config).expanduser().resolve())
        ),
    }
    output = config["output"]
    history_path = None
    if output["history_file"] is not None:
        history_path = resolve_project_path(output["history_file"])
        save_history_csv(result.history, history_path)
    metadata_path = resolve_project_path(output["metadata_file"])
    metadata["resolved_paths"].update(
        {
            "history_file": None if history_path is None else str(history_path),
            "metadata_file": str(metadata_path),
        }
    )
    save_run_metadata(metadata, metadata_path)
    if history_path is not None:
        print(f"Saved training history: {history_path}")
    print(f"Saved run metadata: {metadata_path}")
    sign_summary = metadata["biological_sign_concordance"]
    if sign_summary["status"] == "evaluated":
        score = sign_summary["half_credit_score"]
        score_text = "n/a" if score is None else f"{score:.3f}"
        print(
            "Edge-sign concordance: "
            f"{sign_summary['correct_sign_edges']} correct, "
            f"{sign_summary['near_zero_edges']} near-zero, "
            f"{sign_summary['wrong_sign_edges']} wrong; "
            f"half-credit score={score_text}."
        )
    else:
        print(f"Edge-sign concordance not evaluated: {sign_summary['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
