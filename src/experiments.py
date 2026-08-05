"""Run post-training interventions on a reconstructed quantum gene DAG.

The script rebuilds a trained circuit from ``trainer.py`` metadata and the
saved DAG, then samples an unchanged baseline, single-gene interventions, and
single-edge interventions. Execution settings are intentionally independent
from training settings.

For each configured output stem, two files are written:

* ``.csv``: tidy per-gene marginal activation frequencies.
* ``.json``: full aligned bitstring counts/probabilities and provenance.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from numbers import Integral, Real
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import yaml
try:
    from tqdm.auto import tqdm
except ImportError:
    class _NoOpProgress:
        """Minimal tqdm-compatible wrapper when tqdm is not installed."""

        def __init__(self, iterable: Sequence[int]) -> None:
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix(self, **_: Any) -> None:
            return None

    def tqdm(iterable: Sequence[int], **_: Any) -> _NoOpProgress:
        return _NoOpProgress(iterable)


SUPPORTED_EXECUTION_MODES = frozenset(
    {"aer_noiseless", "aer_noisy", "ibm_hardware"}
)

__all__ = [
    "ExperimentCondition",
    "TrainedModelSpec",
    "activation_marginals",
    "build_experiment_conditions",
    "load_effective_config",
    "merge_config",
    "run_experiments",
]


@dataclass(frozen=True)
class TrainedModelSpec:
    """Circuit structure and numerical parameters recovered from training."""

    ordered_genes: tuple[str, ...]
    scheduled_edges: tuple[tuple[str, str], ...]
    initial_rotation_axis: str
    controlled_rotation_axis: str
    initial_gene_angles: tuple[float, ...]
    optimized_edge_angles: tuple[float, ...]


@dataclass(frozen=True)
class ExperimentCondition:
    """One complete circuit assignment and its intervention description."""

    condition_id: str
    experiment: str
    intervention_target: str | None
    intervention_value: str | None
    initial_values: tuple[float, ...]
    edge_values: tuple[float, ...]
    modifications: Mapping[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline, single-gene, and single-edge experiments on a "
            "trained prior-aware quantum gene DAG."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional YAML file whose values recursively override "
            "configs/default_experiments.yaml."
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
    default_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "default_experiments.yaml"
    )
    config = load_config(default_path)
    if override_path is not None:
        config = merge_config(config, load_config(override_path))
    return config


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parents[1] / candidate


def _require_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration section '{key}' must be a mapping.")
    return value


def _validate_seed(seed: Any, name: str) -> int | None:
    if seed is None:
        return None
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise ValueError(f"{name} must be a nonnegative integer or null.")
    if seed < 0:
        raise ValueError(f"{name} must be a nonnegative integer or null.")
    return int(seed)


def _require_finite(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    strict: bool = False,
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
    """Validate configuration before loading a graph or creating a backend."""
    input_config = _require_mapping(config, "input")
    experiments = _require_mapping(config, "experiments")
    execution = _require_mapping(config, "execution")
    transpilation = _require_mapping(config, "transpilation")
    output = _require_mapping(config, "output")

    results_file = input_config.get("results_file")
    if not isinstance(results_file, (str, Path)) or not str(results_file):
        raise ValueError("input.results_file must be a nonempty path.")
    graph_file = input_config.get("graph_file")
    if graph_file is not None and (
        not isinstance(graph_file, (str, Path)) or not str(graph_file)
    ):
        raise ValueError("input.graph_file must be a nonempty path or null.")

    enabled = []
    for name in ("baseline", "single_gene", "single_edge"):
        value = experiments.get(name)
        if not isinstance(value, bool):
            raise ValueError(f"experiments.{name} must be Boolean.")
        enabled.append(value)
    if not any(enabled):
        raise ValueError("At least one experiment must be enabled.")
    repetitions = experiments.get("repetitions")
    if (
        not isinstance(repetitions, Integral)
        or isinstance(repetitions, bool)
        or repetitions <= 0
    ):
        raise ValueError("experiments.repetitions must be a positive integer.")

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
        _require_finite(
            timeout,
            "execution.result_timeout",
            minimum=0.0,
            strict=True,
        )
    simulator_seed = _validate_seed(
        execution.get("seed_simulator"),
        "execution.seed_simulator",
    )
    if mode == "ibm_hardware" and simulator_seed is not None:
        raise ValueError(
            "execution.seed_simulator must be null for ibm_hardware."
        )
    if mode != "ibm_hardware" and simulator_seed is None:
        raise ValueError(
            "execution.seed_simulator is required for reproducible Aer "
            "repetitions."
        )
    if (
        not isinstance(execution.get("aer_method"), str)
        or not execution.get("aer_method")
    ):
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
        raise ValueError(
            "execution.session_max_time is only valid for ibm_hardware."
        )
    batch_size = execution.get("batch_size")
    if batch_size is not None and (
        not isinstance(batch_size, Integral)
        or isinstance(batch_size, bool)
        or batch_size <= 0
    ):
        raise ValueError("execution.batch_size must be positive or null.")

    level = transpilation.get("optimization_level")
    if (
        not isinstance(level, Integral)
        or isinstance(level, bool)
        or level not in {0, 1, 2, 3}
    ):
        raise ValueError(
            "transpilation.optimization_level must be 0, 1, 2, or 3."
        )
    _validate_seed(
        transpilation.get("seed_transpiler"),
        "transpilation.seed_transpiler",
    )
    if not isinstance(transpilation.get("options"), Mapping):
        raise ValueError("transpilation.options must be a mapping.")

    file_stem = output.get("file_stem")
    if not isinstance(file_stem, (str, Path)) or not str(file_stem):
        raise ValueError("output.file_stem must be a nonempty path.")
    if not isinstance(output.get("show_progress"), bool):
        raise ValueError("output.show_progress must be Boolean.")


def _load_json_object(file_path: str | Path, description: str) -> dict[str, Any]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return value


def _finite_sequence(
    values: Any,
    expected_length: int,
    name: str,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of numbers.")
    if len(values) != expected_length:
        raise ValueError(
            f"{name} must contain {expected_length} values; "
            f"received {len(values)}."
        )
    return tuple(
        _require_finite(value, f"{name}[{index}]")
        for index, value in enumerate(values)
    )


def extract_trained_model(results: Mapping[str, Any]) -> TrainedModelSpec:
    """Extract and validate the circuit reconstruction contract."""
    circuit = results.get("circuit")
    if not isinstance(circuit, Mapping):
        raise ValueError("Training results must contain a 'circuit' object.")

    raw_genes = circuit.get("ordered_genes")
    if (
        isinstance(raw_genes, (str, bytes))
        or not isinstance(raw_genes, Sequence)
        or not raw_genes
        or any(not isinstance(gene, str) or not gene for gene in raw_genes)
        or len(raw_genes) != len(set(raw_genes))
    ):
        raise ValueError(
            "results.circuit.ordered_genes must contain unique gene strings."
        )
    ordered_genes = tuple(raw_genes)

    raw_edges = circuit.get("scheduled_edges")
    if isinstance(raw_edges, (str, bytes)) or not isinstance(raw_edges, Sequence):
        raise ValueError("results.circuit.scheduled_edges must be a sequence.")
    scheduled_edges: list[tuple[str, str]] = []
    known_genes = set(ordered_genes)
    for index, raw_edge in enumerate(raw_edges):
        if (
            isinstance(raw_edge, (str, bytes))
            or not isinstance(raw_edge, Sequence)
            or len(raw_edge) != 2
            or any(not isinstance(gene, str) for gene in raw_edge)
        ):
            raise ValueError(
                f"results.circuit.scheduled_edges[{index}] must contain "
                "two gene strings."
            )
        edge = (raw_edge[0], raw_edge[1])
        if edge[0] not in known_genes or edge[1] not in known_genes:
            raise ValueError(
                f"results.circuit.scheduled_edges[{index}] references an "
                "unknown gene."
            )
        scheduled_edges.append(edge)
    if len(scheduled_edges) != len(set(scheduled_edges)):
        raise ValueError("results.circuit.scheduled_edges contains duplicates.")

    initial_axis = circuit.get("initial_rotation_axis")
    controlled_axis = circuit.get("controlled_rotation_axis")
    if initial_axis not in {"x", "y"}:
        raise ValueError("results circuit initial rotation axis must be x or y.")
    if controlled_axis not in {"x", "y"}:
        raise ValueError("results circuit controlled rotation axis must be x or y.")

    initial_gene_angles = _finite_sequence(
        results.get("initial_gene_angles"),
        len(ordered_genes),
        "results.initial_gene_angles",
    )
    optimized_edge_angles = _finite_sequence(
        results.get("optimized_edge_angles"),
        len(scheduled_edges),
        "results.optimized_edge_angles",
    )
    return TrainedModelSpec(
        ordered_genes=ordered_genes,
        scheduled_edges=tuple(scheduled_edges),
        initial_rotation_axis=initial_axis,
        controlled_rotation_axis=controlled_axis,
        initial_gene_angles=initial_gene_angles,
        optimized_edge_angles=optimized_edge_angles,
    )


def resolve_graph_path(
    input_config: Mapping[str, Any],
    training_results: Mapping[str, Any],
) -> tuple[Path, str]:
    """Find the DAG using override, portable saved path, then absolute path."""
    explicit = input_config.get("graph_file")
    if explicit is not None:
        path = resolve_project_path(explicit)
        if not path.is_file():
            raise FileNotFoundError(
                f"Configured input.graph_file does not exist: {path}"
            )
        return path, "experiment_config"

    candidates: list[tuple[Path, str]] = []
    effective_config = training_results.get("effective_config")
    if isinstance(effective_config, Mapping):
        training_input = effective_config.get("input")
        if isinstance(training_input, Mapping):
            saved_graph = training_input.get("graph_file")
            if isinstance(saved_graph, (str, Path)) and str(saved_graph):
                candidates.append(
                    (resolve_project_path(saved_graph), "training_effective_config")
                )

    resolved_paths = training_results.get("resolved_paths")
    if isinstance(resolved_paths, Mapping):
        saved_resolved = resolved_paths.get("graph_file")
        if isinstance(saved_resolved, (str, Path)) and str(saved_resolved):
            candidates.append(
                (Path(saved_resolved).expanduser(), "training_resolved_path")
            )

    for path, source in candidates:
        if path.is_file():
            return path, source

    attempted = ", ".join(str(path) for path, _ in candidates) or "none"
    raise FileNotFoundError(
        "Could not locate the training DAG. Set input.graph_file explicitly. "
        f"Attempted saved paths: {attempted}."
    )


def build_experiment_conditions(
    model: TrainedModelSpec,
    *,
    include_baseline: bool,
    include_single_gene: bool,
    include_single_edge: bool,
) -> tuple[ExperimentCondition, ...]:
    """Create deterministic complete parameter sets for selected experiments."""
    conditions: list[ExperimentCondition] = []
    baseline_initial = model.initial_gene_angles
    baseline_edges = model.optimized_edge_angles

    if include_baseline:
        conditions.append(
            ExperimentCondition(
                condition_id="baseline",
                experiment="baseline",
                intervention_target=None,
                intervention_value=None,
                initial_values=baseline_initial,
                edge_values=baseline_edges,
                modifications={},
            )
        )

    if include_single_gene:
        for gene_index, gene in enumerate(model.ordered_genes):
            incoming_indices = [
                edge_index
                for edge_index, (_, target) in enumerate(model.scheduled_edges)
                if target == gene
            ]
            incoming_edges = [
                f"{model.scheduled_edges[index][0]}->{gene}"
                for index in incoming_indices
            ]
            for state_name, forced_angle in (("off", 0.0), ("on", math.pi)):
                initial_values = list(baseline_initial)
                initial_values[gene_index] = forced_angle
                edge_values = list(baseline_edges)
                for edge_index in incoming_indices:
                    edge_values[edge_index] = 0.0
                conditions.append(
                    ExperimentCondition(
                        condition_id=f"gene:{gene}:{state_name}",
                        experiment="single_gene",
                        intervention_target=gene,
                        intervention_value=state_name,
                        initial_values=tuple(initial_values),
                        edge_values=tuple(edge_values),
                        modifications={
                            "forced_gene": gene,
                            "forced_initial_angle": forced_angle,
                            "disabled_incoming_edges": incoming_edges,
                        },
                    )
                )

    if include_single_edge:
        for edge_index, (source, target) in enumerate(model.scheduled_edges):
            edge_values = list(baseline_edges)
            original_angle = edge_values[edge_index]
            edge_values[edge_index] = 0.0
            edge_label = f"{source}->{target}"
            conditions.append(
                ExperimentCondition(
                    condition_id=f"edge:{edge_label}:disabled",
                    experiment="single_edge",
                    intervention_target=edge_label,
                    intervention_value="disabled",
                    initial_values=baseline_initial,
                    edge_values=tuple(edge_values),
                    modifications={
                        "disabled_edge": edge_label,
                        "original_edge_angle": original_angle,
                        "intervention_edge_angle": 0.0,
                    },
                )
            )

    if not conditions:
        raise ValueError("The selected experiments produced no conditions.")
    return tuple(conditions)


def activation_marginals(
    probabilities: Mapping[str, float],
    ordered_genes: Sequence[str],
) -> dict[str, float]:
    """Calculate P(gene=1) from biologically aligned bitstring probabilities."""
    genes = tuple(ordered_genes)
    if not genes or len(genes) != len(set(genes)):
        raise ValueError("ordered_genes must contain unique gene strings.")
    marginals = {gene: 0.0 for gene in genes}
    total = 0.0
    for bitstring, probability in probabilities.items():
        if (
            not isinstance(bitstring, str)
            or len(bitstring) != len(genes)
            or set(bitstring) - {"0", "1"}
        ):
            raise ValueError(
                "Every probability key must be a bitstring aligned with "
                "ordered_genes."
            )
        value = _require_finite(probability, f"probabilities[{bitstring!r}]")
        if value < 0.0:
            raise ValueError("Probabilities must be nonnegative.")
        total += value
        for index, gene in enumerate(genes):
            if bitstring[index] == "1":
                marginals[gene] += value
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"Probabilities must sum to 1; received {total:.17g}.")
    return marginals


def _chunks(
    values: Sequence[Any], batch_size: int | None
) -> Sequence[Sequence[Any]]:
    if batch_size is None:
        return (values,)
    return tuple(
        values[start : start + batch_size]
        for start in range(0, len(values), batch_size)
    )


def _condition_record(condition: ExperimentCondition) -> dict[str, Any]:
    return {
        "condition_id": condition.condition_id,
        "experiment": condition.experiment,
        "intervention_target": condition.intervention_target,
        "intervention_value": condition.intervention_value,
        "modifications": deepcopy(dict(condition.modifications)),
    }


def _build_marginal_outputs(
    runs: Sequence[Mapping[str, Any]],
    ordered_genes: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_values: dict[str, list[float]] = {
        gene: [] for gene in ordered_genes
    }
    grouped: dict[tuple[str, str], list[float]] = {}
    baseline_by_repetition: dict[tuple[int, str], float] = {}
    for run in runs:
        for gene, value in run["activation_frequencies"].items():
            grouped.setdefault((run["condition_id"], gene), []).append(value)
            if run["experiment"] == "baseline":
                baseline_values[gene].append(value)
                baseline_by_repetition[(run["repetition"], gene)] = value

    baseline_means = {
        gene: (fmean(values) if values else None)
        for gene, values in baseline_values.items()
    }
    rows: list[dict[str, Any]] = []
    for run in runs:
        for gene in ordered_genes:
            activation_frequency = run["activation_frequencies"][gene]
            baseline_mean = baseline_means[gene]
            paired_baseline = baseline_by_repetition.get(
                (run["repetition"], gene)
            )
            rows.append(
                {
                    "experiment": run["experiment"],
                    "condition_id": run["condition_id"],
                    "intervention_target": run["intervention_target"],
                    "intervention_value": run["intervention_value"],
                    "repetition": run["repetition"],
                    "simulator_seed": run["simulator_seed"],
                    "measured_gene": gene,
                    "activation_frequency": activation_frequency,
                    "baseline_mean_activation_frequency": baseline_mean,
                    "difference_from_baseline_mean": (
                        None
                        if baseline_mean is None
                        else activation_frequency - baseline_mean
                    ),
                    "paired_baseline_activation_frequency": paired_baseline,
                    "difference_from_paired_baseline": (
                        None
                        if paired_baseline is None
                        else activation_frequency - paired_baseline
                    ),
                    "shots": run["shots"],
                    "job_id": run["job_id"],
                }
            )

    condition_lookup = {run["condition_id"]: run for run in runs}
    summaries: list[dict[str, Any]] = []
    for (condition_id, gene), values in grouped.items():
        example = condition_lookup[condition_id]
        mean_value = fmean(values)
        baseline_mean = baseline_means[gene]
        paired_differences = [
            run["activation_frequencies"][gene]
            - baseline_by_repetition[(run["repetition"], gene)]
            for run in runs
            if run["condition_id"] == condition_id
            and (run["repetition"], gene) in baseline_by_repetition
        ]
        summaries.append(
            {
                "experiment": example["experiment"],
                "condition_id": condition_id,
                "intervention_target": example["intervention_target"],
                "intervention_value": example["intervention_value"],
                "measured_gene": gene,
                "repetitions": len(values),
                "mean_activation_frequency": mean_value,
                "sample_standard_deviation": (
                    stdev(values) if len(values) > 1 else None
                ),
                "baseline_mean_activation_frequency": baseline_mean,
                "mean_difference_from_baseline": (
                    None if baseline_mean is None else mean_value - baseline_mean
                ),
                "mean_paired_difference_from_baseline": (
                    fmean(paired_differences) if paired_differences else None
                ),
                "paired_difference_sample_standard_deviation": (
                    stdev(paired_differences)
                    if len(paired_differences) > 1
                    else None
                ),
            }
        )
    return rows, summaries


def run_experiments(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reconstruct, sample, and return JSON metadata plus tidy CSV rows."""
    from quantum_circuit import (
        build_parameterized_circuit,
        load_gene_graph,
        metadata_to_serializable,
    )
    from quantum_execution import (
        CircuitParameterSet,
        close_quantum_executor,
        create_quantum_executor,
        execute_parameter_sets,
        prepare_execution,
    )

    validate_config(config)
    started_at = _utc_now()
    input_config = config["input"]
    selection = config["experiments"]
    execution = config["execution"]
    transpilation = config["transpilation"]

    results_path = resolve_project_path(input_config["results_file"])
    training_results = _load_json_object(results_path, "Training results file")
    model = extract_trained_model(training_results)
    graph_path, graph_path_source = resolve_graph_path(
        input_config,
        training_results,
    )
    graph = load_gene_graph(graph_path)
    circuit, circuit_metadata = build_parameterized_circuit(
        graph,
        rotation_axis=model.initial_rotation_axis,
        initial_rotation_axis=model.initial_rotation_axis,
        controlled_rotation_axis=model.controlled_rotation_axis,
        add_measurements=True,
    )
    rebuilt_genes = tuple(circuit_metadata["ordered_genes"])
    rebuilt_edges = tuple(circuit_metadata["scheduled_edges"])
    if rebuilt_genes != model.ordered_genes:
        raise ValueError(
            "The DAG gene order does not match the order saved by training. "
            f"Saved: {model.ordered_genes}; rebuilt: {rebuilt_genes}."
        )
    if rebuilt_edges != model.scheduled_edges:
        raise ValueError(
            "The DAG edge schedule does not match the schedule saved by "
            f"training. Saved: {model.scheduled_edges}; rebuilt: {rebuilt_edges}."
        )

    conditions = build_experiment_conditions(
        model,
        include_baseline=selection["baseline"],
        include_single_gene=selection["single_gene"],
        include_single_edge=selection["single_edge"],
    )
    parameter_sets = tuple(
        CircuitParameterSet(
            initial_values=condition.initial_values,
            edge_values=condition.edge_values,
            label=condition.condition_id,
        )
        for condition in conditions
    )
    condition_by_id = {
        condition.condition_id: condition for condition in conditions
    }

    repetitions = int(selection["repetitions"])
    base_seed = execution["seed_simulator"]
    mode = execution["mode"]
    batch_size = execution["batch_size"]
    runs: list[dict[str, Any]] = []
    repetition_provenance: list[dict[str, Any]] = []
    progress = tqdm(
        range(repetitions),
        desc="Experiments",
        unit="repetition",
        dynamic_ncols=True,
        disable=not config["output"]["show_progress"],
    )
    for repetition_index in progress:
        simulator_seed = (
            None if mode == "ibm_hardware" else int(base_seed) + repetition_index
        )
        executor = create_quantum_executor(
            mode,
            backend_name=execution["backend_name"],
            aer_method=execution["aer_method"],
            seed_simulator=simulator_seed,
            aer_backend_options=execution["aer_backend_options"],
            sampler_options=execution["sampler_options"],
            use_runtime_session=execution["use_runtime_session"],
            session_max_time=execution["session_max_time"],
        )
        repetition_record: dict[str, Any] = {
            "repetition": repetition_index + 1,
            "simulator_seed": simulator_seed,
            "jobs": [],
        }
        try:
            prepared = prepare_execution(
                executor,
                circuit,
                circuit_metadata,
                optimization_level=transpilation["optimization_level"],
                seed_transpiler=transpilation["seed_transpiler"],
                transpiler_options=transpilation["options"],
            )
            repetition_record["execution_provenance"] = dict(prepared.provenance)
            for batch_index, batch in enumerate(
                _chunks(parameter_sets, batch_size),
                start=1,
            ):
                batch_result = execute_parameter_sets(
                    executor,
                    prepared,
                    batch,
                    shots=execution["shots"],
                    result_timeout=execution["result_timeout"],
                )
                repetition_record["jobs"].append(
                    {
                        "batch": batch_index,
                        "job_id": batch_result.job_id,
                        "condition_ids": [item.label for item in batch],
                        "execution_metadata": dict(
                            batch_result.execution_metadata
                        ),
                    }
                )
                for sample in batch_result.samples:
                    if sample.label not in condition_by_id:
                        raise RuntimeError(
                            f"Unexpected execution result label: {sample.label!r}."
                        )
                    condition = condition_by_id[sample.label]
                    runs.append(
                        {
                            **_condition_record(condition),
                            "repetition": repetition_index + 1,
                            "simulator_seed": simulator_seed,
                            "shots": sample.shots,
                            "job_id": batch_result.job_id,
                            "aligned_counts": dict(sample.aligned_counts),
                            "probabilities": dict(sample.probabilities),
                            "activation_frequencies": activation_marginals(
                                sample.probabilities,
                                model.ordered_genes,
                            ),
                            "sampler_metadata": dict(sample.sampler_metadata),
                        }
                    )
        finally:
            close_quantum_executor(executor)
        repetition_provenance.append(repetition_record)
        progress.set_postfix(
            conditions=len(conditions),
            seed=("hardware" if simulator_seed is None else simulator_seed),
        )

    expected_runs = repetitions * len(conditions)
    if len(runs) != expected_runs:
        raise RuntimeError(
            f"Expected {expected_runs} sampled conditions; received {len(runs)}."
        )
    csv_rows, marginal_summaries = _build_marginal_outputs(
        runs,
        model.ordered_genes,
    )
    metadata = {
        "status": "completed",
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "effective_config": deepcopy(dict(config)),
        "source_training_run": {
            "results_file": str(results_path),
            "status": training_results.get("status"),
            "started_at_utc": training_results.get("started_at_utc"),
            "completed_at_utc": training_results.get("completed_at_utc"),
        },
        "resolved_paths": {
            "training_results_file": str(results_path),
            "graph_file": str(graph_path),
            "graph_path_source": graph_path_source,
        },
        "circuit": metadata_to_serializable(circuit_metadata),
        "trained_parameters": {
            "initial_gene_angles": list(model.initial_gene_angles),
            "optimized_edge_angles": list(model.optimized_edge_angles),
        },
        "experiment_summary": {
            "repetitions": repetitions,
            "conditions_per_repetition": len(conditions),
            "total_sampled_conditions": len(runs),
            "shots_per_condition": execution["shots"],
            "simulator_seeds": [
                None if mode == "ibm_hardware" else int(base_seed) + index
                for index in range(repetitions)
            ],
        },
        "conditions": [_condition_record(condition) for condition in conditions],
        "repetition_provenance": repetition_provenance,
        "runs": runs,
        "marginal_summaries": marginal_summaries,
    }
    return metadata, csv_rows


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


def resolve_output_paths(file_stem: str | Path) -> tuple[Path, Path]:
    stem = resolve_project_path(file_stem)
    if stem.suffix in {".csv", ".json"}:
        stem = stem.with_suffix("")
    return Path(f"{stem}.csv"), Path(f"{stem}.json")


def save_marginals_csv(
    rows: Sequence[Mapping[str, Any]],
    file_path: str | Path,
) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "condition_id",
        "intervention_target",
        "intervention_value",
        "repetition",
        "simulator_seed",
        "measured_gene",
        "activation_frequency",
        "baseline_mean_activation_frequency",
        "difference_from_baseline_mean",
        "paired_baseline_activation_frequency",
        "difference_from_paired_baseline",
        "shots",
        "job_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_experiment_metadata(
    metadata: Mapping[str, Any],
    file_path: str | Path,
) -> None:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_compatible(metadata), file, indent=2)
        file.write("\n")


def main() -> int:
    args = parse_args()
    config = load_effective_config(args.config)
    validate_config(config)
    metadata, csv_rows = run_experiments(config)
    csv_path, json_path = resolve_output_paths(config["output"]["file_stem"])
    metadata["configuration_sources"] = {
        "default": str(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "default_experiments.yaml"
        ),
        "override": (
            None
            if args.config is None
            else str(Path(args.config).expanduser().resolve())
        ),
    }
    metadata["resolved_paths"].update(
        {
            "csv_file": str(csv_path),
            "json_file": str(json_path),
        }
    )
    save_marginals_csv(csv_rows, csv_path)
    save_experiment_metadata(metadata, json_path)
    print(f"Saved marginal frequencies to: {csv_path}")
    print(f"Saved full experiment results to: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
