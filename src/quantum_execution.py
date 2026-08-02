"""Prepare and execute prior-aware quantum circuits on interchangeable backends.

This module owns backend selection, transpilation, finite-shot submission, and
count extraction. It deliberately does not load observed data, calculate a
loss, or optimize circuit parameters; those responsibilities belong in the
trainer.

Running this file directly performs a small noiseless-Aer smoke test.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import math
from numbers import Integral, Real
from typing import Any, Literal

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.transpiler.preset_passmanagers import (
    generate_preset_pass_manager,
)

from quantum_circuit import align_bitstring, make_parameter_bindings


ExecutionMode = Literal["aer_noiseless", "aer_noisy", "ibm_hardware"]
SUPPORTED_EXECUTION_MODES = frozenset(
    {"aer_noiseless", "aer_noisy", "ibm_hardware"}
)

__all__ = [
    "CircuitParameterSet",
    "ExecutionBatchResult",
    "ExecutionMode",
    "PreparedExecution",
    "QuantumExecutor",
    "SampleResult",
    "align_counts",
    "close_quantum_executor",
    "counts_to_probabilities",
    "create_quantum_executor",
    "execute_parameter_sets",
    "prepare_execution",
]


@dataclass(frozen=True)
class CircuitParameterSet:
    """One complete numerical assignment for a parameterized circuit."""

    initial_values: Sequence[float]
    edge_values: Sequence[float]
    label: str | None = None


@dataclass
class QuantumExecutor:
    """Backend-specific runtime objects behind the common execution API."""

    mode: ExecutionMode
    backend_name: str
    backend: Any
    sampler: Any
    provenance: dict[str, Any]
    service: Any | None = None
    session: Any | None = None
    closed: bool = False


@dataclass(frozen=True)
class PreparedExecution:
    """A symbolic circuit transpiled once for one executor backend."""

    circuit: QuantumCircuit
    transpiled_circuit: QuantumCircuit
    circuit_metadata: Mapping[str, Any]
    parameter_order: tuple[Parameter, ...]
    measurement_register: str
    mode: ExecutionMode
    backend_name: str
    provenance: Mapping[str, Any]
    _executor: QuantumExecutor = field(repr=False, compare=False)


@dataclass(frozen=True)
class SampleResult:
    """One finite-shot sample in biological gene order."""

    label: str | None
    aligned_counts: dict[str, int]
    probabilities: dict[str, float]
    shots: int
    sampler_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionBatchResult:
    """Results and job provenance for one batched Sampler submission."""

    samples: tuple[SampleResult, ...]
    job_id: str | None
    execution_metadata: Mapping[str, Any]


def _utc_now() -> str:
    """Return a timezone-aware UTC timestamp suitable for provenance."""
    return datetime.now(timezone.utc).isoformat()


def _package_version(package_name: str) -> str | None:
    """Return an installed distribution version without making it required."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def _backend_name(backend: Any) -> str:
    """Extract a stable backend name across BackendV2 implementations."""
    name = getattr(backend, "name", None)
    if callable(name):
        name = name()
    if not isinstance(name, str) or not name:
        return type(backend).__name__
    return name


def _validate_mode(mode: str) -> ExecutionMode:
    """Validate and narrow an execution-mode string."""
    if not isinstance(mode, str):
        raise TypeError("mode must be a string.")
    if mode not in SUPPORTED_EXECUTION_MODES:
        choices = ", ".join(sorted(SUPPORTED_EXECUTION_MODES))
        raise ValueError(f"mode must be one of: {choices}.")
    return mode  # type: ignore[return-value]


def _validate_seed(seed: int | None, parameter_name: str) -> int | None:
    """Validate a nonnegative optional random seed."""
    if seed is None:
        return None
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise TypeError(f"{parameter_name} must be an integer or None.")
    if seed < 0:
        raise ValueError(f"{parameter_name} must be nonnegative.")
    return int(seed)


def _copy_options(
    options: Mapping[str, Any] | None,
    parameter_name: str,
) -> dict[str, Any]:
    """Copy an optional mapping so caller-owned configuration is not mutated."""
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError(f"{parameter_name} must be a mapping or None.")
    return dict(options)


def _runtime_service(service: Any | None) -> Any:
    """Return an injected IBM service or construct one from saved credentials."""
    if service is not None:
        return service

    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError as error:
        raise ImportError(
            "IBM execution requires the 'qiskit-ibm-runtime' package."
        ) from error

    return QiskitRuntimeService()


def create_quantum_executor(
    mode: ExecutionMode,
    *,
    backend_name: str | None = None,
    aer_method: str = "automatic",
    seed_simulator: int | None = None,
    aer_backend_options: Mapping[str, Any] | None = None,
    sampler_options: Mapping[str, Any] | None = None,
    use_runtime_session: bool = False,
    session_max_time: str | int | None = None,
    service: Any | None = None,
) -> QuantumExecutor:
    """Create a finite-shot executor for Aer or IBM Quantum hardware.

    Parameters are exposed so a trainer can propagate an effective YAML
    configuration without changing this module. For Aer, simulator options
    belong in ``aer_backend_options`` and ``sampler_options`` may contain Aer
    SamplerV2 run options. For hardware, ``sampler_options`` follows IBM
    Runtime's SamplerOptions structure.

    ``aer_noisy`` and ``ibm_hardware`` require ``backend_name``. The former
    obtains that IBM backend through ``service`` and creates an approximate
    device noise simulator with :meth:`AerSimulator.from_backend`.
    """
    normalized_mode = _validate_mode(mode)
    normalized_seed = _validate_seed(seed_simulator, "seed_simulator")

    if not isinstance(aer_method, str) or not aer_method:
        raise ValueError("aer_method must be a non-empty string.")
    if not isinstance(use_runtime_session, bool):
        raise TypeError("use_runtime_session must be a Boolean.")
    if normalized_mode != "ibm_hardware" and use_runtime_session:
        raise ValueError(
            "use_runtime_session is only supported for ibm_hardware."
        )
    if normalized_mode != "ibm_hardware" and session_max_time is not None:
        raise ValueError(
            "session_max_time is only supported for ibm_hardware."
        )
    if session_max_time is not None:
        if isinstance(session_max_time, bool) or not isinstance(
            session_max_time,
            (str, Integral),
        ):
            raise TypeError("session_max_time must be a string, integer, or None.")
        if isinstance(session_max_time, str) and not session_max_time:
            raise ValueError("session_max_time must not be an empty string.")
        if isinstance(session_max_time, Integral) and session_max_time <= 0:
            raise ValueError("session_max_time must be greater than zero.")

    backend_options = _copy_options(
        aer_backend_options,
        "aer_backend_options",
    )
    primitive_options = _copy_options(sampler_options, "sampler_options")
    created_at = _utc_now()
    package_versions = {
        "qiskit": _package_version("qiskit"),
        "qiskit_aer": _package_version("qiskit-aer"),
        "qiskit_ibm_runtime": _package_version("qiskit-ibm-runtime"),
    }

    if normalized_mode == "aer_noiseless":
        try:
            from qiskit_aer import AerSimulator
            from qiskit_aer.primitives import SamplerV2
        except ImportError as error:
            raise ImportError(
                "Aer execution requires the 'qiskit-aer' package."
            ) from error

        if backend_name is not None:
            raise ValueError(
                "backend_name must be None for aer_noiseless execution."
            )
        if service is not None:
            raise ValueError("service is not used for aer_noiseless execution.")
        if "backend_options" in primitive_options:
            raise ValueError(
                "Place Aer simulator options in aer_backend_options, not "
                "sampler_options['backend_options']."
            )
        if "method" in backend_options:
            raise ValueError(
                "Set the Aer simulation method with aer_method, not "
                "aer_backend_options['method']."
            )
        configured_method = aer_method
        backend = AerSimulator(method=configured_method, **backend_options)
        sampler = SamplerV2.from_backend(
            backend,
            seed=normalized_seed,
            options=primitive_options or None,
        )
        effective_backend_name = _backend_name(backend)
        provenance = {
            "mode": normalized_mode,
            "backend_name": effective_backend_name,
            "aer_method": configured_method,
            "seed_simulator": normalized_seed,
            "noise_model_backend": None,
            "aer_backend_options": _to_serializable(backend_options),
            "sampler_options": _to_serializable(primitive_options),
            "created_at_utc": created_at,
            "package_versions": package_versions,
        }
        return QuantumExecutor(
            mode=normalized_mode,
            backend_name=effective_backend_name,
            backend=backend,
            sampler=sampler,
            provenance=provenance,
        )

    if not isinstance(backend_name, str) or not backend_name:
        raise ValueError(f"backend_name is required for {normalized_mode}.")

    runtime_service = _runtime_service(service)
    hardware_backend = runtime_service.backend(backend_name)

    if normalized_mode == "aer_noisy":
        try:
            from qiskit_aer import AerSimulator
            from qiskit_aer.primitives import SamplerV2
        except ImportError as error:
            raise ImportError(
                "Noisy Aer execution requires the 'qiskit-aer' package."
            ) from error

        if "backend_options" in primitive_options:
            raise ValueError(
                "Place Aer simulator options in aer_backend_options, not "
                "sampler_options['backend_options']."
            )
        if "method" in backend_options:
            raise ValueError(
                "Set the Aer simulation method with aer_method, not "
                "aer_backend_options['method']."
            )
        configured_method = aer_method
        backend = AerSimulator.from_backend(
            hardware_backend,
            method=configured_method,
            **backend_options,
        )
        sampler = SamplerV2.from_backend(
            backend,
            seed=normalized_seed,
            options=primitive_options or None,
        )
        effective_backend_name = _backend_name(backend)
        provenance = {
            "mode": normalized_mode,
            "backend_name": effective_backend_name,
            "aer_method": configured_method,
            "seed_simulator": normalized_seed,
            "noise_model_backend": _backend_name(hardware_backend),
            "noise_model_retrieved_at_utc": created_at,
            "aer_backend_options": _to_serializable(backend_options),
            "sampler_options": _to_serializable(primitive_options),
            "created_at_utc": created_at,
            "package_versions": package_versions,
        }
        return QuantumExecutor(
            mode=normalized_mode,
            backend_name=effective_backend_name,
            backend=backend,
            sampler=sampler,
            service=runtime_service,
            provenance=provenance,
        )

    if backend_options:
        raise ValueError(
            "aer_backend_options cannot be used for ibm_hardware."
        )
    if normalized_seed is not None:
        raise ValueError("seed_simulator cannot be used for ibm_hardware.")

    try:
        from qiskit_ibm_runtime import SamplerV2, Session
    except ImportError as error:
        raise ImportError(
            "IBM hardware execution requires 'qiskit-ibm-runtime'."
        ) from error

    session = None
    sampler_mode = hardware_backend
    if use_runtime_session:
        session_kwargs: dict[str, Any] = {"backend": hardware_backend}
        if session_max_time is not None:
            session_kwargs["max_time"] = session_max_time
        session = Session(**session_kwargs)
        sampler_mode = session

    sampler = SamplerV2(
        mode=sampler_mode,
        options=primitive_options or None,
    )
    effective_backend_name = _backend_name(hardware_backend)
    provenance = {
        "mode": normalized_mode,
        "backend_name": effective_backend_name,
        "runtime_session": use_runtime_session,
        "session_max_time": session_max_time,
        "sampler_options": _to_serializable(primitive_options),
        "created_at_utc": created_at,
        "package_versions": package_versions,
    }
    return QuantumExecutor(
        mode=normalized_mode,
        backend_name=effective_backend_name,
        backend=hardware_backend,
        sampler=sampler,
        service=runtime_service,
        session=session,
        provenance=provenance,
    )


def _validate_circuit_metadata(
    circuit: QuantumCircuit,
    circuit_metadata: Mapping[str, Any],
) -> tuple[list[str], tuple[Parameter, ...], str]:
    """Validate the structural contract supplied by quantum_circuit.py."""
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("circuit must be a QuantumCircuit.")
    if not isinstance(circuit_metadata, Mapping):
        raise TypeError("circuit_metadata must be a mapping.")

    required_fields = {
        "ordered_genes",
        "initial_angles",
        "edge_angles",
        "add_measurements",
    }
    missing = sorted(required_fields - set(circuit_metadata))
    if missing:
        raise KeyError(f"circuit_metadata is missing required fields: {missing}.")
    if circuit_metadata["add_measurements"] is not True:
        raise ValueError("Execution requires a circuit with measurements.")

    raw_ordered_genes = circuit_metadata["ordered_genes"]
    if (
        isinstance(raw_ordered_genes, (str, bytes))
        or not isinstance(raw_ordered_genes, Sequence)
    ):
        raise TypeError("ordered_genes must be a sequence of gene strings.")
    ordered_genes = list(raw_ordered_genes)
    if (
        not ordered_genes
        or any(not isinstance(gene, str) for gene in ordered_genes)
        or len(ordered_genes) != len(set(ordered_genes))
    ):
        raise ValueError("ordered_genes must contain unique gene strings.")
    if len(ordered_genes) != circuit.num_qubits:
        raise ValueError(
            "The number of ordered genes must equal the number of qubits."
        )
    if len(ordered_genes) != circuit.num_clbits:
        raise ValueError(
            "The number of ordered genes must equal the number of classical bits."
        )
    if len(circuit.cregs) != 1:
        raise ValueError(
            "Execution requires exactly one classical measurement register."
        )
    measurement_register = circuit.cregs[0]
    if len(measurement_register) != len(ordered_genes):
        raise ValueError(
            "The measurement register size must equal the number of genes."
        )
    if circuit.count_ops().get("measure", 0) != len(ordered_genes):
        raise ValueError("Every gene qubit must be measured exactly once.")
    measurement_pairs = {
        (
            circuit.find_bit(instruction.qubits[0]).index,
            circuit.find_bit(instruction.clbits[0]).index,
        )
        for instruction in circuit.data
        if instruction.operation.name == "measure"
    }
    expected_measurement_pairs = {
        (index, index)
        for index in range(len(ordered_genes))
    }
    if measurement_pairs != expected_measurement_pairs:
        raise ValueError(
            "Measurements must map qubit i to classical bit i as created by "
            "measure_all()."
        )

    expected_parameters = tuple(circuit_metadata["initial_angles"]) + tuple(
        circuit_metadata["edge_angles"]
    )
    if set(circuit.parameters) != set(expected_parameters):
        raise ValueError("Circuit parameters do not match circuit_metadata.")

    return ordered_genes, expected_parameters, measurement_register.name


def prepare_execution(
    executor: QuantumExecutor,
    circuit: QuantumCircuit,
    circuit_metadata: Mapping[str, Any],
    *,
    optimization_level: int = 1,
    seed_transpiler: int | None = None,
    transpiler_options: Mapping[str, Any] | None = None,
) -> PreparedExecution:
    """Transpile a still-symbolic circuit once for the executor's backend."""
    if not isinstance(executor, QuantumExecutor):
        raise TypeError("executor must be a QuantumExecutor.")
    if executor.closed:
        raise RuntimeError("executor has already been closed.")
    if (
        not isinstance(optimization_level, Integral)
        or isinstance(optimization_level, bool)
        or optimization_level not in {0, 1, 2, 3}
    ):
        raise ValueError("optimization_level must be 0, 1, 2, or 3.")
    normalized_seed = _validate_seed(seed_transpiler, "seed_transpiler")
    ordered_genes, expected_parameters, measurement_register = (
        _validate_circuit_metadata(circuit, circuit_metadata)
    )

    extra_options = _copy_options(transpiler_options, "transpiler_options")
    reserved_options = {
        "backend",
        "target",
        "optimization_level",
        "seed_transpiler",
    }
    conflicts = sorted(reserved_options & set(extra_options))
    if conflicts:
        raise ValueError(
            "transpiler_options contains separately managed keys: "
            f"{conflicts}."
        )

    pass_manager = generate_preset_pass_manager(
        backend=executor.backend,
        optimization_level=int(optimization_level),
        seed_transpiler=normalized_seed,
        **extra_options,
    )
    transpiled_circuit = pass_manager.run(circuit)
    parameter_order = tuple(transpiled_circuit.parameters)
    if set(parameter_order) != set(expected_parameters):
        missing = sorted(
            str(parameter)
            for parameter in set(expected_parameters) - set(parameter_order)
        )
        unexpected = sorted(
            str(parameter)
            for parameter in set(parameter_order) - set(expected_parameters)
        )
        raise RuntimeError(
            "Transpilation changed the symbolic parameter set. "
            f"Missing: {missing}; unexpected: {unexpected}."
        )

    prepared_at = _utc_now()
    provenance = {
        **executor.provenance,
        "prepared_at_utc": prepared_at,
        "optimization_level": int(optimization_level),
        "seed_transpiler": normalized_seed,
        "transpiler_options": _to_serializable(extra_options),
        "logical_num_qubits": circuit.num_qubits,
        "transpiled_num_qubits": transpiled_circuit.num_qubits,
        "logical_depth": circuit.depth(),
        "transpiled_depth": transpiled_circuit.depth(),
        "logical_gate_counts": dict(circuit.count_ops()),
        "transpiled_gate_counts": dict(transpiled_circuit.count_ops()),
        "parameter_order": [str(parameter) for parameter in parameter_order],
        "ordered_genes": ordered_genes,
        "layout": (
            None
            if transpiled_circuit.layout is None
            else str(transpiled_circuit.layout)
        ),
    }

    return PreparedExecution(
        circuit=circuit,
        transpiled_circuit=transpiled_circuit,
        circuit_metadata=circuit_metadata,
        parameter_order=parameter_order,
        measurement_register=measurement_register,
        mode=executor.mode,
        backend_name=executor.backend_name,
        provenance=provenance,
        _executor=executor,
    )


def _validate_parameter_sets(
    parameter_sets: Sequence[CircuitParameterSet],
) -> tuple[CircuitParameterSet, ...]:
    """Validate a nonempty batch and its optional labels."""
    if isinstance(parameter_sets, (str, bytes)):
        raise TypeError("parameter_sets must be a sequence of parameter sets.")
    normalized = tuple(parameter_sets)
    if not normalized:
        raise ValueError("parameter_sets must contain at least one item.")

    for index, parameter_set in enumerate(normalized):
        if not isinstance(parameter_set, CircuitParameterSet):
            raise TypeError(
                f"parameter_sets[{index}] must be a CircuitParameterSet."
            )
        if parameter_set.label is not None and not isinstance(
            parameter_set.label,
            str,
        ):
            raise TypeError(f"parameter_sets[{index}].label must be a string.")

    return normalized


def _validate_shots(shots: int) -> int:
    """Validate a positive finite-shot count."""
    if not isinstance(shots, Integral) or isinstance(shots, bool):
        raise TypeError("shots must be an integer.")
    if shots <= 0:
        raise ValueError("shots must be greater than zero.")
    return int(shots)


def _build_parameter_matrix(
    prepared: PreparedExecution,
    parameter_sets: Sequence[CircuitParameterSet],
) -> tuple[tuple[float, ...], ...]:
    """Order each complete binding exactly as the ISA circuit expects it."""
    rows: list[tuple[float, ...]] = []
    for parameter_set in parameter_sets:
        bindings = make_parameter_bindings(
            prepared.circuit_metadata,
            parameter_set.initial_values,
            parameter_set.edge_values,
        )
        if set(bindings) != set(prepared.parameter_order):
            raise ValueError(
                "Parameter bindings do not match the prepared circuit."
            )
        rows.append(
            tuple(bindings[parameter] for parameter in prepared.parameter_order)
        )
    return tuple(rows)


def align_counts(
    counts: Mapping[str, int],
    ordered_genes: Sequence[str],
) -> dict[str, int]:
    """Convert Qiskit-order counts to the biological ``ordered_genes`` order."""
    if not isinstance(counts, Mapping):
        raise TypeError("counts must be a mapping.")

    aligned: dict[str, int] = {}
    for bitstring, count in counts.items():
        if not isinstance(bitstring, str):
            raise TypeError("Every counts key must be a bitstring.")
        if not isinstance(count, Integral) or isinstance(count, bool):
            raise TypeError("Every count must be an integer.")
        if count < 0:
            raise ValueError("Counts must be nonnegative.")
        aligned_bitstring = align_bitstring(bitstring, ordered_genes)
        aligned[aligned_bitstring] = (
            aligned.get(aligned_bitstring, 0) + int(count)
        )
    return aligned


def counts_to_probabilities(counts: Mapping[str, int]) -> dict[str, float]:
    """Normalize a nonempty count mapping without changing its support."""
    if not isinstance(counts, Mapping):
        raise TypeError("counts must be a mapping.")
    total = 0
    normalized_counts: dict[str, int] = {}
    for bitstring, count in counts.items():
        if not isinstance(bitstring, str):
            raise TypeError("Every counts key must be a string.")
        if not isinstance(count, Integral) or isinstance(count, bool):
            raise TypeError("Every count must be an integer.")
        if count < 0:
            raise ValueError("Counts must be nonnegative.")
        normalized_counts[bitstring] = int(count)
        total += int(count)
    if total <= 0:
        raise ValueError("counts must contain at least one sampled outcome.")
    return {
        bitstring: count / total
        for bitstring, count in normalized_counts.items()
    }


def _to_serializable(value: Any) -> Any:
    """Best-effort conversion of Qiskit metadata to JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _to_serializable(item)
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_to_serializable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _to_serializable(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _extract_sampler_counts(
    pub_result: Any,
    measurement_register: str,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Extract one classical register from a Sampler V2 PUB result."""
    data = getattr(pub_result, "data", None)
    if data is None or not hasattr(data, measurement_register):
        raise RuntimeError(
            "Sampler result does not contain the expected measurement register "
            f"{measurement_register!r}."
        )
    bit_array = getattr(data, measurement_register)
    get_counts = getattr(bit_array, "get_counts", None)
    if not callable(get_counts):
        raise RuntimeError("Sampler measurement data does not expose get_counts().")

    raw_counts = get_counts()
    if not isinstance(raw_counts, Mapping):
        raise RuntimeError("Sampler get_counts() did not return a mapping.")
    counts = dict(raw_counts)
    metadata = _to_serializable(getattr(pub_result, "metadata", {}))
    return counts, metadata


def _job_id(job: Any) -> str | None:
    """Return a job identifier when the provider exposes one."""
    identifier = getattr(job, "job_id", None)
    if callable(identifier):
        identifier = identifier()
    return identifier if isinstance(identifier, str) and identifier else None


def execute_parameter_sets(
    executor: QuantumExecutor,
    prepared: PreparedExecution,
    parameter_sets: Sequence[CircuitParameterSet],
    *,
    shots: int,
    result_timeout: float | None = None,
) -> ExecutionBatchResult:
    """Execute complete parameter sets in one finite-shot Sampler V2 job."""
    if not isinstance(executor, QuantumExecutor):
        raise TypeError("executor must be a QuantumExecutor.")
    if executor.closed:
        raise RuntimeError("executor has already been closed.")
    if not isinstance(prepared, PreparedExecution):
        raise TypeError("prepared must be a PreparedExecution.")
    if prepared._executor is not executor:
        raise ValueError("prepared execution belongs to a different executor.")
    if result_timeout is not None and (
        not isinstance(result_timeout, Real)
        or isinstance(result_timeout, bool)
        or not math.isfinite(float(result_timeout))
        or result_timeout <= 0
    ):
        raise ValueError("result_timeout must be a positive finite number.")

    normalized_shots = _validate_shots(shots)
    normalized_parameter_sets = _validate_parameter_sets(parameter_sets)
    parameter_rows = _build_parameter_matrix(
        prepared,
        normalized_parameter_sets,
    )
    pubs = [
        (prepared.transpiled_circuit, parameter_row)
        for parameter_row in parameter_rows
    ]

    submitted_at = _utc_now()
    job = executor.sampler.run(pubs, shots=normalized_shots)
    if result_timeout is None:
        primitive_result = job.result()
    else:
        primitive_result = job.result(timeout=float(result_timeout))
    completed_at = _utc_now()

    pub_results = tuple(primitive_result)
    if len(pub_results) != len(normalized_parameter_sets):
        raise RuntimeError(
            "Sampler returned a different number of results than submissions."
        )

    ordered_genes = prepared.circuit_metadata["ordered_genes"]
    samples: list[SampleResult] = []
    for parameter_set, pub_result in zip(
        normalized_parameter_sets,
        pub_results,
    ):
        raw_counts, sampler_metadata = _extract_sampler_counts(
            pub_result,
            prepared.measurement_register,
        )
        aligned = align_counts(raw_counts, ordered_genes)
        effective_shots = sum(aligned.values())
        if effective_shots != normalized_shots:
            raise RuntimeError(
                "Sampler counts do not sum to the requested shot count: "
                f"expected {normalized_shots}, received {effective_shots}."
            )
        samples.append(
            SampleResult(
                label=parameter_set.label,
                aligned_counts=aligned,
                probabilities=counts_to_probabilities(aligned),
                shots=effective_shots,
                sampler_metadata=sampler_metadata,
            )
        )

    execution_metadata = {
        "mode": executor.mode,
        "backend_name": executor.backend_name,
        "requested_shots_per_parameter_set": normalized_shots,
        "number_of_parameter_sets": len(normalized_parameter_sets),
        "submitted_at_utc": submitted_at,
        "completed_at_utc": completed_at,
        "primitive_metadata": _to_serializable(
            getattr(primitive_result, "metadata", {})
        ),
    }
    return ExecutionBatchResult(
        samples=tuple(samples),
        job_id=_job_id(job),
        execution_metadata=execution_metadata,
    )


def close_quantum_executor(executor: QuantumExecutor) -> None:
    """Close an executor's Runtime session, if one was opened."""
    if not isinstance(executor, QuantumExecutor):
        raise TypeError("executor must be a QuantumExecutor.")
    if executor.closed:
        return
    if executor.session is not None:
        executor.session.close()
    executor.closed = True


def main() -> None:
    """Run a two-parameter-set finite-shot smoke test on noiseless Aer."""
    import networkx as nx

    from quantum_circuit import build_parameterized_circuit

    graph = nx.DiGraph(
        [
            ("GENE_A", "GENE_B"),
            ("GENE_A", "GENE_C"),
            ("GENE_B", "GENE_C"),
        ]
    )
    circuit, metadata = build_parameterized_circuit(graph)
    executor = create_quantum_executor(
        "aer_noiseless",
        seed_simulator=42,
    )

    try:
        prepared = prepare_execution(
            executor,
            circuit,
            metadata,
            optimization_level=1,
            seed_transpiler=42,
        )
        result = execute_parameter_sets(
            executor,
            prepared,
            [
                CircuitParameterSet(
                    initial_values=(0.2, 0.4, 0.6),
                    edge_values=(0.1, -0.2, 0.3),
                    label="baseline",
                ),
                CircuitParameterSet(
                    initial_values=(0.7, 0.4, 0.6),
                    edge_values=(-0.1, 0.2, -0.3),
                    label="intervention_like",
                ),
            ],
            shots=2048,
        )
    finally:
        close_quantum_executor(executor)

    print("Backend:", prepared.backend_name)
    print("Transpiled parameter order:", prepared.provenance["parameter_order"])
    for sample in result.samples:
        print(f"{sample.label}: {sample.shots} shots, {sample.aligned_counts}")


if __name__ == "__main__":
    main()
