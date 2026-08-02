"""Construct and parameterize prior-aware gene-regulatory quantum circuits.

The functions in this module define circuit structure only. Backend selection,
transpilation, execution, and parameter training belong in separate modules.
Running this file directly performs a small, self-contained smoke test.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import gzip
import json
import math
from numbers import Real
from pathlib import Path
import pickle
from typing import Any

import networkx as nx
from qiskit.circuit import Parameter, ParameterVector, QuantumCircuit


SUPPORTED_ROTATION_AXES = frozenset({"x", "y"})


def load_gene_graph(file_path: str | Path) -> nx.DiGraph:
    """Load a gene-regulatory graph from a trusted gzip-compressed pickle.

    Pickle files can execute arbitrary code while loading. Only use this
    function with graph files created by you or obtained from a trusted source.
    """
    path = Path(file_path)

    with gzip.open(path, "rb") as file:
        graph = pickle.load(file)

    if not isinstance(graph, nx.DiGraph) or isinstance(graph, nx.MultiDiGraph):
        raise TypeError("The saved object must be a NetworkX DiGraph.")

    return graph


def _validate_gene_graph(graph: nx.DiGraph) -> None:
    """Validate the graph properties required by circuit construction."""
    if not isinstance(graph, nx.DiGraph) or isinstance(graph, nx.MultiDiGraph):
        raise TypeError("graph must be a NetworkX DiGraph.")

    if graph.number_of_nodes() == 0:
        raise ValueError("graph must contain at least one gene.")

    non_string_nodes = [node for node in graph.nodes if not isinstance(node, str)]
    if non_string_nodes:
        raise TypeError(
            "All graph nodes must be gene-name strings. "
            f"Invalid nodes: {non_string_nodes!r}."
        )

    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("graph must be a directed acyclic graph (DAG).")


def _normalize_rotation_axis(axis: str, parameter_name: str) -> str:
    """Return a supported, lower-case rotation-axis name."""
    if not isinstance(axis, str):
        raise TypeError(f"{parameter_name} must be a string.")

    normalized_axis = axis.lower()
    if normalized_axis not in SUPPORTED_ROTATION_AXES:
        raise ValueError(f"{parameter_name} must be either 'x' or 'y'.")

    return normalized_axis


def _resolve_rotation_axes(
    rotation_axis: str,
    initial_rotation_axis: str | None,
    controlled_rotation_axis: str | None,
) -> tuple[str, str]:
    """Resolve shared and independently overridden rotation axes."""
    shared_axis = _normalize_rotation_axis(rotation_axis, "rotation_axis")

    initial_axis = (
        shared_axis
        if initial_rotation_axis is None
        else _normalize_rotation_axis(
            initial_rotation_axis,
            "initial_rotation_axis",
        )
    )
    controlled_axis = (
        shared_axis
        if controlled_rotation_axis is None
        else _normalize_rotation_axis(
            controlled_rotation_axis,
            "controlled_rotation_axis",
        )
    )

    return initial_axis, controlled_axis


def make_qubit_mapping(
    graph: nx.DiGraph,
) -> tuple[list[str], dict[str, int]]:
    """Create a deterministic topological gene-to-qubit mapping.

    Graph dependencies determine the order. Gene names break ties whenever
    multiple nodes are simultaneously available.
    """
    _validate_gene_graph(graph)

    ordered_genes = list(nx.lexicographical_topological_sort(graph, key=str))
    gene_to_qubit = {
        gene: qubit_index
        for qubit_index, gene in enumerate(ordered_genes)
    }

    return ordered_genes, gene_to_qubit


def make_edge_schedule(
    graph: nx.DiGraph,
    ordered_genes: Sequence[str],
) -> list[tuple[str, str]]:
    """Create a deterministic edge order for controlled rotations.

    Source genes follow ``ordered_genes``. Targets of the same source are
    ordered alphabetically.
    """
    _validate_gene_graph(graph)

    graph_genes = set(graph.nodes)
    supplied_genes = list(ordered_genes)

    if len(supplied_genes) != len(set(supplied_genes)):
        raise ValueError("ordered_genes must not contain duplicates.")

    if set(supplied_genes) != graph_genes:
        raise ValueError(
            "ordered_genes must contain every graph gene exactly once."
        )

    return [
        (source, target)
        for source in supplied_genes
        for target in sorted(graph.successors(source))
    ]


def build_parameterized_circuit(
    graph: nx.DiGraph,
    rotation_axis: str = "y",
    initial_rotation_axis: str | None = None,
    controlled_rotation_axis: str | None = None,
    add_measurements: bool = True,
) -> tuple[QuantumCircuit, dict[str, Any]]:
    """Build a fully parameterized gene-regulatory quantum circuit.

    Every gene is represented by one qubit and one symbolic initial-rotation
    parameter. Every directed edge is represented by one symbolic controlled-
    rotation parameter. No numerical values are assigned by this function.

    ``rotation_axis`` supplies the shared default for both gate families.
    ``initial_rotation_axis`` and ``controlled_rotation_axis`` independently
    override it, allowing combinations such as RY+CRX or RX+CRY.

    Parameters
    ----------
    graph
        Gene-regulatory DAG whose nodes are gene-name strings.
    rotation_axis
        Shared axis used when a gate-family-specific override is not supplied.
        Must be ``"x"`` or ``"y"``.
    initial_rotation_axis
        Optional axis override for the single-qubit initial rotations.
    controlled_rotation_axis
        Optional axis override for the edge-controlled rotations.
    add_measurements
        If true, measure every qubit into the single classical register created
        by :meth:`QuantumCircuit.measure_all`.

    Returns
    -------
    circuit
        Circuit parameterized by every initial and edge rotation angle.
    metadata
        In-memory structural metadata, including deterministic biological
        ordering and parameter-to-gene/edge mappings.
    """
    if not isinstance(add_measurements, bool):
        raise TypeError("add_measurements must be a Boolean.")

    initial_axis, controlled_axis = _resolve_rotation_axes(
        rotation_axis,
        initial_rotation_axis,
        controlled_rotation_axis,
    )
    ordered_genes, gene_to_qubit = make_qubit_mapping(graph)
    scheduled_edges = make_edge_schedule(graph, ordered_genes)

    initial_angles = ParameterVector("initial_angle", len(ordered_genes))
    edge_angles = ParameterVector("edge_angle", len(scheduled_edges))

    circuit = QuantumCircuit(
        len(ordered_genes),
        name=f"gene_dag_r{initial_axis}_cr{controlled_axis}",
    )

    initial_rotation = circuit.rx if initial_axis == "x" else circuit.ry
    controlled_rotation = (
        circuit.crx if controlled_axis == "x" else circuit.cry
    )

    for qubit_index, initial_angle in enumerate(initial_angles):
        initial_rotation(initial_angle, qubit_index)

    for edge_index, (source, target) in enumerate(scheduled_edges):
        controlled_rotation(
            edge_angles[edge_index],
            gene_to_qubit[source],
            gene_to_qubit[target],
        )

    if add_measurements:
        circuit.measure_all()

    metadata: dict[str, Any] = {
        "ordered_genes": ordered_genes,
        "scheduled_edges": scheduled_edges,
        "initial_rotation_axis": initial_axis,
        "controlled_rotation_axis": controlled_axis,
        "add_measurements": add_measurements,
        "initial_angles": initial_angles,
        "edge_angles": edge_angles,
        "gene_to_initial_parameter": {
            gene: initial_angles[index]
            for index, gene in enumerate(ordered_genes)
        },
        "edge_to_parameter": {
            edge: edge_angles[index]
            for index, edge in enumerate(scheduled_edges)
        },
    }

    return circuit, metadata


def _validate_parameter_values(
    values: Sequence[float],
    expected_length: int,
    parameter_name: str,
) -> tuple[float, ...]:
    """Validate and normalize one numerical parameter sequence."""
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{parameter_name} must be a sequence of numbers.")

    normalized_values = tuple(values)
    if len(normalized_values) != expected_length:
        raise ValueError(
            f"{parameter_name} must contain {expected_length} values; "
            f"received {len(normalized_values)}."
        )

    for index, value in enumerate(normalized_values):
        if not isinstance(value, Real) or isinstance(value, bool):
            raise TypeError(
                f"{parameter_name}[{index}] must be a real number."
            )
        if not math.isfinite(float(value)):
            raise ValueError(
                f"{parameter_name}[{index}] must be finite."
            )

    return tuple(float(value) for value in normalized_values)


def make_parameter_bindings(
    metadata: Mapping[str, Any],
    initial_values: Sequence[float],
    edge_values: Sequence[float],
) -> dict[Parameter, float]:
    """Map one complete set of numerical values to circuit parameters.

    Both value sequences are required on every call. During training,
    ``initial_values`` can remain fixed while ``edge_values`` changes. For an
    intervention run, both sequences can be replaced through the same path.
    """
    try:
        initial_angles = metadata["initial_angles"]
        edge_angles = metadata["edge_angles"]
    except KeyError as error:
        raise KeyError(
            "metadata must contain 'initial_angles' and 'edge_angles'."
        ) from error

    normalized_initial_values = _validate_parameter_values(
        initial_values,
        len(initial_angles),
        "initial_values",
    )
    normalized_edge_values = _validate_parameter_values(
        edge_values,
        len(edge_angles),
        "edge_values",
    )

    return {
        **dict(zip(initial_angles, normalized_initial_values)),
        **dict(zip(edge_angles, normalized_edge_values)),
    }


def bind_circuit_parameters(
    circuit: QuantumCircuit,
    metadata: Mapping[str, Any],
    initial_values: Sequence[float],
    edge_values: Sequence[float],
) -> QuantumCircuit:
    """Return a circuit copy with all initial and edge angles assigned."""
    parameter_bindings = make_parameter_bindings(
        metadata,
        initial_values,
        edge_values,
    )

    expected_parameters = set(parameter_bindings)
    circuit_parameters = set(circuit.parameters)
    if circuit_parameters != expected_parameters:
        missing = sorted(
            str(parameter)
            for parameter in circuit_parameters - expected_parameters
        )
        unexpected = sorted(
            str(parameter)
            for parameter in expected_parameters - circuit_parameters
        )
        raise ValueError(
            "Circuit and metadata parameters do not match. "
            f"Unbound by metadata: {missing}; absent from circuit: {unexpected}."
        )

    return circuit.assign_parameters(parameter_bindings, inplace=False)


def align_bitstring(bitstring: str, ordered_genes: Sequence[str]) -> str:
    """Reverse a Qiskit bitstring to match ``ordered_genes`` order.

    This helper expects output from the single classical register produced by
    ``measure_all()``. Bitstrings containing register separators are rejected.
    """
    if not isinstance(bitstring, str):
        raise TypeError("bitstring must be a string.")

    if set(bitstring) - {"0", "1"}:
        raise ValueError("bitstring must contain only '0' and '1'.")

    if len(bitstring) != len(ordered_genes):
        raise ValueError(
            "The bitstring length must equal the number of ordered genes."
        )

    return bitstring[::-1]


def metadata_to_serializable(
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Convert in-memory circuit metadata to a JSON-compatible structure."""
    required_fields = {
        "ordered_genes",
        "scheduled_edges",
        "initial_rotation_axis",
        "controlled_rotation_axis",
        "add_measurements",
    }
    missing_fields = sorted(required_fields - set(metadata))
    if missing_fields:
        raise KeyError(f"Metadata is missing required fields: {missing_fields}.")

    return {
        "ordered_genes": list(metadata["ordered_genes"]),
        "scheduled_edges": [
            list(edge)
            for edge in metadata["scheduled_edges"]
        ],
        "initial_rotation_axis": metadata["initial_rotation_axis"],
        "controlled_rotation_axis": metadata["controlled_rotation_axis"],
        "add_measurements": metadata["add_measurements"],
    }


def save_circuit_metadata(
    metadata: Mapping[str, Any],
    file_path: str | Path,
) -> None:
    """Save circuit-structure metadata as JSON, creating its directory."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata_to_serializable(metadata), file, indent=2)
        file.write("\n")


def load_circuit_metadata(file_path: str | Path) -> dict[str, Any]:
    """Load and validate JSON circuit-structure metadata."""
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if not isinstance(metadata, dict):
        raise TypeError("Circuit metadata must be a JSON object.")

    required_fields = {
        "ordered_genes",
        "scheduled_edges",
        "initial_rotation_axis",
        "controlled_rotation_axis",
        "add_measurements",
    }
    missing_fields = sorted(required_fields - set(metadata))
    if missing_fields:
        raise ValueError(f"Metadata is missing required fields: {missing_fields}.")

    ordered_genes = metadata["ordered_genes"]
    if (
        not isinstance(ordered_genes, list)
        or not ordered_genes
        or any(not isinstance(gene, str) for gene in ordered_genes)
        or len(ordered_genes) != len(set(ordered_genes))
    ):
        raise ValueError(
            "ordered_genes must be a non-empty list of unique strings."
        )

    raw_edges = metadata["scheduled_edges"]
    if not isinstance(raw_edges, list):
        raise ValueError("scheduled_edges must be a list.")

    scheduled_edges: list[tuple[str, str]] = []
    known_genes = set(ordered_genes)
    for index, edge in enumerate(raw_edges):
        if (
            not isinstance(edge, list)
            or len(edge) != 2
            or any(not isinstance(gene, str) for gene in edge)
        ):
            raise ValueError(
                f"scheduled_edges[{index}] must contain two gene strings."
            )
        source, target = edge
        if source not in known_genes or target not in known_genes:
            raise ValueError(
                f"scheduled_edges[{index}] references an unknown gene."
            )
        scheduled_edges.append((source, target))

    if len(scheduled_edges) != len(set(scheduled_edges)):
        raise ValueError("scheduled_edges must not contain duplicates.")

    initial_axis = _normalize_rotation_axis(
        metadata["initial_rotation_axis"],
        "initial_rotation_axis",
    )
    controlled_axis = _normalize_rotation_axis(
        metadata["controlled_rotation_axis"],
        "controlled_rotation_axis",
    )
    if not isinstance(metadata["add_measurements"], bool):
        raise TypeError("add_measurements must be a Boolean.")

    return {
        "ordered_genes": ordered_genes,
        "scheduled_edges": scheduled_edges,
        "initial_rotation_axis": initial_axis,
        "controlled_rotation_axis": controlled_axis,
        "add_measurements": metadata["add_measurements"],
    }


def main() -> None:
    """Run a portable smoke test of construction, binding, and alignment."""
    test_graph = nx.DiGraph(
        [
            ("GENE_A", "GENE_B"),
            ("GENE_A", "GENE_C"),
            ("GENE_B", "GENE_C"),
        ]
    )

    circuit, metadata = build_parameterized_circuit(test_graph)
    initial_values = [0.20, 0.35, 0.50]
    edge_values = [0.05, -0.10, 0.15]
    bound_circuit = bind_circuit_parameters(
        circuit,
        metadata,
        initial_values,
        edge_values,
    )

    if bound_circuit.parameters:
        raise RuntimeError("Smoke test failed: parameters remain unbound.")

    print("Ordered genes:", metadata["ordered_genes"])
    print("Scheduled edges:", metadata["scheduled_edges"])
    print("Parameterized circuit:")
    print(circuit.draw(output="text"))
    print("Bound circuit has no free parameters.")
    print(
        "Aligned example bitstring:",
        align_bitstring("101", metadata["ordered_genes"]),
    )


if __name__ == "__main__":
    main()
