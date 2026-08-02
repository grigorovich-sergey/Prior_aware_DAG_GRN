from io import StringIO
from copy import deepcopy
import re
import numpy as np
import pandas as pd
import requests
import sys
import yaml
import argparse
import gzip
import pickle
from pathlib import Path
import warnings
import networkx as nx
import matplotlib.pyplot as plt


REGULATORY_SIGN_VALUES = {
    "activation": 1,
    "repression": -1,
    "mixed": 0,
    "unknown": 0,
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Select a prior-aware gene-regulatory DAG."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Optional YAML file whose values override configs/default.yaml."
        ),
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """
    Load the configuration from YAML.

    Args:
        config_path: path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.
    """
    path = Path(config_path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file does not exist: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration file '{path}' must contain "
            "a top-level YAML mapping."
        )

    return config


def merge_config(base: dict, overrides: dict) -> dict:
    """Recursively merge override values into a copy of the base config."""
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_effective_config(override_path: str | None) -> dict:
    """Load the repository default config and apply an optional short override."""
    default_path = Path(__file__).resolve().parents[1] / "configs" / "default_genes.yaml"
    config = load_config(default_path)
    if override_path is not None:
        config = merge_config(config, load_config(override_path))
    return config


def resolve_project_path(path: str | Path) -> Path:
    """Resolve relative input/output paths from the repository root."""
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[1] / path


def validate_config(config: dict) -> None:
    """Validate settings whose relationships are required by the pipeline."""
    generation = config["geneset_generation"]
    anchors = normalize_gene_symbols(generation["anchors"])
    layers_genes = generation.get("layers_genes")

    if generation["num_layers"] < 1:
        raise ValueError("geneset_generation.num_layers must be at least 1.")

    counts = generation["number_by_layer"]
    if layers_genes is None and len(counts) != generation["num_layers"]:
        raise ValueError(
            "geneset_generation.number_by_layer must contain exactly "
            "num_layers entries when layers_genes is null."
        )
    if any(not isinstance(value, int) or value < 0 for value in counts):
        raise ValueError("number_by_layer values must be non-negative integers.")

    if layers_genes is not None:
        if not isinstance(layers_genes, list) or not layers_genes:
            raise ValueError(
                "geneset_generation.layers_genes must be null or a "
                "non-empty gene list."
            )
        if any(not isinstance(gene, str) for gene in layers_genes):
            raise ValueError("Every layers_genes entry must be a gene symbol string.")
        normalized = normalize_gene_symbols(layers_genes)
        overlap = set(anchors) & set(normalized)
        if overlap:
            raise ValueError(
                "Anchors must not be repeated in layers_genes: "
                + ", ".join(sorted(overlap))
            )

    if config["plotting"]["minimum_peeled_candidates"] < 1:
        raise ValueError("plotting.minimum_peeled_candidates must be at least 1.")

def fetch_collectri_from_omnipath(
    config: dict,
    timeout: int = 60,
    require_directed: bool = True,
    require_signed: bool = True,
    include_self_loops: bool = True,
    license_type: str = "academic",
) -> pd.DataFrame:
    """
    Download human CollecTRI TF-target interactions from OmniPath.

    Parameters
    ----------
    timeout
        Maximum time, in seconds, to wait for the API response.
    require_directed
        If True, request only interactions with known direction.
    require_signed
        If True, request only interactions with a known activation or
        inhibition sign.
    include_self_loops
        If True, retain self-regulatory interactions in the downloaded data.
    license_type
        OmniPath license filter. For academic research, use "academic".

    Returns
    -------
    pandas.DataFrame
        Raw table returned by the OmniPath API.
    """
    
    OMNIPATH_INTERACTIONS_URL = config["geneset_generation"]["omnipath_url"]
    
    params = {
        "datasets": "collectri",
        "types": "transcriptional",
        "organisms": 9606,
        "genesymbols": "yes",
        "directed": "yes" if require_directed else "no",
        "signed": "yes" if require_signed else "no",
        "loops": "yes" if include_self_loops else "no",
        "fields": "sources,references,curation_effort,type",
        "license": license_type,
        "format": "tsv",
    }

    try:
        response = requests.get(
            OMNIPATH_INTERACTIONS_URL,
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()

    except requests.RequestException as error:
        raise RuntimeError(
            "Could not retrieve CollecTRI interactions from the OmniPath API."
        ) from error

    if not response.text.strip():
        raise RuntimeError("The OmniPath API returned an empty response.")

    try:
        raw_interactions = pd.read_csv(
            StringIO(response.text),
            sep="\t",
            low_memory=False,
        )
    except Exception as error:
        raise RuntimeError(
            "The OmniPath response could not be parsed as a TSV table."
        ) from error

    if raw_interactions.empty:
        raise RuntimeError("The OmniPath API returned no CollecTRI interactions.")

    return raw_interactions

def _convert_to_boolean(column: pd.Series) -> pd.Series:
    """
    Convert OmniPath's 0/1 or text Boolean columns to Python Booleans.
    """
    if pd.api.types.is_bool_dtype(column):
        return column

    if pd.api.types.is_numeric_dtype(column):
        return column.fillna(0).astype(int).astype(bool)

    return (
        column.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes"})
    )

def _derive_sign(interactions: pd.DataFrame) -> pd.Series:
    """
    Derive a readable sign label from OmniPath stimulation/inhibition fields.
    """
    stimulation = interactions["is_stimulation"]
    inhibition = interactions["is_inhibition"]

    return pd.Series(
        np.select(
            condlist=[
                stimulation & ~inhibition,
                inhibition & ~stimulation,
                stimulation & inhibition,
            ],
            choicelist=[
                "activation",
                "repression",
                "mixed",
            ],
            default="unknown",
        ),
        index=interactions.index,
        dtype="string",
    )

def standardize_collectri_columns(
    raw_interactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert the raw OmniPath table to the schema used by later analysis.

    The function preserves the original UniProt or complex identifiers,
    gene-symbol labels, directionality, sign, evidence sources, references,
    and curation effort.
    """
    required_columns = {
        "source",
        "target",
        "source_genesymbol",
        "target_genesymbol",
        "is_directed",
        "is_stimulation",
        "is_inhibition",
    }

    missing_columns = required_columns - set(raw_interactions.columns)

    if missing_columns:
        available = ", ".join(raw_interactions.columns)
        missing = ", ".join(sorted(missing_columns))

        raise ValueError(
            f"Expected OmniPath columns are missing: {missing}.\n"
            f"Available columns: {available}"
        )

    interactions = raw_interactions.copy()

    boolean_columns = [
        "is_directed",
        "is_stimulation",
        "is_inhibition",
        "consensus_direction",
        "consensus_stimulation",
        "consensus_inhibition",
    ]

    for column in boolean_columns:
        if column in interactions.columns:
            interactions[column] = _convert_to_boolean(interactions[column])

    interactions = interactions.rename(
        columns={
            "source": "regulator_id",
            "target": "target_id",
            "source_genesymbol": "regulator",
            "target_genesymbol": "target",
        }
    )

    interactions["regulator"] = (
        interactions["regulator"]
        .astype("string")
        .str.strip()
    )
    interactions["target"] = (
        interactions["target"]
        .astype("string")
        .str.strip()
    )

    interactions = interactions.dropna(
        subset=["regulator", "target"]
    )

    interactions = interactions[
        (interactions["regulator"] != "")
        & (interactions["target"] != "")
    ].copy()

    interactions["sign"] = _derive_sign(interactions)

    preferred_column_order = [
        "regulator",
        "target",
        "regulator_id",
        "target_id",
        "is_directed",
        "is_stimulation",
        "is_inhibition",
        "sign",
        "consensus_direction",
        "consensus_stimulation",
        "consensus_inhibition",
        "sources",
        "references",
        "curation_effort",
        "type",
    ]

    available_columns = [
        column
        for column in preferred_column_order
        if column in interactions.columns
    ]

    interactions = interactions[available_columns]
    interactions = interactions.drop_duplicates().reset_index(drop=True)

    return interactions

def load_collectri_from_omnipath(
    config: dict,
    timeout: int = 60,
    require_directed: bool = True,
    require_signed: bool = True,
    include_self_loops: bool = True,
    license_type: str = "academic",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Download and standardize human CollecTRI interactions.
    """
    raw_interactions = fetch_collectri_from_omnipath(
        timeout=timeout,
        require_directed=require_directed,
        require_signed=require_signed,
        include_self_loops=include_self_loops,
        license_type=license_type,
        config=config,
    )

    interactions = standardize_collectri_columns(raw_interactions)

    if verbose:
        number_of_regulators = interactions["regulator"].nunique()
        number_of_targets = interactions["target"].nunique()

        number_of_self_loops = (
            interactions["regulator"] == interactions["target"]
        ).sum()

        sign_counts = interactions["sign"].value_counts(dropna=False)

        print("CollecTRI data loaded from OmniPath")
        print(f"Interactions:      {len(interactions):,}")
        print(f"Unique regulators: {number_of_regulators:,}")
        print(f"Unique targets:    {number_of_targets:,}")
        print(f"Self-loops:        {number_of_self_loops:,}")
        print("\nSigns:")
        print(sign_counts.to_string())

    return interactions

def normalize_gene_symbols(genes: list[str] | tuple[str, ...] | set[str]) -> list[str]:
    """
    Normalize gene symbols while preserving their original order.

    Normalization:
    - remove surrounding whitespace;
    - convert to uppercase;
    - remove empty values;
    - remove duplicates.
    """
    normalized = []

    for gene in genes:
        if gene is None:
            continue

        symbol = str(gene).strip().upper()

        if symbol and symbol not in normalized:
            normalized.append(symbol)

    if not normalized:
        raise ValueError("No valid anchor gene symbols were provided.")

    return normalized

def validate_gene_symbols(
    interactions: pd.DataFrame,
    genes: list[str] | tuple[str, ...] | set[str],
    list_name: str,
) -> list[str]:
    """Return normalized genes only when every symbol exists in CollecTRI."""
    required_columns = {"regulator", "target"}
    missing_columns = required_columns - set(interactions.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Interaction table is missing required columns: {missing_text}"
        )

    normalized_genes = normalize_gene_symbols(genes)
    available_symbols = (
        set(interactions["regulator"].dropna())
        | set(interactions["target"].dropna())
    )
    missing_genes = [
        gene for gene in normalized_genes if gene not in available_symbols
    ]

    if missing_genes:
        raise ValueError(
            f"Unknown gene symbol(s) in {list_name}: "
            + ", ".join(missing_genes)
            + ". Every provided gene must exactly match a symbol in the "
              "current CollecTRI interaction table."
        )

    return normalized_genes

def validate_anchors(
    interactions: pd.DataFrame,
    anchors: list[str] | tuple[str, ...] | set[str],
) -> tuple[list[str], pd.DataFrame]:
    """
    Validate anchor genes against a standardized CollecTRI interaction table.

    Parameters
    ----------
    interactions
        Standardized interaction table containing `regulator` and `target`.
    anchors
        Manually selected anchor gene symbols.

    Returns
    -------
    normalized_anchors
        Normalized anchor symbols in the original input order.
    anchor_summary
        Regulatory coverage summary for each anchor.
    """
    normalized_anchors = validate_gene_symbols(
        interactions,
        anchors,
        list_name="anchors",
    )
    available_symbol_set = (
        set(interactions["regulator"].dropna())
        | set(interactions["target"].dropna())
    )

    summary_rows = []

    for anchor in normalized_anchors:
        incoming = interactions[
            interactions["target"] == anchor
        ]

        outgoing = interactions[
            interactions["regulator"] == anchor
        ]

        upstream_regulators = set(incoming["regulator"])
        downstream_targets = set(outgoing["target"])

        upstream_regulators.discard(anchor)
        downstream_targets.discard(anchor)

        self_loop_count = (
            (interactions["regulator"] == anchor)
            & (interactions["target"] == anchor)
        ).sum()

        summary_rows.append(
            {
                "anchor": anchor,
                "present_in_interactions": anchor in available_symbol_set,
                "incoming_rows": len(incoming),
                "outgoing_rows": len(outgoing),
                "unique_upstream_regulators": len(upstream_regulators),
                "unique_downstream_targets": len(downstream_targets),
                "self_loops": int(self_loop_count),
            }
        )

    anchor_summary = pd.DataFrame(summary_rows)

    return normalized_anchors, anchor_summary

def build_upstream_candidate_pool(
    interactions: pd.DataFrame,
    anchors: list[str] | set[str],
    number_of_layers: int = 2,
) -> set[str]:
    """
    Find unique upstream regulators within a specified number of layers.

    Anchors are treated as Layer 0 and are not included in the returned
    candidate pool.

    Parameters
    ----------
    interactions
        Standardized CollecTRI table with `regulator` and `target` columns.
    anchors
        Validated anchor gene symbols.
    number_of_layers
        Number of upstream regulatory layers to traverse.

    Returns
    -------
    set[str]
        Unique upstream candidate genes, excluding anchors.
    """
    if number_of_layers < 1:
        raise ValueError("number_of_layers must be at least 1.")

    anchor_set = set(anchors)
    candidate_pool = set()

    # Start by looking for regulators of the anchors.
    current_targets = anchor_set.copy()
    visited_genes = anchor_set.copy()

    for _ in range(number_of_layers):
        upstream_sources = set(
            interactions.loc[
                interactions["target"].isin(current_targets),
                "regulator",
            ].dropna()
        )

        # Keep only genes that have not already been encountered.
        new_sources = upstream_sources - visited_genes

        if not new_sources:
            break

        candidate_pool.update(new_sources)
        visited_genes.update(new_sources)

        # In the next iteration, find regulators of this new layer.
        current_targets = new_sources

    return candidate_pool

def build_candidate_graph(
    interactions: pd.DataFrame,
    anchors: list[str] | set[str],
    candidate_pool: set[str],
) -> tuple[pd.DataFrame, nx.DiGraph]:
    """
    Build a directed regulatory graph containing anchors and candidates.

    Only interactions where both regulator and target are in the combined
    gene set are retained.

    Returns
    -------
    selected_interactions
        Subset of the CollecTRI table used to construct the graph.
    graph
        Directed NetworkX graph.
    """
    anchor_set = set(anchors)
    network_genes = anchor_set | set(candidate_pool)

    selected_interactions = interactions[
        interactions["regulator"].isin(network_genes)
        & interactions["target"].isin(network_genes)
    ].copy()

    graph = nx.from_pandas_edgelist(
        selected_interactions,
        source="regulator",
        target="target",
        edge_attr=True,
        create_using=nx.DiGraph,
    )

    # Ensure anchors or candidates without retained edges are still included.
    graph.add_nodes_from(network_genes)

    # Mark anchors for later use.
    nx.set_node_attributes(
        graph,
        {
            gene: gene in anchor_set
            for gene in network_genes
        },
        name="is_anchor",
    )

    return selected_interactions, graph

def assign_minimal_layers(
    graph: nx.DiGraph,
    anchors: list[str] | set[str],
) -> dict[str, int]:
    """
    Assign each node its minimum upstream distance from an anchor.

    Anchors are Layer 0, direct anchor regulators are Layer 1, and so on.
    """
    anchor_set = set(anchors)

    layer_by_node = {
        gene: 0
        for gene in anchor_set
    }

    current_targets = anchor_set.copy()
    layer = 1

    while current_targets:
        new_sources = set()

        for target in current_targets:
            if target in graph:
                new_sources.update(graph.predecessors(target))

        # Nodes already assigned have a lower minimum layer.
        new_sources -= set(layer_by_node)

        if not new_sources:
            break

        for gene in new_sources:
            layer_by_node[gene] = layer

        current_targets = new_sources
        layer += 1

    unassigned = set(graph.nodes) - set(layer_by_node)

    if unassigned:
        warnings.warn(
            f"{len(unassigned)} graph nodes are not upstream-connected "
            "to any anchor."
        )

    # Also store the layer directly on the graph nodes.
    nx.set_node_attributes(
        graph,
        layer_by_node,
        name="layer",
    )

    return layer_by_node


def reconstruct_selected_gene_layers(
    graph: nx.DiGraph,
    anchors: list[str] | set[str],
    fallback_layer: int = 1,
) -> dict[str, int]:
    """Infer expert-selected gene layers using the candidate-pool logic.

    Layers are minimum upstream distances from an anchor. Selected genes that
    are not upstream-connected to any anchor keep the initial fallback layer.
    """
    layer_by_node = assign_minimal_layers(graph, anchors)
    unassigned = set(graph.nodes) - set(layer_by_node)

    if unassigned:
        fallback = {gene: fallback_layer for gene in unassigned}
        layer_by_node.update(fallback)
        nx.set_node_attributes(graph, fallback, name="layer")

    return layer_by_node

def select_layer_1(
    graph: nx.DiGraph,
    layer_by_node: dict[str, int],
    anchors: list[str] | set[str],
    number_to_keep: int,
    min_sources: int = 2,
    weight_anchor_targets: float = 3.0,
    weight_same_layer_targets: float = 2.0,
    weight_upstream_sources: float = 1.0,
) -> tuple[set[str], pd.DataFrame]:
    """
    Rank Layer 1 genes and enforce minimum anchor-source coverage.

    Returns
    -------
    selected
        Selected Layer 1 gene names.
    ranking
        Ranked Layer 1 table with component scores and final selection.
    """
    if number_to_keep < 0:
        raise ValueError("number_to_keep must be non-negative.")

    if min_sources < 0:
        raise ValueError("min_sources must be non-negative.")

    anchor_set = set(anchors)

    layer1 = {
        gene
        for gene, layer in layer_by_node.items()
        if layer == 1
    }

    layer2 = {
        gene
        for gene, layer in layer_by_node.items()
        if layer == 2
    }

    ranking_rows = []

    for gene in layer1:
        targets = set(graph.successors(gene))
        sources = set(graph.predecessors(gene))

        anchor_targets = targets & anchor_set

        same_layer_targets = (
            targets & layer1
        ) - {gene}

        upstream_sources = (
            sources & (layer1 | layer2)
        ) - {gene}

        score = (
            weight_anchor_targets * len(anchor_targets)
            + weight_same_layer_targets * len(same_layer_targets)
            + weight_upstream_sources * len(upstream_sources)
        )

        ranking_rows.append(
            {
                "gene": gene,
                "score": score,
                "anchor_targets": len(anchor_targets),
                "same_layer_targets": len(same_layer_targets),
                "upstream_sources": len(upstream_sources),
            }
        )

    ranking = pd.DataFrame(ranking_rows)

    if ranking.empty:
        ranking["selected"] = pd.Series(dtype=bool)
        return set(), ranking

    ranking = ranking.sort_values(
        [
            "score",
            "anchor_targets",
            "same_layer_targets",
            "upstream_sources",
            "gene",
        ],
        ascending=[
            False,
            False,
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    keep = min(number_to_keep, len(ranking))

    # Initial score-based selection.
    selected = set(
        ranking.head(keep)["gene"]
    )

    rank_position = {
        gene: position
        for position, gene in enumerate(ranking["gene"])
    }

    # Available direct Layer 1 sources for every anchor.
    anchor_sources = {
        anchor: set(graph.predecessors(anchor)) & layer1
        for anchor in anchor_set
    }

    required_sources = {}

    for anchor, available_sources in anchor_sources.items():
        required_sources[anchor] = min(
            min_sources,
            len(available_sources),
        )

        if len(available_sources) < min_sources:
            warnings.warn(
                f"Anchor {anchor} has only "
                f"{len(available_sources)} available Layer 1 source(s); "
                f"requested minimum is {min_sources}. "
                "All available sources will be enforced."
            )

    def calculate_coverage(nodes: set[str]) -> dict[str, int]:
        return {
            anchor: len(nodes & sources)
            for anchor, sources in anchor_sources.items()
        }

    # Replace low-ranked genes until anchor coverage is satisfied,
    # or no fixed-size repair is possible.
    while True:
        coverage = calculate_coverage(selected)

        deficient_anchors = {
            anchor
            for anchor in anchor_set
            if coverage[anchor] < required_sources[anchor]
        }

        if not deficient_anchors:
            break

        unselected_options = layer1 - selected

        # Only consider genes that help at least one deficient anchor.
        unselected_options = {
            gene
            for gene in unselected_options
            if any(
                gene in anchor_sources[anchor]
                for anchor in deficient_anchors
            )
        }

        if not unselected_options:
            break

        # Prefer genes that help several deficient anchors.
        # Original ranking is used as the tie-breaker.
        gene_to_add = sorted(
            unselected_options,
            key=lambda gene: (
                -sum(
                    gene in anchor_sources[anchor]
                    for anchor in deficient_anchors
                ),
                rank_position[gene],
            ),
        )[0]

        selected.add(gene_to_add)

        if len(selected) > keep:
            coverage_after_addition = calculate_coverage(selected)

            # Do not undo coverage already achieved by the addition.
            coverage_thresholds = {
                anchor: min(
                    required_sources[anchor],
                    coverage_after_addition[anchor],
                )
                for anchor in anchor_set
            }

            removable_genes = []

            for gene in selected - {gene_to_add}:
                trial_selection = selected - {gene}
                trial_coverage = calculate_coverage(trial_selection)

                can_remove = all(
                    trial_coverage[anchor]
                    >= coverage_thresholds[anchor]
                    for anchor in anchor_set
                )

                if can_remove:
                    removable_genes.append(gene)

            if not removable_genes:
                # The new source cannot fit without breaking another
                # anchor requirement.
                selected.remove(gene_to_add)
                break

            # Remove the lowest-ranked safely removable gene.
            gene_to_remove = max(
                removable_genes,
                key=lambda gene: rank_position[gene],
            )

            selected.remove(gene_to_remove)

    final_coverage = calculate_coverage(selected)

    unmet_requirements = {
        anchor: (
            final_coverage[anchor],
            required_sources[anchor],
        )
        for anchor in anchor_set
        if final_coverage[anchor] < required_sources[anchor]
    }

    if unmet_requirements:
        details = ", ".join(
            f"{anchor}: {current}/{required}"
            for anchor, (current, required)
            in sorted(unmet_requirements.items())
        )

        warnings.warn(
            "Layer 1 size constraint prevented full anchor coverage. "
            f"Unmet requirements: {details}"
        )

    ranking["selected"] = ranking["gene"].isin(selected)

    return selected, ranking

def select_upstream_layer(
    graph: nx.DiGraph,
    layer_by_node: dict[str, int],
    current_layer: int,
    selected_previous_layer: set[str],
    already_selected: set[str],
    number_to_keep: int,
    weight_previous_targets: float = 2.0,
    weight_same_layer_targets: float = 1.0,
) -> tuple[set[str], pd.DataFrame]:
    """
    Rank and select candidates from Layer 2 or above.
    """
    if current_layer < 2:
        raise ValueError("current_layer must be at least 2.")

    if number_to_keep < 0:
        raise ValueError("number_to_keep must be non-negative.")

    current_candidates = {
        gene
        for gene, layer in layer_by_node.items()
        if layer == current_layer
    }

    # Safeguard against reconsidering previously selected nodes.
    current_candidates -= set(already_selected)

    previous_layer = set(selected_previous_layer)

    ranking_rows = []

    for gene in current_candidates:
        targets = set(graph.successors(gene))

        previous_layer_targets = (
            targets & previous_layer
        )

        same_layer_targets = (
            targets & current_candidates
        ) - {gene}

        score = (
            weight_previous_targets
            * len(previous_layer_targets)
            + weight_same_layer_targets
            * len(same_layer_targets)
        )

        ranking_rows.append(
            {
                "gene": gene,
                "score": score,
                "previous_layer_targets": len(
                    previous_layer_targets
                ),
                "same_layer_targets": len(
                    same_layer_targets
                ),
            }
        )

    ranking = pd.DataFrame(ranking_rows)

    if ranking.empty:
        ranking["selected"] = pd.Series(dtype=bool)
        return set(), ranking

    ranking = ranking.sort_values(
        [
            "score",
            "previous_layer_targets",
            "same_layer_targets",
            "gene",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    selected = set(
        ranking.head(
            min(number_to_keep, len(ranking))
        )["gene"]
    )

    ranking["selected"] = ranking["gene"].isin(selected)

    return selected, ranking

def propagate_structural_layers(
    graph: nx.DiGraph,
    anchors: set[str],
) -> nx.DiGraph:
    """
    Raise candidate layers until every retained edge satisfies:

        source layer >= target layer

    Anchors remain fixed at Layer 0.
    """
    graph = graph.copy()

    changed = True

    while changed:
        changed = False

        for source, target in graph.edges:
            # Anchors remain fixed at Layer 0.
            if source in anchors:
                continue

            source_layer = graph.nodes[source]["layer"]
            target_layer = graph.nodes[target]["layer"]

            if source_layer < target_layer:
                graph.nodes[source]["layer"] = target_layer
                changed = True

    return graph

def build_selected_graphs(
    interactions: pd.DataFrame,
    selected_by_layer: dict[int, set[str]],
) -> tuple[pd.DataFrame, nx.DiGraph, nx.DiGraph, pd.DataFrame]:
    """
    Build full and structurally layered graphs from selected genes.

    The original selection layer is stored as `selection_layer`.
    The adjusted structural layer is stored as `layer`.

    Self-loops and anchor-to-non-anchor edges are removed.
    Candidate-to-candidate edges are preserved by raising source layers.
    """
    selection_layer_by_node = {
        gene: layer
        for layer, genes in selected_by_layer.items()
        for gene in genes
    }

    selected_genes = set(selection_layer_by_node)
    anchors = set(selected_by_layer.get(0, set()))

    selected_interactions = interactions[
        interactions["regulator"].isin(selected_genes)
        & interactions["target"].isin(selected_genes)
    ].copy()

    selected_full_graph = nx.from_pandas_edgelist(
        selected_interactions,
        source="regulator",
        target="target",
        edge_attr=True,
        create_using=nx.DiGraph,
    )

    # Keep selected genes even when they have no internal edges.
    selected_full_graph.add_nodes_from(selected_genes)

    nx.set_node_attributes(
        selected_full_graph,
        selection_layer_by_node,
        name="selection_layer",
    )

    # Structural layers initially equal the selection layers.
    nx.set_node_attributes(
        selected_full_graph,
        selection_layer_by_node,
        name="layer",
    )

    nx.set_node_attributes(
        selected_full_graph,
        {
            gene: gene in anchors
            for gene in selected_genes
        },
        name="is_anchor",
    )

    structural_graph = selected_full_graph.copy()
    excluded_rows = []

    for source, target in list(structural_graph.edges):
        source_layer = selection_layer_by_node[source]
        target_layer = selection_layer_by_node[target]

        if source == target:
            reason = "self_loop"

        elif source in anchors and target not in anchors:
            reason = "anchor_to_nonanchor"

        else:
            continue

        excluded_rows.append(
            {
                "regulator": source,
                "target": target,
                "regulator_selection_layer": source_layer,
                "target_selection_layer": target_layer,
                "reason": reason,
            }
        )

        structural_graph.remove_edge(source, target)

    structural_graph = propagate_structural_layers(
        structural_graph,
        anchors,
    )

    excluded_edges = pd.DataFrame(excluded_rows)

    return (
        selected_interactions,
        selected_full_graph,
        structural_graph,
        excluded_edges,
    )

def count_unique_citations(references) -> int:
    """
    Count unique publication identifiers in an OmniPath references field.
    """
    if references is None or pd.isna(references):
        return 0

    # OmniPath references normally contain numeric PubMed identifiers.
    citation_ids = re.findall(r"\d+", str(references))

    return len(set(citation_ids))


def add_edge_support_attributes(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Add citation count and numeric curation effort to every edge.
    """
    graph = graph.copy()

    for source, target, data in graph.edges(data=True):
        data["n_unique_citations"] = count_unique_citations(
            data.get("references")
        )

        try:
            data["curation_effort"] = float(
                data.get("curation_effort", 0)
            )
        except (TypeError, ValueError):
            data["curation_effort"] = 0.0

    return graph


def add_regulatory_sign_attributes(graph: nx.DiGraph) -> nx.DiGraph:
    """Normalize and explicitly encode the known sign of every edge.

    The readable ``sign`` attribute is retained. ``regulatory_sign`` is added
    for downstream model checks: activation is ``1``, repression is ``-1``,
    and mixed or unknown evidence is ``0`` (not evaluable as one sign).
    """
    graph = graph.copy()

    for source, target, data in graph.edges(data=True):
        raw_sign = data.get("sign", "unknown")
        sign = str(raw_sign).strip().lower()
        if sign not in REGULATORY_SIGN_VALUES:
            warnings.warn(
                f"Edge {source}->{target} has unsupported sign {raw_sign!r}; "
                "it will be stored as unknown."
            )
            sign = "unknown"

        data["sign"] = sign
        data["regulatory_sign"] = REGULATORY_SIGN_VALUES[sign]

    return graph

def canonicalize_cycle(cycle: list[str]) -> tuple[str, ...]:
    """
    Rotate a cycle so that it always has the same representation.

    Direction is preserved. Only the starting node changes.
    """
    rotations = [
        tuple(cycle[i:] + cycle[:i])
        for i in range(len(cycle))
    ]

    return min(rotations)


def get_graph_cycles(
    graph: nx.DiGraph,
) -> list[tuple[str, ...]]:
    """
    Return all directed cycles, shortest first.
    """
    cycles = {
        canonicalize_cycle(cycle)
        for cycle in nx.simple_cycles(graph)
        if len(cycle) >= 2
    }

    return sorted(
        cycles,
        key=lambda cycle: (len(cycle), cycle),
    )


def get_cycle_edges(
    cycle: tuple[str, ...],
) -> list[tuple[str, str]]:
    """
    Convert a cycle node sequence into its directed edges.
    """
    return [
        (cycle[i], cycle[(i + 1) % len(cycle)])
        for i in range(len(cycle))
    ]

def find_weakest_cycle_edge(
    graph: nx.DiGraph,
    cycle: tuple[str, ...],
) -> tuple[tuple[str, str] | None, list[dict]]:
    """
    Find a uniquely weakest edge in a cycle.

    Decision order:
    1. Fewest unique citations.
    2. Lowest curation effort.
    3. Weakest downstream direction based on original selection layers.

    Returns None if the weakest edges remain tied.
    """
    edge_information = []

    for source, target in get_cycle_edges(cycle):
        edge_data = graph.edges[source, target]

        source_selection_layer = graph.nodes[source][
            "selection_layer"
        ]
        target_selection_layer = graph.nodes[target][
            "selection_layer"
        ]

        edge_information.append(
            {
                "source": source,
                "target": target,
                "n_unique_citations": edge_data.get(
                    "n_unique_citations",
                    0,
                ),
                "curation_effort": edge_data.get(
                    "curation_effort",
                    0,
                ),
                "downstream_step": (
                    source_selection_layer
                    - target_selection_layer
                ),
            }
        )

    # First criterion: citation support.
    minimum_citations = min(
        edge["n_unique_citations"]
        for edge in edge_information
    )

    weakest = [
        edge
        for edge in edge_information
        if edge["n_unique_citations"] == minimum_citations
    ]

    if len(weakest) == 1:
        edge = weakest[0]
        return (edge["source"], edge["target"]), weakest

    # Second criterion: curation effort.
    minimum_curation = min(
        edge["curation_effort"]
        for edge in weakest
    )

    weakest = [
        edge
        for edge in weakest
        if edge["curation_effort"] == minimum_curation
    ]

    if len(weakest) == 1:
        edge = weakest[0]
        return (edge["source"], edge["target"]), weakest

    # Third criterion: preserve stronger downstream edges.
    minimum_downstream_step = min(
        edge["downstream_step"]
        for edge in weakest
    )

    weakest = [
        edge
        for edge in weakest
        if edge["downstream_step"] == minimum_downstream_step
    ]

    if len(weakest) == 1:
        edge = weakest[0]
        return (edge["source"], edge["target"]), weakest

    return None, weakest

def break_graph_cycles(
    graph: nx.DiGraph,
) -> tuple[nx.DiGraph, pd.DataFrame, list[tuple[str, ...]]]:
    """
    Break directed cycles throughout the complete graph.

    Cycles are processed shortest first. The weakest edge is chosen by:
    1. fewest unique citations;
    2. lowest curation effort.
    """
    working_graph = add_edge_support_attributes(graph)

    removed_rows = []
    warned_cycles = set()
    unresolved_cycles = []

    while True:
        cycles = get_graph_cycles(working_graph)

        if not cycles:
            unresolved_cycles = []
            break

        edge_removed = False
        unresolved_this_pass = []

        for cycle in cycles:
            edge_to_remove, tied_edges = find_weakest_cycle_edge(
                working_graph,
                cycle,
            )

            if edge_to_remove is None:
                unresolved_this_pass.append(cycle)

                if cycle not in warned_cycles:
                    cycle_text = " -> ".join(
                        list(cycle) + [cycle[0]]
                    )

                    warnings.warn(
                        f"Unresolved cycle: {cycle_text}. "
                        "Weakest edges have equal citation counts "
                        "and curation effort."
                    )

                    warned_cycles.add(cycle)

                continue

            source, target = edge_to_remove
            edge_data = working_graph.edges[source, target]

            removed_rows.append(
                {
                    "regulator": source,
                    "target": target,
                    "cycle": " -> ".join(
                        list(cycle) + [cycle[0]]
                    ),
                    "cycle_length": len(cycle),
                    "layer": working_graph.nodes[source]["layer"],
                    "n_unique_citations": edge_data[
                        "n_unique_citations"
                    ],
                    "curation_effort": edge_data[
                        "curation_effort"
                    ],
                }
            )

            working_graph.remove_edge(source, target)
            edge_removed = True

            # Recalculate every cycle after one removal.
            break

        if not edge_removed:
            unresolved_cycles = unresolved_this_pass
            break

    removed_edges = pd.DataFrame(removed_rows)

    return working_graph, removed_edges, unresolved_cycles

def reorganize_plot_layers(
    graph: nx.DiGraph,
    minimum_peeled_candidates: int = 3,
) -> nx.DiGraph:
    """
    Create compact concentric plotting layers.

    Steps
    -----
    1. Remove gaps between positive layer numbers.
    2. Keep the current maximum structural layer N fixed.
    3. Repeatedly peel nodes from Layer N whose sources are all
       already located above Layer N.
    4. Put each newly peeled group at N + 1 and shift previously
       peeled groups one layer outward.
    5. Stop when fewer than `minimum_peeled_candidates` are eligible.

    The original `layer` attribute is unchanged. Results are stored
    in the `plot_layer` node attribute.
    """
    graph = graph.copy()

    plot_layers = {
        node: data["layer"]
        for node, data in graph.nodes(data=True)
    }

    # Remove gaps between positive layer numbers.
    used_layers = sorted({
        layer
        for layer in plot_layers.values()
        if layer > 0
    })

    compact_mapping = {
        old_layer: new_layer
        for new_layer, old_layer in enumerate(
            used_layers,
            start=1,
        )
    }

    for node, layer in plot_layers.items():
        if layer > 0:
            plot_layers[node] = compact_mapping[layer]

    maximum_base_layer = max(
        plot_layers.values(),
        default=0,
    )

    if maximum_base_layer == 0:
        nx.set_node_attributes(
            graph,
            plot_layers,
            name="plot_layer",
        )
        return graph

    while True:
        peeled_candidates = []

        for node, layer in plot_layers.items():
            if layer != maximum_base_layer:
                continue

            sources = graph.predecessors(node)

            # True for nodes with no sources, or whose sources
            # have all already been peeled outward.
            if all(
                plot_layers[source] > maximum_base_layer
                for source in sources
            ):
                peeled_candidates.append(node)

        if len(peeled_candidates) < minimum_peeled_candidates:
            break

        # Move previously peeled shells outward.
        for node in plot_layers:
            if plot_layers[node] > maximum_base_layer:
                plot_layers[node] += 1

        # Place the newest peeled group next to the base layer.
        for node in peeled_candidates:
            plot_layers[node] = maximum_base_layer + 1

    nx.set_node_attributes(
        graph,
        plot_layers,
        name="plot_layer",
    )

    return graph

def draw_gene_graph(
    graph: nx.DiGraph,
    layer_attribute: str = "plot_layer",
    figsize: tuple[int, int] = (12, 12),
    node_size: int = 1300,
    anchor_node_size: int = 1800,
    font_size: int = 8,
):
    """
    Draw a layered gene network as concentric circles.

    Anchors occupy the center shell.
    Layer 1 forms the first surrounding circle, Layer 2 the next, etc.
    """
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot draw an empty graph.")

    # Fall back to the structural layer if plot_layer is unavailable.
    if not all(
        layer_attribute in graph.nodes[node]
        for node in graph.nodes
    ):
        layer_attribute = "layer"

    layers = sorted({
        graph.nodes[node][layer_attribute]
        for node in graph.nodes
    })

    # Alphabetical ordering makes the plot reproducible.
    shells = [
        sorted(
            node
            for node in graph.nodes
            if graph.nodes[node][layer_attribute] == layer
        )
        for layer in layers
    ]

    pos = nx.shell_layout(
        graph,
        nlist=shells,
        rotate=0,
        scale=1,
    )

    anchors = {
        node
        for node in graph.nodes
        if graph.nodes[node].get("is_anchor", False)
    }

    non_anchors = set(graph.nodes) - anchors

    # Keep multiple anchors concentrated near the center.
    if len(anchors) > 1:
        for node in anchors:
            pos[node] = pos[node] * 0.35

    fig, ax = plt.subplots(figsize=figsize)

    # Optional circular guides for the non-anchor layers.
    for layer_index in range(1, len(shells)):
        shell_nodes = shells[layer_index]

        if not shell_nodes:
            continue

        radius = (
            pos[shell_nodes[0]][0] ** 2
            + pos[shell_nodes[0]][1] ** 2
        ) ** 0.5

        circle = plt.Circle(
            (0, 0),
            radius,
            fill=False,
            linewidth=0.7,
            alpha=0.2,
        )
        ax.add_patch(circle)

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        arrows=True,
        arrowsize=14,
        arrowstyle="-|>",
        width=1.0,
        alpha=0.55,
        node_size=node_size,
        connectionstyle="arc3,rad=0.05",
    )

    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=sorted(non_anchors),
        node_size=node_size,
        node_color="lightblue",
        edgecolors="black",
        linewidths=0.7,
        ax=ax,
    )

    nx.draw_networkx_nodes(
        graph,
        pos,
        nodelist=sorted(anchors),
        node_size=anchor_node_size,
        node_shape="s",
        node_color="gold",
        edgecolors="black",
        linewidths=1.2,
        ax=ax,
    )

    nx.draw_networkx_labels(
        graph,
        pos,
        font_size=font_size,
        font_weight="bold",
        ax=ax,
    )

    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()

    return fig, ax

def save_gene_graph(
    graph: nx.DiGraph,
    file_path: str | Path,
) -> None:
    """
    Save a NetworkX gene-regulatory graph with all attributes.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(file_path, "wb") as file:
        pickle.dump(
            graph,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        
def main() -> int:
    """Select or load genes, reconstruct layers, and save a validated DAG."""
    args = parse_args()
    config = load_effective_config(args.config)
    validate_config(config)

    generation = config["geneset_generation"]
    data_options = config["omnipath"]
    selection = config["selection"]
    plotting = config["plotting"]
    output = config["output"]

    anchors = generation["anchors"]
    layers_genes = generation.get("layers_genes")

    collectri_edges = load_collectri_from_omnipath(
        config=config,
        timeout=data_options["timeout_seconds"],
        require_directed=data_options["require_directed"],
        require_signed=data_options["require_signed"],
        include_self_loops=data_options["include_self_loops"],
        license_type=data_options["license_type"],
        verbose=data_options["verbose"],
    )

    anchors, anchor_summary = validate_anchors(
        interactions=collectri_edges,
        anchors=anchors,
    )

    if layers_genes is not None:
        layers_genes = validate_gene_symbols(
            interactions=collectri_edges,
            genes=layers_genes,
            list_name="layers_genes",
        )

    print("Validated anchors:")
    print(anchors)

    print("\nAnchor regulatory coverage:")
    print(anchor_summary.to_string(index=False))

    using_explicit_genes = layers_genes is not None

    if using_explicit_genes:
        selected_by_layer = {0: set(anchors)}
        selected_by_layer[1] = set(layers_genes)
        print(
            "Using explicitly configured genes; all start in Layer 1 and "
            "candidate formation and ranking are skipped."
        )
    else:
        candidate_pool = build_upstream_candidate_pool(
            interactions=collectri_edges,
            anchors=anchors,
            number_of_layers=generation["num_layers"],
        )
        _, candidate_graph = build_candidate_graph(
            interactions=collectri_edges,
            anchors=anchors,
            candidate_pool=candidate_pool,
        )
        layer_by_node = assign_minimal_layers(candidate_graph, anchors)
        number_by_layer = generation["number_by_layer"]
        selected_by_layer = {0: set(anchors)}
        selected_by_layer[1], _ = select_layer_1(
            graph=candidate_graph,
            layer_by_node=layer_by_node,
            anchors=anchors,
            number_to_keep=number_by_layer[0],
            min_sources=selection["layer_1_min_sources"],
            weight_anchor_targets=selection["weight_anchor_targets"],
            weight_same_layer_targets=selection["weight_same_layer_targets"],
            weight_upstream_sources=selection["weight_upstream_sources"],
        )
        already_selected = selected_by_layer[0] | selected_by_layer[1]
        for current_layer in range(2, len(number_by_layer) + 1):
            selected_by_layer[current_layer], _ = select_upstream_layer(
                graph=candidate_graph,
                layer_by_node=layer_by_node,
                current_layer=current_layer,
                selected_previous_layer=selected_by_layer[current_layer - 1],
                already_selected=already_selected,
                number_to_keep=number_by_layer[current_layer - 1],
                weight_previous_targets=selection["weight_previous_targets"],
                weight_same_layer_targets=selection["weight_same_layer_targets"],
            )
            already_selected.update(selected_by_layer[current_layer])

    selected_genes = set().union(
        *selected_by_layer.values()
    )

    print(f"Total selected genes: {len(selected_genes)}")

    for layer, genes in selected_by_layer.items():
        print(
            f"Layer {layer}: {len(genes)} genes: "
            f"Genes selected: {', '.join(sorted(genes))}"
        )

    (
        selected_interactions,
        selected_full_graph,
        layered_graph,
        excluded_edges,
    ) = build_selected_graphs(
        interactions=collectri_edges,
        selected_by_layer=selected_by_layer,
    )

    if using_explicit_genes:
        reconstructed_layers = reconstruct_selected_gene_layers(
            layered_graph,
            anchors,
        )
        print("\nReconstructed structural layers:")
        for layer in sorted(set(reconstructed_layers.values())):
            genes = sorted(
                gene
                for gene, assigned_layer in reconstructed_layers.items()
                if assigned_layer == layer
            )
            print(f"Layer {layer}: {len(genes)} genes: {', '.join(genes)}")

    print("Selected full graph")
    print(f"Nodes: {selected_full_graph.number_of_nodes()}")
    print(f"Edges: {selected_full_graph.number_of_edges()}")

    print("\nLayer-compatible graph")
    print(f"Nodes: {layered_graph.number_of_nodes()}")
    print(f"Edges: {layered_graph.number_of_edges()}")

    print(f"\nExcluded edges: {len(excluded_edges)}")

    dag_graph, removed_cycle_edges, unresolved_cycles = (
        break_graph_cycles(layered_graph)
    )

    print("Before cycle removal:")
    print(f"Nodes: {layered_graph.number_of_nodes()}")
    print(f"Edges: {layered_graph.number_of_edges()}")

    print("\nAfter cycle removal:")
    print(f"Nodes: {dag_graph.number_of_nodes()}")
    print(f"Edges: {dag_graph.number_of_edges()}")

    print("\nRemoved cycle edges:")
    print(removed_cycle_edges.to_string(index=False))

    print("\nUnresolved cycles:")
    print(unresolved_cycles)

    if unresolved_cycles or not nx.is_directed_acyclic_graph(dag_graph):
        warnings.warn(
            "The reconstructed graph still contains unresolved cycles. "
            "Execution stopped; no graph or figure was saved."
        )
        return 1

    # Preserve a normalized readable sign and an aligned numeric sign on the
    # final graph consumed by the quantum-circuit and trainer modules.
    dag_graph = add_regulatory_sign_attributes(dag_graph)

    graph_path = resolve_project_path(output["graph_file"])
    figure_path = resolve_project_path(output["figure_file"])
    save_gene_graph(dag_graph, graph_path)

    plot_graph = reorganize_plot_layers(
        dag_graph,
        plotting["minimum_peeled_candidates"],
    )
    fig, _ = draw_gene_graph(
        plot_graph,
        figsize=tuple(plotting["figsize"]),
        node_size=plotting["node_size"],
        anchor_node_size=plotting["anchor_node_size"],
        font_size=plotting["font_size"],
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=plotting["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"Saved DAG: {graph_path}")
    print(f"Saved figure: {figure_path}")
    return 0
    
if __name__ == "__main__":
    sys.exit(main())
