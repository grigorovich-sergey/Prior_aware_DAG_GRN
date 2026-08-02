"""Train prior-aware gene-regulatory quantum circuits.

This module is being built incrementally.  Its current responsibility is to
provide the observed-distribution contract used by future loss functions and
to project finite-shot circuit output onto that observed support.

The real observation-table schema has not yet been selected.  For now,
``read_observed_frequencies`` is an explicit synthetic-data stub.  Running
this file directly demonstrates the temporary input and projection behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Integral, Real
from pathlib import Path
import random


__all__ = [
    "ProjectedSampleDistribution",
    "project_sampled_counts_to_observed_support",
    "read_observed_frequencies",
]


@dataclass(frozen=True)
class ProjectedSampleDistribution:
    """Sampled output conditioned on the observed bitstring support.

    ``matching_probabilities`` sums to one and is suitable for comparison
    with the observed frequencies.  Shot diagnostics are retained because
    conditioning removes sampled probability mass outside the dataset's
    observed support.
    """

    matching_counts: dict[str, int]
    matching_probabilities: dict[str, float]
    total_shots: int
    matching_shots: int
    excluded_shots: int
    matching_fraction: float


def _validate_bitstring_length(bitstring_length: int) -> int:
    """Validate the number of genes represented by each bitstring."""
    if (
        not isinstance(bitstring_length, Integral)
        or isinstance(bitstring_length, bool)
    ):
        raise TypeError("bitstring_length must be an integer.")
    if bitstring_length <= 0:
        raise ValueError("bitstring_length must be greater than zero.")
    return int(bitstring_length)


def _validate_seed(seed: int | None) -> int | None:
    """Validate an optional nonnegative random seed."""
    if seed is None:
        return None
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise TypeError("seed must be an integer or None.")
    if seed < 0:
        raise ValueError("seed must be nonnegative.")
    return int(seed)


def _validate_observed_frequencies(
    observed_frequencies: Mapping[str, float],
) -> tuple[dict[str, float], int]:
    """Validate a normalized observed distribution and infer its width."""
    if not isinstance(observed_frequencies, Mapping):
        raise TypeError("observed_frequencies must be a mapping.")
    if not observed_frequencies:
        raise ValueError("observed_frequencies must not be empty.")

    normalized: dict[str, float] = {}
    bitstring_length: int | None = None
    for bitstring, frequency in observed_frequencies.items():
        if not isinstance(bitstring, str):
            raise TypeError("Every observed-frequency key must be a string.")
        if not bitstring or set(bitstring) - {"0", "1"}:
            raise ValueError(
                "Every observed-frequency key must be a nonempty bitstring."
            )
        if bitstring_length is None:
            bitstring_length = len(bitstring)
        elif len(bitstring) != bitstring_length:
            raise ValueError(
                "All observed bitstrings must have the same length."
            )

        if not isinstance(frequency, Real) or isinstance(frequency, bool):
            raise TypeError("Every observed frequency must be a real number.")
        numeric_frequency = float(frequency)
        if not math.isfinite(numeric_frequency):
            raise ValueError("Every observed frequency must be finite.")
        if numeric_frequency <= 0.0:
            raise ValueError(
                "Every bitstring in the observed support must have a positive "
                "frequency. Remove zero-frequency entries from the mapping."
            )
        normalized[bitstring] = numeric_frequency

    total = math.fsum(normalized.values())
    if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            "Observed frequencies must sum to 1; "
            f"received {total:.17g}."
        )

    # Remove negligible floating-point drift while preserving the input's
    # relative frequencies and insertion order.
    normalized = {
        bitstring: frequency / total
        for bitstring, frequency in normalized.items()
    }
    assert bitstring_length is not None
    return normalized, bitstring_length


def read_observed_frequencies(
    observations_file: str | Path | None = None,
    *,
    bitstring_length: int,
    number_of_observed_bitstrings: int | None = None,
    seed: int | None = None,
) -> dict[str, float]:
    """Return a temporary synthetic observed-frequency distribution.

    This function defines the return contract for the future observation-table
    reader: a mapping from gene-ordered bitstrings to positive frequencies that
    sum to one.  The synthetic support is sampled without replacement and is
    always a strict subset of the ``2 ** bitstring_length`` possible outcomes.

    ``observations_file`` is reserved for the real reader.  Passing a path now
    raises ``NotImplementedError`` instead of silently ignoring user data.
    """
    if observations_file is not None:
        if not isinstance(observations_file, (str, Path)):
            raise TypeError("observations_file must be a path or None.")
        raise NotImplementedError(
            "Reading an observation file is not implemented because its table "
            "schema has not yet been defined. Pass observations_file=None to "
            "generate the temporary synthetic distribution."
        )

    width = _validate_bitstring_length(bitstring_length)
    normalized_seed = _validate_seed(seed)
    total_outcomes = 1 << width

    if number_of_observed_bitstrings is None:
        support_size = min(8, max(1, total_outcomes // 2))
    else:
        if (
            not isinstance(number_of_observed_bitstrings, Integral)
            or isinstance(number_of_observed_bitstrings, bool)
        ):
            raise TypeError(
                "number_of_observed_bitstrings must be an integer or None."
            )
        support_size = int(number_of_observed_bitstrings)

    if support_size <= 0:
        raise ValueError(
            "number_of_observed_bitstrings must be greater than zero."
        )
    if support_size >= total_outcomes:
        raise ValueError(
            "number_of_observed_bitstrings must be smaller than the total "
            "number of possible bitstrings so the synthetic support remains "
            "incomplete."
        )

    rng = random.Random(normalized_seed)
    selected_outcomes: set[int] = set()
    while len(selected_outcomes) < support_size:
        selected_outcomes.add(rng.getrandbits(width))

    bitstrings = [
        format(outcome, f"0{width}b")
        for outcome in sorted(selected_outcomes)
    ]
    weights = [rng.uniform(0.1, 1.0) for _ in bitstrings]
    weight_total = math.fsum(weights)
    frequencies = {
        bitstring: weight / weight_total
        for bitstring, weight in zip(bitstrings, weights)
    }

    # Assign the final residual explicitly so ordinary sum() also returns one
    # within the tightest practical floating-point precision.
    last_bitstring = bitstrings[-1]
    preceding_total = math.fsum(
        frequency
        for bitstring, frequency in frequencies.items()
        if bitstring != last_bitstring
    )
    frequencies[last_bitstring] = 1.0 - preceding_total

    validated, _ = _validate_observed_frequencies(frequencies)
    return validated


def project_sampled_counts_to_observed_support(
    sampled_counts: Mapping[str, int],
    observed_frequencies: Mapping[str, float],
) -> ProjectedSampleDistribution:
    """Condition sampled counts on bitstrings present in observed data.

    Sampled strings absent from ``observed_frequencies`` are excluded.  Every
    observed string remains in the returned mappings, with a zero count and
    zero probability when it was not sampled.  Matching counts are then
    renormalized to sum to one.

    At least one sampled shot must match the observed support.  With no matches,
    the requested conditional distribution is undefined and a ``ValueError``
    is raised for the future objective function to handle explicitly.
    """
    observed, bitstring_length = _validate_observed_frequencies(
        observed_frequencies
    )
    if not isinstance(sampled_counts, Mapping):
        raise TypeError("sampled_counts must be a mapping.")
    if not sampled_counts:
        raise ValueError("sampled_counts must not be empty.")

    validated_counts: dict[str, int] = {}
    for bitstring, count in sampled_counts.items():
        if not isinstance(bitstring, str):
            raise TypeError("Every sampled-count key must be a string.")
        if len(bitstring) != bitstring_length or set(bitstring) - {"0", "1"}:
            raise ValueError(
                "Every sampled-count key must be a bitstring with the same "
                "length as the observed bitstrings."
            )
        if not isinstance(count, Integral) or isinstance(count, bool):
            raise TypeError("Every sampled count must be an integer.")
        if count < 0:
            raise ValueError("Every sampled count must be nonnegative.")
        validated_counts[bitstring] = int(count)

    total_shots = sum(validated_counts.values())
    if total_shots <= 0:
        raise ValueError("sampled_counts must contain at least one shot.")

    matching_counts = {
        bitstring: validated_counts.get(bitstring, 0)
        for bitstring in observed
    }
    matching_shots = sum(matching_counts.values())
    if matching_shots == 0:
        raise ValueError(
            "No sampled outcomes match the observed bitstring support; the "
            "conditional sampled frequencies cannot be evaluated."
        )

    matching_probabilities = {
        bitstring: count / matching_shots
        for bitstring, count in matching_counts.items()
    }
    excluded_shots = total_shots - matching_shots
    return ProjectedSampleDistribution(
        matching_counts=matching_counts,
        matching_probabilities=matching_probabilities,
        total_shots=total_shots,
        matching_shots=matching_shots,
        excluded_shots=excluded_shots,
        matching_fraction=matching_shots / total_shots,
    )


def main() -> None:
    """Demonstrate the temporary observed-data and support projection flow."""
    observed = read_observed_frequencies(
        bitstring_length=4,
        number_of_observed_bitstrings=5,
        seed=42,
    )
    unsupported_bitstring = next(
        format(value, "04b")
        for value in range(16)
        if format(value, "04b") not in observed
    )
    sampled_counts = {
        **{
            bitstring: 10 * (index + 1)
            for index, bitstring in enumerate(observed)
        },
        unsupported_bitstring: 50,
    }
    projected = project_sampled_counts_to_observed_support(
        sampled_counts,
        observed,
    )

    print("Synthetic observed frequencies:", observed)
    print("Projected sampled frequencies:", projected.matching_probabilities)
    print(
        "Matching shots:",
        f"{projected.matching_shots}/{projected.total_shots}",
    )


if __name__ == "__main__":
    main()
