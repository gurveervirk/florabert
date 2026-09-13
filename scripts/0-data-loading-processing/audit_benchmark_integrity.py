#!/usr/bin/env python3
"""Audit sequence and split integrity for the FloraBERT benchmark.

This script intentionally does not import torch or any model package. It downloads
the public Hugging Face datasets, writes hash-only manifests, checks exact and
reverse-complement overlap, and optionally runs the expensive MMseqs2 cluster
audit.

The exact project split is only considered cluster-audited when MMseqs2 is run
over the combined expression and MLM corpus, or when a manifest containing all
relevant records and cluster assignments is supplied. A downstream-only cluster
manifest can validate split separation but cannot establish cross-dataset
near-duplicate non-overlap.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Iterable


DEFAULT_EXPRESSION_REPO = "Gurveer05/maize-nam-gene-expression-data"
DEFAULT_MLM_REPO = "Gurveer05/maize-promoter-sequences"
MLM_FILENAMES = ("all_seqs_train.txt", "all_seqs_test.txt")
SEQUENCE_COLUMN_CANDIDATES = ("sequence", "seq", "text", "promoter", "input")
COMPLEMENT_TABLE = str.maketrans(
    "ACGTNRYKMSWBDHV",
    "TGCANYRMKSWVHDB",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit exact, reverse-complement, and optional cluster overlap."
    )
    parser.add_argument(
        "--audit-root",
        type=Path,
        default=Path(
            os.environ.get(
                "FLORABERT_AUDIT_ROOT",
                "/content/florabert_benchmark_audit",
            )
        ),
    )
    parser.add_argument(
        "--expression-repo",
        default=os.environ.get(
            "FLORABERT_EXPRESSION_DATASET",
            DEFAULT_EXPRESSION_REPO,
        ),
    )
    parser.add_argument(
        "--mlm-repo",
        default=os.environ.get(
            "FLORABERT_MLM_DATASET",
            DEFAULT_MLM_REPO,
        ),
    )
    parser.add_argument(
        "--sequence-column",
        default=os.environ.get(
            "FLORABERT_EXPRESSION_SEQUENCE_COLUMN",
            "",
        ).strip() or None,
    )
    parser.add_argument(
        "--cluster-manifest",
        type=Path,
        default=(
            Path(os.environ["FLORABERT_CLUSTER_MANIFEST"]).expanduser()
            if os.environ.get("FLORABERT_CLUSTER_MANIFEST", "").strip()
            else None
        ),
        help=(
            "Optional manifest with sequence_sha256, cluster_id, and split. "
            "A downstream-only manifest validates split separation but not "
            "cross-dataset near-duplicate overlap."
        ),
    )
    parser.add_argument(
        "--run-mmseqs",
        action="store_true",
        default=os.environ.get("FLORABERT_RUN_MMSEQS", "").lower()
        in {"1", "true", "yes", "y", "on"},
        help="Run MMseqs2 over the combined expression and MLM corpus.",
    )
    parser.add_argument(
        "--mmseqs-bin",
        default=os.environ.get("FLORABERT_MMSEQS_BIN", "").strip()
        or shutil.which("mmseqs")
        or "",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=500,
        help="Maximum exact-overlap examples retained per MLM file.",
    )
    return parser.parse_args()


def optional_token_kwargs() -> dict:
    token = (
        os.environ.get("HF_TOKEN", "").strip()
        or os.environ.get("HUGGINGFACE_HUB_TOKEN", "").strip()
        or None
    )
    return {"token": token} if token else {}


def normalize_sequence(value: object) -> str:
    if value is None:
        return ""
    return "".join(str(value).upper().split())


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT_TABLE)[::-1]


def sequence_hashes(sequence: str) -> tuple[str, str]:
    normalized = normalize_sequence(sequence)
    if not normalized:
        return "", ""
    forward = hashlib.sha256(normalized.encode("ascii", errors="replace")).hexdigest()
    reverse = hashlib.sha256(
        reverse_complement(normalized).encode("ascii", errors="replace")
    ).hexdigest()
    return forward, reverse


def iter_text_sequences(path: Path) -> Iterable[tuple[int, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            sequence = normalize_sequence(line)
            if sequence:
                yield row_index, sequence


def choose_sequence_column(columns: list[str], override: str | None) -> str:
    if override:
        if override not in columns:
            raise KeyError(
                f"Configured sequence column {override!r} is absent from {columns}."
            )
        return override

    matches = [name for name in SEQUENCE_COLUMN_CANDIDATES if name in columns]
    if len(matches) != 1:
        raise ValueError(
            "Could not identify one expression sequence column. "
            f"Columns={columns}; candidates={matches}. "
            "Pass --sequence-column."
        )
    return matches[0]


def write_expression_manifest(
    expression_splits: dict,
    sequence_column_override: str | None,
    audit_root: Path,
) -> tuple[Path, dict[str, set[str]], dict[str, set[str]], list[dict]]:
    manifest_path = audit_root / "expression_sequence_manifest.tsv"
    hashes_by_split: dict[str, set[str]] = {}
    hash_to_splits: dict[str, set[str]] = defaultdict(set)
    count_rows: list[dict] = []

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            [
                "source_split",
                "source_row",
                "sequence_length",
                "sequence_sha256",
                "reverse_complement_sha256",
            ]
        )

        for split_name, dataset in expression_splits.items():
            source_name = f"hf::{split_name}"
            sequence_column = choose_sequence_column(
                list(dataset.column_names),
                sequence_column_override,
            )
            hashes: set[str] = set()
            total_rows = 0
            empty_rows = 0

            for row_index, row in enumerate(dataset):
                total_rows += 1
                sequence = normalize_sequence(row.get(sequence_column))
                forward_hash, reverse_hash = sequence_hashes(sequence)
                if not forward_hash:
                    empty_rows += 1
                    continue

                hashes.add(forward_hash)
                hash_to_splits[forward_hash].add(source_name)
                writer.writerow(
                    [
                        source_name,
                        row_index,
                        len(sequence),
                        forward_hash,
                        reverse_hash,
                    ]
                )

            hashes_by_split[source_name] = hashes
            count_rows.append(
                {
                    "source_split": source_name,
                    "rows": total_rows,
                    "nonempty_rows": total_rows - empty_rows,
                    "empty_rows": empty_rows,
                    "unique_sequences": len(hashes),
                    "sequence_column": sequence_column,
                }
            )

    return manifest_path, hashes_by_split, hash_to_splits, count_rows


def audit_exact_overlaps(
    mlm_paths: dict[str, Path],
    expression_hashes_by_split: dict[str, set[str]],
    audit_root: Path,
    max_examples: int,
) -> tuple[list[dict], list[dict]]:
    summary_rows: list[dict] = []
    split_rows: list[dict] = []
    example_rows: list[dict] = []

    for filename, path in mlm_paths.items():
        total_rows = 0
        unique_hashes: set[str] = set()
        forward_matches: set[str] = set()
        reverse_matches: set[str] = set()
        either_matches: set[str] = set()
        forward_rows = 0
        reverse_rows = 0
        either_rows = 0
        duplicate_rows = 0
        split_match_counts: Counter[tuple[str, str]] = Counter()

        for row_index, sequence in iter_text_sequences(path):
            total_rows += 1
            forward_hash, reverse_hash = sequence_hashes(sequence)
            if not forward_hash:
                continue

            if forward_hash in unique_hashes:
                duplicate_rows += 1
            unique_hashes.add(forward_hash)

            forward_splits = [
                name
                for name, hashes in expression_hashes_by_split.items()
                if forward_hash in hashes
            ]
            reverse_splits = [
                name
                for name, hashes in expression_hashes_by_split.items()
                if reverse_hash != forward_hash and reverse_hash in hashes
            ]

            has_forward = bool(forward_splits)
            has_reverse = bool(reverse_splits)
            has_either = has_forward or has_reverse

            if has_forward:
                forward_rows += 1
                forward_matches.add(forward_hash)
            if has_reverse:
                reverse_rows += 1
                reverse_matches.add(reverse_hash)
            if has_either:
                either_rows += 1
                either_matches.add(forward_hash)

            for split_name in sorted(set(forward_splits + reverse_splits)):
                match_kind = []
                if split_name in forward_splits:
                    match_kind.append("forward")
                if split_name in reverse_splits:
                    match_kind.append("reverse_complement")
                split_match_counts[(split_name, "+".join(match_kind))] += 1

            if has_either:
                if has_forward and has_reverse:
                    match_type = "forward_and_reverse_complement"
                elif has_forward:
                    match_type = "forward"
                else:
                    match_type = "reverse_complement"

                file_examples = sum(
                    1 for row in example_rows if row["mlm_file"] == filename
                )
                if file_examples < max_examples:
                    example_rows.append(
                        {
                            "mlm_file": filename,
                            "mlm_row": row_index,
                            "match_type": match_type,
                            "mlm_sequence_sha256": forward_hash,
                            "mlm_reverse_complement_sha256": reverse_hash,
                            "expression_splits": ",".join(
                                sorted(set(forward_splits + reverse_splits))
                            ),
                            "sequence_length": len(sequence),
                        }
                    )

        summary_rows.append(
            {
                "mlm_file": filename,
                "total_nonempty_rows": total_rows,
                "unique_mlm_sequences": len(unique_hashes),
                "duplicate_rows": duplicate_rows,
                "forward_match_rows": forward_rows,
                "forward_match_unique_sequences": len(forward_matches),
                "reverse_complement_match_rows": reverse_rows,
                "reverse_complement_match_unique_sequences": len(reverse_matches),
                "either_orientation_match_rows": either_rows,
                "either_orientation_match_unique_sequences": len(either_matches),
            }
        )

        for (split_name, match_kind), count in sorted(split_match_counts.items()):
            split_rows.append(
                {
                    "mlm_file": filename,
                    "expression_source_split": split_name,
                    "match_kind": match_kind,
                    "matched_rows": count,
                }
            )

    with (audit_root / "exact_overlap_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = list(summary_rows[0]) if summary_rows else [
            "mlm_file",
            "total_nonempty_rows",
            "unique_mlm_sequences",
            "duplicate_rows",
            "forward_match_rows",
            "forward_match_unique_sequences",
            "reverse_complement_match_rows",
            "reverse_complement_match_unique_sequences",
            "either_orientation_match_rows",
            "either_orientation_match_unique_sequences",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    with (audit_root / "exact_overlap_by_expression_split.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "mlm_file",
            "expression_source_split",
            "match_kind",
            "matched_rows",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(split_rows)

    with (audit_root / "exact_overlap_examples.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "mlm_file",
            "mlm_row",
            "match_type",
            "mlm_sequence_sha256",
            "mlm_reverse_complement_sha256",
            "expression_splits",
            "sequence_length",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(example_rows)

    return summary_rows, split_rows


def read_cluster_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t") if sample else csv.excel_tab
        reader = csv.DictReader(handle, dialect=dialect)
        rows = list(reader)

    required = {"sequence_sha256", "cluster_id", "split"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(
            f"Cluster manifest requires {sorted(required)}; "
            f"found {reader.fieldnames}."
        )
    return {"rows": rows, "fieldnames": list(reader.fieldnames or [])}


def audit_supplied_cluster_manifest(
    path: Path,
    mlm_paths: dict[str, Path],
    audit_root: Path,
) -> dict:
    manifest = read_cluster_manifest(path)
    rows = manifest["rows"]
    cluster_split_sets: dict[str, set[str]] = defaultdict(set)
    hash_to_cluster: dict[str, str] = {}
    has_source_column = "source" in manifest["fieldnames"]

    for row in rows:
        sequence_hash = str(row["sequence_sha256"]).strip()
        cluster_id = str(row["cluster_id"]).strip()
        split = str(row["split"]).strip()
        if not sequence_hash or not cluster_id or not split:
            continue
        hash_to_cluster[sequence_hash] = cluster_id
        cluster_split_sets[cluster_id].add(split)

    split_leaking_clusters = [
        {
            "cluster_id": cluster_id,
            "splits": ",".join(sorted(splits)),
        }
        for cluster_id, splits in cluster_split_sets.items()
        if len(splits) > 1
    ]

    mlm_exact_cluster_rows = []
    for filename, path_value in mlm_paths.items():
        matched = Counter()
        for _, sequence in iter_text_sequences(path_value):
            forward_hash, reverse_hash = sequence_hashes(sequence)
            cluster_id = hash_to_cluster.get(forward_hash) or hash_to_cluster.get(
                reverse_hash
            )
            if cluster_id:
                matched[cluster_id] += 1
        for cluster_id, count in sorted(matched.items()):
            mlm_exact_cluster_rows.append(
                {
                    "mlm_file": filename,
                    "cluster_id": cluster_id,
                    "matched_mlm_rows": count,
                }
            )

    with (audit_root / "expression_cluster_split_leakage.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["cluster_id", "splits"],
        )
        writer.writeheader()
        writer.writerows(split_leaking_clusters)

    with (audit_root / "cluster_overlap.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["mlm_file", "cluster_id", "matched_mlm_rows"],
        )
        writer.writeheader()
        writer.writerows(mlm_exact_cluster_rows)

    result = {
        "status": (
            "partial_supplied_manifest"
            if not has_source_column
            else "supplied_manifest_requires_review"
        ),
        "manifest": str(path.resolve()),
        "manifest_rows": len(rows),
        "manifest_unique_clusters": len(cluster_split_sets),
        "clusters_spanning_multiple_expression_splits": len(
            split_leaking_clusters
        ),
        "mlm_exact_or_reverse_complement_cluster_matches": len(
            {row["cluster_id"] for row in mlm_exact_cluster_rows}
        ),
        "cross_dataset_near_duplicate_audit_complete": False,
        "reason": (
            "A downstream-only cluster manifest can validate split separation "
            "and exact MLM matches, but it cannot detect 80%-identity clusters "
            "shared with MLM without cluster assignments for the combined corpus."
        ),
    }
    with (audit_root / "cluster_audit_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)
    return result


def safe_record_id(prefix: str, source_name: str, row_index: int) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(source_name))
    return f"{prefix}__{safe_name}__{row_index}"


def write_combined_fasta(
    expression_splits: dict,
    expression_sequence_column_override: str | None,
    mlm_paths: dict[str, Path],
    output_path: Path,
) -> int:
    record_count = 0
    with output_path.open("w", encoding="ascii") as handle:
        for split_name, dataset in expression_splits.items():
            sequence_column = choose_sequence_column(
                list(dataset.column_names),
                expression_sequence_column_override,
            )
            for row_index, row in enumerate(dataset):
                sequence = normalize_sequence(row.get(sequence_column))
                if not sequence:
                    continue
                handle.write(
                    f">{safe_record_id('expr', split_name, row_index)}\n"
                    f"{sequence}\n"
                )
                record_count += 1

        for filename, path in mlm_paths.items():
            for row_index, sequence in iter_text_sequences(path):
                handle.write(
                    f">{safe_record_id('mlm', filename, row_index)}\n"
                    f"{sequence}\n"
                )
                record_count += 1
    return record_count


def record_id_source(record_id: str) -> tuple[str | None, str | None]:
    parts = record_id.split("__", 2)
    if len(parts) < 3:
        return None, None
    if parts[0] == "expr":
        return "expression", parts[1]
    if parts[0] == "mlm":
        return "mlm", parts[1]
    return None, None


def run_mmseqs_cluster_audit(
    expression_splits: dict,
    expression_sequence_column_override: str | None,
    mlm_paths: dict[str, Path],
    audit_root: Path,
    mmseqs_binary: str,
) -> dict:
    if not mmseqs_binary:
        raise FileNotFoundError(
            "MMseqs2 was requested but no executable was found. Set "
            "FLORABERT_MMSEQS_BIN or put mmseqs on PATH."
        )

    fasta_path = audit_root / "combined_expression_and_mlm.fa"
    record_count = write_combined_fasta(
        expression_splits,
        expression_sequence_column_override,
        mlm_paths,
        fasta_path,
    )
    mmseqs_root = audit_root / "mmseqs"
    mmseqs_root.mkdir(parents=True, exist_ok=True)
    output_prefix = mmseqs_root / "combined"
    tmp_root = mmseqs_root / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    command = [
        mmseqs_binary,
        "easy-cluster",
        str(fasta_path),
        str(output_prefix),
        str(tmp_root),
        "--min-seq-id",
        "0.8",
    ]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)

    cluster_tsv = Path(str(output_prefix) + "_cluster.tsv")
    if not cluster_tsv.is_file():
        raise FileNotFoundError(
            "MMseqs2 completed but expected output was not found: "
            f"{cluster_tsv}"
        )

    cluster_members: dict[str, set[str]] = defaultdict(set)
    with cluster_tsv.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                continue
            representative, member = fields[0], fields[1]
            cluster_members[representative].update((representative, member))

    cross_dataset_rows = []
    expression_split_leakage = []

    for cluster_id, members in sorted(cluster_members.items()):
        sources = set()
        expression_splits = set()
        mlm_files = set()

        for record_id in members:
            source_type, source_name = record_id_source(record_id)
            if source_type is None:
                raise ValueError(
                    f"Unexpected MMseqs record ID in cluster {cluster_id}: "
                    f"{record_id}"
                )
            sources.add(source_type)
            if source_type == "expression":
                expression_splits.add(source_name)
            else:
                mlm_files.add(source_name)

        if len(expression_splits) > 1:
            expression_split_leakage.append(
                {
                    "cluster_id": cluster_id,
                    "expression_splits": ",".join(sorted(expression_splits)),
                }
            )

        if sources == {"expression", "mlm"}:
            cross_dataset_rows.append(
                {
                    "cluster_id": cluster_id,
                    "expression_splits": ",".join(sorted(expression_splits)),
                    "mlm_files": ",".join(sorted(mlm_files)),
                    "cluster_record_count": len(members),
                }
            )

    with (audit_root / "cluster_overlap.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "cluster_id",
            "expression_splits",
            "mlm_files",
            "cluster_record_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cross_dataset_rows)

    with (audit_root / "expression_cluster_split_leakage.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = ["cluster_id", "expression_splits"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(expression_split_leakage)

    result = {
        "status": "complete_from_mmseqs2",
        "mmseqs_binary": str(mmseqs_binary),
        "min_seq_id": 0.8,
        "combined_fasta": str(fasta_path),
        "combined_fasta_records": record_count,
        "clusters": len(cluster_members),
        "cross_expression_mlm_clusters": len(cross_dataset_rows),
        "expression_clusters_spanning_multiple_hf_splits": len(
            expression_split_leakage
        ),
        "cross_dataset_near_duplicate_audit_complete": True,
    }
    with (audit_root / "cluster_audit_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2)
    return result


def load_expression_dataset(repo_id: str, cache_root: Path) -> dict:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The datasets package is required in the remote runtime. "
            "Install it with: pip install datasets"
        ) from exc

    loaded = load_dataset(
        repo_id,
        cache_dir=str(cache_root),
        **optional_token_kwargs(),
    )
    if hasattr(loaded, "items"):
        return dict(loaded.items())
    return {"default": loaded}


def download_mlm_files(repo_id: str, mlm_root: Path) -> dict[str, Path]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required in the remote runtime. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    paths = {}
    for filename in MLM_FILENAMES:
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=str(mlm_root),
            **optional_token_kwargs(),
        )
        path = Path(downloaded).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Downloaded MLM file is missing: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"Downloaded MLM file is empty: {path}")
        paths[filename] = path
    return paths


def main() -> int:
    args = parse_args()
    audit_root = args.audit_root.expanduser().resolve()
    cache_root = audit_root / "hf_cache"
    mlm_root = audit_root / "maize_mlm"
    audit_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    mlm_root.mkdir(parents=True, exist_ok=True)

    print("Audit root:", audit_root, flush=True)
    print("Expression dataset:", args.expression_repo, flush=True)
    print("Maize MLM dataset:", args.mlm_repo, flush=True)
    print("Run MMseqs2:", args.run_mmseqs, flush=True)
    print("HF token supplied by environment:", bool(optional_token_kwargs()), flush=True)

    mlm_paths = download_mlm_files(args.mlm_repo, mlm_root)
    expression_splits = load_expression_dataset(args.expression_repo, cache_root)

    print("Expression splits:", flush=True)
    for split_name, dataset in expression_splits.items():
        print(
            f"  {split_name}: {len(dataset):,} rows; "
            f"columns={dataset.column_names}",
            flush=True,
        )

    (
        expression_manifest,
        expression_hashes_by_split,
        expression_hash_to_splits,
        expression_counts,
    ) = write_expression_manifest(
        expression_splits,
        args.sequence_column,
        audit_root,
    )
    exact_summary_rows, exact_split_rows = audit_exact_overlaps(
        mlm_paths,
        expression_hashes_by_split,
        audit_root,
        args.max_examples,
    )

    exact_totals = {
        "total_either_orientation_match_rows": sum(
            row["either_orientation_match_rows"] for row in exact_summary_rows
        ),
        "total_forward_match_rows": sum(
            row["forward_match_rows"] for row in exact_summary_rows
        ),
        "total_reverse_complement_match_rows": sum(
            row["reverse_complement_match_rows"] for row in exact_summary_rows
        ),
    }

    if args.cluster_manifest:
        cluster_result = audit_supplied_cluster_manifest(
            args.cluster_manifest.expanduser().resolve(),
            mlm_paths,
            audit_root,
        )
    elif args.run_mmseqs:
        cluster_result = run_mmseqs_cluster_audit(
            expression_splits,
            args.sequence_column,
            mlm_paths,
            audit_root,
            args.mmseqs_bin,
        )
    else:
        cluster_result = {
            "status": "not_run",
            "cross_dataset_near_duplicate_audit_complete": False,
            "reason": (
                "No combined MMseqs2 audit was requested and no cluster manifest "
                "was supplied."
            ),
        }
        with (audit_root / "cluster_audit_summary.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(cluster_result, handle, indent=2)

    strict_inductive_ready = (
        exact_totals["total_either_orientation_match_rows"] == 0
        and cluster_result.get("status") == "complete_from_mmseqs2"
        and cluster_result.get("cross_expression_mlm_clusters", 0) == 0
        and cluster_result.get("expression_clusters_spanning_multiple_hf_splits", 0)
        == 0
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "expression_dataset": args.expression_repo,
        "mlm_dataset": args.mlm_repo,
        "mlm_files": list(MLM_FILENAMES),
        "expression_manifest": str(expression_manifest),
        "expression_counts": expression_counts,
        "exact_overlap": exact_totals,
        "exact_overlap_rows": exact_summary_rows,
        "cluster_audit": cluster_result,
        "strict_inductive_benchmark_ready": strict_inductive_ready,
        "interpretation": (
            "No exact/reverse-complement overlap and no 0.8-identity cluster "
            "overlap were observed under the configured audit."
            if strict_inductive_ready
            else
            "The strict inductive benchmark is not established. Inspect exact "
            "overlap reports and complete the combined cluster audit before "
            "interpreting architecture or maize-DAPT differences."
        ),
    }

    with (audit_root / "audit_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2), flush=True)
    print("Audit outputs:", flush=True)
    for path in sorted(audit_root.iterdir()):
        print(" ", path.name, flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
