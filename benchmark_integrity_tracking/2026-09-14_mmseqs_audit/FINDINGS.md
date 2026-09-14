# Benchmark-integrity findings: 2026-09-14 MMseqs2 audit

## Executive conclusion

The audit completed successfully, but the current data arrangement does not
support a strict inductive comparison of plant ModernBERT versus maize MLM
adaptation.

The maize promoter MLM corpus overlaps the gene-expression corpus heavily:

- 765,753 of 1,100,984 MLM rows (69.55%) match an expression sequence exactly
  or as its reverse complement;
- 138,947 MMseqs2 clusters contain both expression and MLM records at the
  configured 0.8 minimum sequence identity; and
- 10,359 MMseqs2 clusters contain expression records from both the expression
  train and expression test splits.

The machine-readable verdict is therefore:

```text
strict_inductive_benchmark_ready = false
```

The current maize-DAPT result should be labelled a transductive or
leakage-exposed experiment, not evidence of clean inductive generalization.
This does not mean the model run is unusable; it means its improvement cannot
be attributed solely to better biological generalization to unseen promoter
sequences.

## Evidence and provenance

Evidence archive:

- [`florabert_benchmark_audit_mmseqs.zip`](evidence/florabert_benchmark_audit_mmseqs.zip)
- SHA-256: `41c3a96d131f0769d62e384fd5d645189a9c4542012c1e6a2f02f7861e8ef2b2`
- Archive contents: eight report files; approximately 117.7 MB uncompressed
- The archive contains hash-only manifests and cluster metadata, not raw
  promoter sequences

Datasets:

- Expression: [`Gurveer05/maize-nam-gene-expression-data`](https://huggingface.co/datasets/Gurveer05/maize-nam-gene-expression-data)
- MLM: [`Gurveer05/maize-promoter-sequences`](https://huggingface.co/datasets/Gurveer05/maize-promoter-sequences)
- MLM files: `all_seqs_train.txt` and `all_seqs_test.txt`

Code and execution:

- Repository branch: `audit/benchmark-integrity`
- Implementation commit used for the Kaggle notebook: `d9736e9`
- Notebook: [`benchmark_integrity_audit_kaggle.ipynb`](../../notebooks/benchmark_integrity_audit_kaggle.ipynb)
- MMseqs2 workflow: `easy-cluster`
- Minimum sequence identity: `0.8`
- MMseqs2 split-memory limit: `20G`
- MMseqs2 threads: `2`
- The execution log reported MMseqs2 build identifier
  `a13af93468c997d73572c9326b69ac0c66c6583a`

The archive does not itself record the Kaggle runtime identifier, the precise
Hugging Face dataset revisions, or the repository commit printed by the
notebook at runtime. Those should be captured in a future run's provenance
file.

## Input counts

The expression dataset was loaded with two splits and the `sequence` column:

| Source split | Rows | Nonempty rows | Unique sequences |
|---|---:|---:|---:|
| `hf::train` | 424,066 | 424,066 | 371,997 |
| `hf::test` | 281,361 | 281,361 | 246,601 |

The MLM files contained:

| MLM file | Nonempty rows | Unique sequences | Duplicate rows |
|---|---:|---:|---:|
| `all_seqs_train.txt` | 770,688 | 655,842 | 114,846 |
| `all_seqs_test.txt` | 330,296 | 302,326 | 27,970 |

The combined MMseqs2 FASTA contained 1,806,411 records, exactly matching the
sum of all expression and MLM rows. No input rows were lost in construction of
the combined audit input.

## Exact and reverse-complement overlap

The audit normalizes sequences by removing whitespace and uppercasing them. It
hashes expression sequences in the forward orientation, then checks each MLM
sequence in both forward and reverse-complement orientations.

The `either_orientation` count is a union. Forward and reverse-complement
counts must not be added because some rows match in both orientations.

| MLM file | Either-orientation matching rows | Rate | Matching unique sequences | Rate |
|---|---:|---:|---:|---:|
| `all_seqs_train.txt` | 536,003 / 770,688 | 69.55% | 455,520 / 655,842 | 69.46% |
| `all_seqs_test.txt` | 229,750 / 330,296 | 69.56% | 210,056 / 302,326 | 69.48% |
| **Combined** | **765,753 / 1,100,984** | **69.55%** | — | — |

Forward matches dominate, but reverse-complement matches are also present:

- MLM train: 531,494 forward rows and 18,769 reverse-complement rows;
- MLM test: 227,890 forward rows and 7,965 reverse-complement rows.

The per-expression-split report shows that both MLM files match both the
expression train and expression test data. Therefore the issue is not confined
to a single downstream split or to a possible MLM test-file edge case.

The expression manifest itself contains 74 unique sequences shared exactly
between the expression train and test splits. This exact split duplication is
small compared with the near-duplicate cluster leakage below, but it should
also be removed for a strict split.

## MMseqs2 cluster audit

The completed cluster report contains:

```text
total clusters:                         210,559
cross expression/MLM clusters:          138,947
expression clusters spanning splits:      10,359
```

The 138,947 cross-dataset clusters break down as follows:

| Expression split membership | Cross-dataset clusters |
|---|---:|
| Expression train only | 78,363 |
| Expression test only | 50,225 |
| Both expression splits | 10,359 |

By MLM membership:

| MLM membership | Cross-dataset clusters |
|---|---:|
| MLM train only | 51,730 |
| MLM test only | 16,492 |
| Both MLM files | 70,725 |

Summing the cluster member counts reported for cross-dataset clusters gives
1,640,618 record memberships, or approximately 90.82% of the 1,806,411
combined records. This is a record-membership total, not a count of unique
genes or independent biological loci. MMseqs2 clustering is also potentially
transitive: membership in the same cluster does not mean every pair of records
has a direct pairwise 80% identity alignment.

The expression train/test split is therefore not cluster-independent at the
same 0.8-identity criterion. This affects the baseline plant-checkpoint
regression run as well as the maize-adapted run.

## What this means for model comparisons

The present plant-to-regression and plant-to-maize-MLM-to-regression results
are not a clean inductive ablation because:

1. the MLM corpus has direct sequence exposure to most downstream sequences;
2. the MLM corpus has extensive near-duplicate exposure at the configured
   identity threshold; and
3. expression train and test contain substantial near-duplicate clusters.

The comparison can still answer a narrower question: whether continued MLM on
this maize corpus changes performance under the current, transductive data
arrangement. It cannot establish that the adapted model generalizes better to
unseen promoter families.

## Archive limitations

The ZIP contains `cluster_overlap.csv`, but that file records one row per
cross-dataset cluster with the cluster ID, source labels, and member count. It
does not contain the full record-to-cluster membership mapping. Consequently,
the ZIP alone is insufficient to construct a filtered MLM corpus that removes
all affected near-duplicate clusters.

The full MMseqs2 `_cluster.tsv` membership output should be retained from the
Kaggle scratch directory, or the audit script should be extended to export a
hash-only record-level cluster manifest. The latter is preferable for future
reproducibility and filtering.

## Recommended next steps

### 1. Freeze the audit evidence

Preserve the ZIP, its SHA-256, the complete MMseqs2 cluster membership TSV, the
Kaggle execution metadata, the exact repository commit, MMseqs2 version, and
Hugging Face dataset revisions.

### 2. Create a leakage-controlled supervised split

Cluster all expression sequences before assigning train, validation, and test
sets. Assign whole clusters to one split only. Keep the final test clusters
untouched during both regression and any model-selection process.

### 3. Create a leakage-controlled MLM corpus

For a strict inductive DAPT condition, remove from the MLM corpus:

- every exact forward match to an expression sequence;
- every reverse-complement match; and
- every MLM record in a cross-dataset near-duplicate cluster at the chosen
  threshold.

The final filtering should use record-level cluster assignments, not only the
summary ZIP currently available.

### 4. Rerun the controlled ablation

Using the same plant checkpoint, tokenizer, MLM recipe, regression recipe, and
evaluation semantics, compare:

- plant-pretrained ModernBERT → regression; and
- the same plant-pretrained ModernBERT → leakage-controlled maize MLM →
  regression.

Report validation and held-out test metrics separately, including conventional
sklearn R², Pearson r², MSE, prediction/target mean and standard deviation, and
per-tissue results where practical.

### 5. Keep the current result as a labelled secondary condition

Do not discard the existing run. Record it as transductive or
leakage-exposed DAPT, and do not combine it with the strict inductive result in
the same headline claim.

### 6. Add sensitivity checks after the primary controlled run

Once the 0.8-identity controlled benchmark is working, repeat the audit at one
or more stricter identity thresholds to show how conclusions depend on the
definition of a near duplicate. This should be a documented sensitivity
analysis, not a replacement for the pre-specified primary threshold.
