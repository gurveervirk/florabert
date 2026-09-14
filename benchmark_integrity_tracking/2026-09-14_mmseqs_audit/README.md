# 2026-09-14 MMseqs2 benchmark-integrity audit

Status: completed, with strict inductive benchmark rejected.

Primary findings are in [`FINDINGS.md`](FINDINGS.md). The evidence archive is:

[`evidence/florabert_benchmark_audit_mmseqs.zip`](evidence/florabert_benchmark_audit_mmseqs.zip)

The archive checksum is recorded in
[`evidence/SHA256SUMS.txt`](evidence/SHA256SUMS.txt). The original archive that
was supplied for inspection remains at the repository root as
`florabert_benchmark_audit_mmseqs.zip`; the evidence copy was verified against
the same checksum.

Relevant code and notebook:

- [`audit_benchmark_integrity.py`](../../scripts/0-data-loading-processing/audit_benchmark_integrity.py)
- [`benchmark_integrity_audit_kaggle.ipynb`](../../notebooks/benchmark_integrity_audit_kaggle.ipynb)
- Git branch: `audit/benchmark-integrity`
- Audit implementation commit: `d9736e9` (`Add Kaggle benchmark integrity audit`)
