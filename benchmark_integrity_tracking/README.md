# Benchmark-integrity tracking

This directory records benchmark-integrity audits and the evidence used to
interpret FloraBERT experiments. Each dated subdirectory is intended to be a
self-contained audit record with:

- a findings report;
- provenance and configuration details;
- checksums for local evidence archives; and
- links to the code and notebooks that produced the audit.

The first record is [`2026-09-14_mmseqs_audit`](2026-09-14_mmseqs_audit/).

Large evidence archives are kept beside their reports for local tracking. The
first audit archive is included with its dated record because it is the primary
evidence for the findings. Future large archives should be committed only when
their provenance value justifies the repository size; each report must record
the exact SHA-256 checksum either way.
