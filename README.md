# Oracle

**Predictive-responsive trading framework for Indian derivatives markets.**

Oracle is a Machine Learning / Deep Learning framework that generates probabilistic
information about volatility events and, where warranted, translates that information
into trading positions via the Atlas product line.

## Framework Identity

- **Oracle is NOT an extension of Helios.** Complete isolation at infrastructure, data, code, and capital levels.
- **Specification-first methodology.** Specifications are frozen before code is written.
- **Governance-dominant.** Oracle is a Control System that uses Machine Learning — not a Machine Learning System with controls.

## Repository Structure
```
src/oracle/          — Core application code
  ingestion/         — Market data ingestion pipeline
  features/          — Feature store and computation
  models/            — Model serving and inference
  pipeline/          — Probability pipeline (Probability → Eligibility → Structure → Size → Exit)
  governance/        — Governance enforcement
  odal/              — Oracle Data Access Layer
  config/            — Configuration service
  utils/             — Shared utilities
tests/               — Test suites (unit, integration, spec compliance)
infrastructure/      — AWS resource definitions (DynamoDB, IAM)
docs/                — Specifications, TDRs, consistency logs
data/                — Reference data (event calendar, expiry calendar)
research/            — Notebooks and experiments (not deployed)
```

## Key Specifications

- Oracle Master Governance Document v0.2
- Oracle Architecture Specification v0.2
- Oracle-V Product Specification v0.1
- Oracle-V Data Contract v0.1

## Technology

- **Python:** 3.11+
- **AWS Region:** ap-south-1 (Mumbai)
- **Persistence:** DynamoDB (Oracle_ prefix)
- **ML Framework:** TBD (TDR pending)

## Isolation

This repository has zero dependencies on Helios. No imports, no shared data, no shared config.
CI/CD will enforce this boundary per Architecture §16.4 (CODE-001 through CODE-008).
