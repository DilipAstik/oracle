# ORACLE FRAMEWORK — DORMANT OPERATING CHARTER (v1.0)

**Effective Date:** 2026-02-21
**Status:** Controlled Dormancy (Data Accumulation Phase)
**Document ID:** ORACLE-DOC-001

---

## 1. Purpose

This Charter defines the governance, maintenance, and review protocol for the **Oracle Framework** during its Dormant Phase following completion of the Oracle-V Research Cycle.

Dormancy does **not** represent project suspension.
It represents a **time-dependent maturation period** required to accumulate sufficient longitudinal market data before predictive research can be responsibly resumed.

Oracle transitions from an **Active Research System** to a **Passive Intelligence Infrastructure**.

### 1.1 Document Lineage

| Relationship | Document |
|---|---|
| Governing documents | Oracle Master Governance Document v0.2, Oracle-V Product Specification v0.1 |
| Triggering document | R7 Research Exit Report (ORACLE-V-RESEARCH-EXIT-001, 2026-02-21) |
| Companion document | Oracle-V Research Cycle 1 — R1–R6 Results (committed to Git) |
| Superseded by | Oracle Reactivation Decision Record (upon exit from Dormancy) |

---

## 2. Strategic Rationale

Oracle-V demonstrated:

- Infrastructure and methodology are sound.
- Predictive feasibility is constrained primarily by **dataset length and regime diversity**, not model design.
- Additional modelling effort without additional data would create optimisation bias rather than insight.

Therefore Oracle must now **collect market history, not consume development bandwidth**.

---

## 3. Dormant Phase Objectives

During this phase Oracle shall:

1. **Continuously accumulate clean, versioned derivatives datasets.**
2. **Preserve reproducibility of the research pipeline.**
3. **Track dataset maturation metrics required for reactivation.**
4. **Integrate slow-moving structural data sources (e.g., VIX, regulatory signals).**
5. **Avoid speculative model iteration.**

Oracle becomes a **market observatory**, not a trading intelligence engine.

---

## 4. Operating Mode

| Dimension | Active Phase | Dormant Phase |
|---|---|---|
| Research Activity | Model development | None |
| Data Collection | Supporting research | Primary mission |
| Engineering Change | Frequent | Minimal / controlled |
| Validation | Continuous | Quarterly |
| Output | Predictions | Dataset maturity diagnostics |

---

## 5. Permitted Activities

The following workstreams are allowed:

### 5.1 Passive Data Operations

- **C1 (Daily ingestion pipeline)** remains operational — automated, zero manual intervention. NSE F&O Bhavcopy files downloaded and ingested into S3 every trading day.
- **C2 (IV computation, baselines, labels)** runs monthly as a data quality health check (~15 minutes). Verifies IV convergence rates remain >95% and label distributions are within expected bounds.
- **C3 (Feature engineering)** does **not** run during Dormancy. Feature computation executes only upon formal reactivation.
- Schema stability is maintained across all stages.
- Data validation logs retained per Governance.

### 5.2 Data Enrichment (Low-Frequency Enhancements)

Allowed because they improve dataset completeness without introducing modelling bias.

#### 5.2.1 Priority Enrichment: India VIX Integration

Integration of India VIX daily data into the ingestion pipeline is the **highest-priority enrichment task**. This unlocks three deferred CRITICAL features (`india_vix_level`, `india_vix_percentile_rank`, `india_vix_change`) and may enable accelerated reactivation per §8.1 below.

**Target completion:** Within first quarter of Dormancy.

#### 5.2.2 Secondary Enrichment

- F&O regulatory signals (ban lists, structural changes)
- Event calendar archival datasets
- Data quality backfills if required

These are **additive**, not analytical.

### 5.3 Infrastructure Preservation

- Dependency updates required for security or runtime stability.
- Reproducibility tests.
- Storage lifecycle management.

---

## 6. Prohibited Activities

To avoid research drift, the following are **not permitted** during Dormancy:

- New Oracle predictive products (R, S, T, etc.)
- Feature engineering aimed at improving model performance.
- Hyperparameter optimisation.
- Backtesting for signal discovery.
- Publication of probabilistic outputs for trading use.
- Any attempt to "salvage" Oracle-V via incremental tweaks.

These activities resume only upon formal reactivation.

---

## 7. Quarterly Preservation Review (QPR)

Instead of continuous intervention, Oracle is validated every **90 days**.

### 7.1 Review Tasks (Half-Day Cycle)

1. Run full pipeline reproducibility check.
2. Generate Dataset Growth Report:
   - Total trading days accumulated
   - Regime dispersion indicators (rolling 60-day expansion base rates per §8 definitions)
   - Label distribution drift vs original research
3. Validate feature availability progression.
4. Run baseline statistical diagnostics (no ML training).
5. Archive report:
   ```
   /oracle/governance/qpr/Oracle_QPR_<YYYY>_Q<N>.md
   ```

### 7.2 Review Output

A single classification:

- **STABLE** — Continue Dormant Mode
- **ENRICH** — Minor data additions required
- **READY-FOR-REASSESSMENT** — Approaching activation threshold

### 7.3 Escalation Protocol

If a QPR discovers data integrity issues (ingestion gaps, convergence degradation, storage failures), the issue is classified and escalated:

| Severity | Condition | Response |
|---|---|---|
| MINOR | Ingestion gap ≤ 10 trading days, data backfillable | Backfill within current QPR cycle |
| MAJOR | Ingestion gap > 10 trading days, or IV convergence rate below 90%, or data corruption detected | Immediate remediation outside QPR schedule |
| CRITICAL | Pipeline non-functional for 30+ consecutive trading days | Emergency review — assess whether accumulated data is compromised; document impact on reactivation timeline |

MAJOR and CRITICAL issues are logged in `/oracle/governance/incidents/` with root cause analysis and remediation confirmation.

---

## 8. Reactivation Criteria (Objective Triggers)

Oracle exits Dormancy only when **all** of the following criteria are met:

| Criterion | Threshold | Measurement Method |
|---|---|---|
| Minimum dataset length | ≥ 750 trading days | Count of unique trading days in S3 with successfully ingested Bhavcopy |
| Regime diversity | ≥ 3 volatility regimes observed | See §8.2 below |
| Event coverage | At least one macro dislocation cycle | See §8.3 below |
| Feature completeness | All CRITICAL features populated | All 8 CRITICAL features (§5.2.6 of Product Spec) available with ≥85% coverage |
| Pipeline stability | 4 consecutive QPR "STABLE" ratings | No MAJOR or CRITICAL escalations in the 4 most recent QPRs |

Reactivation is governance-driven, not discretionary.

### 8.1 Accelerated Reactivation Path

If the following conditions are **all** met, Oracle may be reactivated before the 750-day primary threshold:

| Condition | Threshold |
|---|---|
| India VIX integration complete | 3 VIX features populated with ≥85% coverage |
| Dataset length | ≥ 650 trading days |
| Regime diversity | ≥ 3 volatility regimes observed (§8.2) |
| Pipeline stability | 4 consecutive QPR "STABLE" ratings |

The accelerated path requires a **documented justification** that VIX features provide sufficient regime context to compensate for the reduced dataset length. This justification is reviewed during the QPR that triggers READY-FOR-REASSESSMENT and must be approved before research resumes.

### 8.2 Volatility Regime Definition

A volatility regime is defined as a sustained period (≥ 40 trading days) where the rolling 60-day expansion base rate falls into one of three bands:

| Regime | Rolling 60-Day Expansion Base Rate |
|---|---|
| Low-Expansion | < 10% |
| Medium-Expansion | 10% – 25% |
| High-Expansion | > 25% |

The dataset must contain at least one period of ≥ 40 trading days in **each** band, assessed independently for each underlying (NIFTY, BANKNIFTY). This uses the R1g rolling base rate methodology established during Research Cycle 1.

### 8.3 Macro Dislocation Definition

The dataset must include at least one period satisfying **either** of the following:

- NIFTY experienced a drawdown ≥ 5% within 10 trading days, OR
- India VIX (when available) exceeded its 90th historical percentile for ≥ 5 consecutive trading days

This ensures the training data contains at least one stress episode, providing the model with examples of non-normal market behaviour.

---

## 9. Resource Allocation Guidance

| Activity | Expected Time |
|---|---|
| Daily automated ingestion | 0 manual hours (automated) |
| Weekly ingestion health check | ~10 minutes |
| Monthly C2 pipeline health check | ~15 minutes |
| Quarterly Preservation Review (QPR) | ~4 hours |
| Data enrichment tasks | Occasional (≤ 2 days per quarter) |

**Total ongoing commitment:** ~30 minutes per week, ~1 hour per month, ~4 hours per quarter.

Primary organisational focus remains on **Helios live deployment and capital operations**.

---

## 10. Risk Management During Dormancy

Dormancy avoids three known research risks:

| Risk | How Dormancy Mitigates |
|---|---|
| **Overfitting-by-Iteration** | No model training on the same short dataset |
| **Narrative Bias** | No forced interpretation of predictive structure where none yet exists |
| **Infrastructure Decay** | Quarterly preservation reviews catch degradation before it compounds |

---

## 11. Expected Duration

Estimated Dormant Phase: **12–18 months** (data-dependent, not calendar-driven).

| Path | Estimated Reactivation | Condition |
|---|---|---|
| Primary (750 days) | ~Jan 2027 | Standard threshold met |
| Accelerated (650 days + VIX) | ~Oct 2026 | VIX integrated, documented justification approved |
| Extended | Beyond Jan 2027 | Regime diversity or pipeline stability criteria not met |

Oracle's edge is expected to emerge from **time aggregation**, not algorithmic novelty.

---

## 12. Governance Status Statement

**Operator Reminder:**

> Oracle is not paused. Oracle is observing.
>
> The Framework is functioning as a long-horizon intelligence asset whose value
> compounds silently until sufficient market history exists to support
> statistically defensible prediction.

---

## Document Control

| Version | Date | Change |
|---|---|---|
| 1.0 | 2026-02-21 | Initial charter following R7 Research Exit Report |

---

**Approved By:** _________________________
**Date:** ________________________________
