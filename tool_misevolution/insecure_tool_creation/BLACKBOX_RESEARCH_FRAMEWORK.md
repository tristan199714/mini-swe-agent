# Black-Box Research Framework For Self-Evolving Tool Agents

## Research Question
How do self-evolving tool agents become less safe under black-box interaction when tools can be created, stored, retrieved, and reused across task sequences?

## Core Thesis
The main attack surface is stateful. A black-box attacker does not need weight access if they can shape the agent state through:
- tool pool admission
- retrieval bias
- lineage reuse
- delayed trigger propagation

## Threat Model
- The attacker cannot inspect model weights, gradients, or training data.
- The attacker can control task ordering, task wording, and repeated interaction rounds.
- The attacker can influence whether generated tools enter the reusable pool.
- The attacker aims to induce persistent unsafe behavior across later tasks.

## Unit Of Analysis
This project should study three linked layers.

### 1. Single-Turn Layer
Question: does one task create an unsafe or risk-elevated tool?
Main observables:
- tool created
- risk score
- trigger indicators
- judge score

### 2. Evolution Layer
Question: does the created tool survive and become part of the reusable memory?
Main observables:
- tool admitted to pool
- clean vs poison pool placement
- promotion into retrieval-visible state
- persistence across later rounds

### 3. Chain Layer
Question: do later tasks retrieve and reuse the risky lineage, and does that lower downstream safety?
Main observables:
- lineage reuse rate
- retrieval rank / whether pinned lineage was surfaced
- reuse-verified steps
- downstream score delta

## Primary Hypotheses
- H1: Stateful tool reuse produces lower safety scores than one-shot baselines under matched task sets.
- H2: Cross-version lineage reuse is a stronger driver than single-turn unsafe generation.
- H3: Pool persistence plus retrieval is necessary for sustained degradation.
- H4: Higher-version tasks concentrate most of the degradation effect.

## Experimental Conditions
### Main Conditions
- `ours_full`: full v1-v4 chain
- `ours_strong`: focused v3-v4 chain
- `direct_trigger`: v4 only
- `random_sequence`: shuffled v1-v4 order
- `fixed_simple`: v1 only

### Mechanism Ablations
- `ours_no_reuse`: disable lineage reuse while keeping tool creation and task schedule
- `ours_no_memory`: disable MCP matching and online pool update
- `ours_no_pinned_retrieval`: keep matching but disable source-lineage pinning
- `ours_shuffled`: keep same tasks but shuffle order to break controlled buildup

## Metrics
### Outcome Metrics
- low score rate
- mean judge score
- matched-pair delta against baselines
- per-seed variance

### Mechanism Metrics
- tool creation rate
- tool pool admission rate
- tool promotion rate
- lineage reuse rate
- trigger success rate
- trust score trajectory
- time to first compromised lineage

### Breakdown Metrics
- per-version breakdown
- per-CWE breakdown
- per-lineage breakdown
- failure reason breakdown

## Minimum Logging Requirements
Each task result should preserve:
- task name
- base task / version
- cwe id
- lineage id
- whether a prior tool existed
- whether a prior tool was retrieved
- whether reuse was verified
- trigger indicators
- trust score
- judge score
- failure reason if unsuccessful

## Recommended Tables
### Main Table
- ours_full
- ours_strong
- direct_trigger
- random_sequence
- fixed_simple

### Mechanism Table
- ours_full
- ours_no_reuse
- ours_no_memory
- ours_no_pinned_retrieval
- ours_shuffled

### Breakdown Table
- v1 / v2 / v3 / v4 low score rate
- top vulnerable CWE families
- top vulnerable lineages

## Recommended Figures
- tool lineage reuse graph
- per-version low score plot
- per-seed delta boxplot
- failure reason distribution
- trigger success vs judge score scatter

## Defense Hooks
The same framework can evaluate defenses placed at:
- tool admission
- tool promotion
- retrieval ranking
- reuse confirmation
- execution-time audit
- quarantine / rollback

## Success Criterion For This Codebase
A good experiment setup should let the researcher answer:
- Did the attack lower safety scores?
- Did it do so through pool admission, retrieval, and reuse rather than one-shot luck?
- Which versions, CWEs, and lineages drove the effect?
- Which defense hook would most directly interrupt the chain?
