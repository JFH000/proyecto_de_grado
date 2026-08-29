# Summary: "Multi-Agent Reinforcement Learning for Autonomous Driving: A Survey"

**arXiv:** [2408.09675](https://arxiv.org/abs/2408.09675)
**Authors:** Ruiqi Zhang, Jing Hou, Florian Walter, Shangding Gu, Jiayi Guan, Florian Röhrbein, Yali Du, Panpan Cai, Guang Chen, Alois Knoll (Tongji Univ. / TU Munich / UC Berkeley et al.)
**Type:** Survey (IEEE journal preprint), reviews 200+ publications
**Companion repo:** https://github.com/ispc-lab/MARL4AD (maintained, continuously updated)

## What the paper covers

A comprehensive survey of MARL applied to autonomous driving (AD) and intelligent transportation systems, organized into four parts:

1. **Benchmarks** — proposes 6 criteria for judging simulators (realism/fidelity, scalability, diversity, efficiency, transferability, maintenance/support), then surveys simulators split into two families:
   - *Traffic-flow-oriented* (SUMO, Flow, Highway-env, CityFlow, SMARTS, MetaDrive, Waymax, etc.) — prioritize parallelization/throughput over rendering fidelity.
   - *Fidelity-oriented* (TORCS, CARLA, MACAD, NVIDIA Isaac Sim, VISTA) — realistic sensors/physics, far more expensive to run.
   Also covers driving datasets (INTERACTION, AD4RL) and competitions (SMARTS @ NeurIPS, CARLA Leaderboard, AI-DO).

2. **RL/MARL preliminaries** — standard MDP → Markov Game → Dec-POMDP formalization, on/off-policy and value/policy-based RL, then the central architectural split for MARL:
   - **CTDE** (Centralized Training, Decentralized Execution): a central critic sees all agents' joint observations/actions during training; only local actors run at test time. Assumes "unconstrained and instantaneous information exchange" during training — the survey flags this as unrealistic for real fleets.
   - **DTDE** (Decentralized Training, Decentralized Execution): agents train and act independently, communicating (if at all) only with nearby peers. More scalable, but suffers non-stationarity and partial observability.
   Also surveys the standard MARL pain points: non-stationarity, partial observability, credit assignment, **scalability** (joint action space grows exponentially with agent count — this is the "curse of dimensionality" phenomenon your advisor referenced from the 90s), and communication-mechanism design (predefined vs. learned protocols).

3. **SOTA methods**, categorized as centralized MARL (PRIMAL, MADDPG, MAPPO), independent policy optimization / DTDE (MAPPER, G2RL, PRIMAL₂, PIPO), **learning with social preference** (CoPO, SAPO — see below, most relevant to your thesis), and safe/trustworthy RL (soft reward penalties → Lagrangian/probabilistic constraints → hard control-barrier-function guarantees). A comparison table (Table 2) catalogs ~15 methods by training scheme, simulator, max agent count, vehicle kinematics, action space, and whether real-world tests were run.

4. **Open challenges & future directions** — multi-modal sensor fusion, long-tail/sim-to-real robustness, safety certification, **explainability**, and forward-looking bets on model-based MARL, offline MARL datasets, human-in-the-loop learning, and LLMs/VLMs for driving.

## The two algorithms most worth reading in full

- **CoPO** (Peng et al., NeurIPS 2021) — [Learning to Simulate Self-Driven Particles Systems with Coordinated Policy Optimization](https://arxiv.org/abs/2110.13827). DTDE scheme on MetaDrive (up to 40 heterogeneous vehicles). Each agent optimizes a convex combination of its own reward and its neighbors' average reward, `r_i^C = cos(φ)·r_i + sin(φ)·r_i^N`, where the mixing angle φ (the "Local Coordination Factor") is itself **learned** via a bi-level/hierarchical gradient (local policy update, then a global-objective gradient step on φ's distribution parameters). This is a decentralized-training scheme with a *learned global coordination signal* derived without full centralization — architecturally the closest thing in this survey to what a federated aggregation step could replace or augment.
- **SAPO** (Dai et al., CoRL 2023) — [Socially-Attentive Policy Optimization in Multi-Agent Self-Driving System](https://arxiv.org/abs/2311.14922). Extends the CoPO idea with a multi-head attention layer over neighbors (SMARTS simulator) to *select* which nearby agent matters most, rather than pooling all neighbors uniformly.

## Connection to your project (Federated Multi-Agent RL)

Your two current workstreams (per `resumenes/resumen_reunion_2.md`) are (1) characterizing multi-agent training scenarios across papers, and (2) assessing how transferable CTDE/DTDE techniques are to a federated setting. This survey is a strong source for both:

- **It confirms the gap directly.** Across 200+ reviewed papers, MARL-for-AD work is cleanly split into CTDE (real-time centralized critic, i.e., synchronous and communication-heavy) and DTDE (independent or locally-communicating agents, no aggregation step at all). **None of the surveyed methods do anything resembling periodic federated parameter aggregation** (local training rounds + intermittent averaging/sync, à la FedAvg). That's worth stating explicitly as evidence in your thesis's gap analysis — this is a live, unaddressed niche in the AD/MARL literature specifically, not just in MARL generally.
- **CTDE's central critic vs. a federated server.** CTDE already assumes a component that aggregates information from all agents — but only *during training*, and with an unrealistic "unconstrained, instantaneous" communication assumption. That's precisely the assumption federated learning is designed to relax (bounded bandwidth, periodic rounds, no raw-data sharing). Reframing CTDE's central critic as a federated aggregator (partial-information, periodic sync, aggregating *parameters/gradients* rather than raw joint observations) is a natural and well-motivated angle for your "transferability of CTDE/DTDE to federated" question — you could use CoPO or MADDPG as the CTDE baseline to federate.
- **The scalability discussion is your reunion-1 "more agents → worse training" phenomenon, formalized.** Section 4.4 explicitly ties it to exponential joint-action-space growth and cites two direct algorithmic answers worth reading next:
  - Qu, Lin, Wierman & Li, *"Scalable Multi-Agent Reinforcement Learning for Networked Systems with Average Reward"* (NeurIPS 2020) — scalable actor-critic exploiting exponential decay of influence over network distance, for networked (not fully-connected) agents.
  - Egorov & Shpilman, *"Scalable Multi-Agent Model-Based Reinforcement Learning"* (AAMAS 2022, "MAMBA") — each agent keeps its own learned world model and generates synthetic rollouts, removing the need for constant environment interaction. This model-based-per-agent design is also structurally federated-friendly: each agent's world model is a natural candidate for local training + periodic aggregation.
- **Section 5.3.2 ("Private Information Concern") is the survey's explicit privacy/decentralization motivation** — it identifies that MARL fleets must constantly exchange position/trajectory data, that this is a real privacy/security exposure, and proposes decentralized data storage as a mitigation, citing Sharma & Park, *"Blockchain based hybrid network architecture for the smart city"* (2018). This is a one-paragraph nod rather than a worked solution — i.e., the survey itself notices the *motivation* for federated-style architectures (don't centralize raw data) but doesn't connect it to any federated-learning MARL method. That's a citable hook for your intro/motivation section: "even surveys of the field note the privacy exposure of MARL communication, but no reviewed method addresses it via federated learning."
- **Simulator choice for your own implementation.** Given the thesis will likely need an experimental testbed, MetaDrive (used by CoPO) and SMARTS (used by SAPO, Huawei-maintained, both actively updated as of 2024) stand out from the survey's own criteria as the best fit: both have native MARL support, are traffic-flow-oriented (cheap enough to run many agents/many training rounds — important if you're also simulating federated rounds on top of RL training), and both already have a DTDE baseline algorithm (CoPO / SAPO) you could adapt into a federated variant instead of starting from scratch.
- **Table 2's comparison schema (training scheme / simulator / max agents / action space / real-world test)** is a ready-made template for the "compare experimental setups across papers" cataloging task from reunion 2 — worth reusing those exact columns (plus a new "aggregation scheme" column) when you build your own comparison table across the federated-MARL papers you're collecting in `papers/federated_marl/`.

## Suggested next reads
1. CoPO paper in full (arXiv:2110.13827) — the LCF bi-level optimization is the most promising decentralized-with-a-learned-global-signal mechanism to compare against a true federated-averaging approach.
2. Qu et al. 2020 (scalable networked-agent actor-critic) and Egorov & Shpilman 2022 (MAMBA) — both direct scalability answers, both structurally amenable to a federated reframing.
3. Skim the MARL4AD GitHub repo (github.com/ispc-lab/MARL4AD) for continuously updated pointers, since this survey is actively maintained.
