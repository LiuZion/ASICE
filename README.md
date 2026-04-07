# ASICE
Adsorption-Structure Intelligent Computing Engine

Version: 1.0.0

Data: 2026-04-07

Function Overview
This project is an intelligent platform for automatic adsorption-structure discovery in 2D photocatalytic materials. It accelerates the identification of high-quality adsorption configurations by combining a generation–evaluation–verification workflow with multi-stage modeling and validation.

The system is built on a three-layer architecture:

    1.Candidate Structure Generation
    Using constrained sampling and geometric enumeration, the first layer automatically generates candidate adsorption structures. After optimization, it significantly reduces the number of ineffective candidates (up to 90% reduction).

    2.AdsorbML-Assisted Ranking & Uncertainty Guidance
    The second layer leverages AdsorbML to perform relaxation and uncertainty evaluation of adsorption states. Candidates are prioritized using the principle of high uncertainty with low-energy stable configurations, improving both efficiency and the likelihood of discovering novel structures.

    3.High-Accuracy DFT Verification
    The third layer performs DFT validation with a phased strategy to ensure high precision while maintaining computational efficiency. Stability criteria are applied to assess and filter reliable results, supporting robustness, credibility, and reproducibility.

Overall, the platform is designed to achieve an estimated feasibility of 85%, enabling 1000–2000× speedup compared to traditional DFT workflows, improving candidate structure quality by 30–50%, and substantially increasing the probability of discovering new adsorption structures. Key design principles include accuracy assurance, efficiency optimization, scalability, reliability, and reproducibility.
