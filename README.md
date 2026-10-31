# Distributionally Robust Portfolio Choice under Stochastic Volatility

A selective review and reproducible four-ETF computational study by
**Sheehan Mathur**. Established methods are studied without claiming a new DRO
algorithm.

- [Versioned software release v1.0.0](https://github.com/Sheehan007/Distributionally-Robust-Optimization/releases/tag/v1.0.0)
- [Paper PDF](preprint_study/paper/main.pdf)
- [Paper source](preprint_study/paper/main.tex)
- [Study code, requirements, and reproduction commands](preprint_study/README.md)
- [Provider-data access and redistribution policy](preprint_study/DATA_ACCESS.md)
- [Software citation](CITATION.cff)

The public record includes 168 trained neural checkpoints, 28 annual Heston
fits, aggregate results, canonical input hashes, and all manuscript sources.
Provider price files, raw responses, and date-by-date experimental caches are
excluded. Exact historical numerical replication requires appropriately
authorized access to matching inputs; current provider histories can differ.

The main matched DRO comparison uses 20,000 scenarios under three independent
seeds. Larger likelihood searches and two-significant-figure parameter tables
address numerical precision. The paper explicitly discusses gold concentration
in 2025 and retains statistically inconclusive return conclusions.

A preliminary single-asset analysis over 2019-2024 motivated the design; that
period is exploratory. Legacy exploratory modules in dro_study/ and the root
scripts remain as project history. The publication workflow is self-contained
in preprint_study/.

Software is [MIT licensed](LICENSE). The manuscript, author-created research
figures, and tables are [CC BY 4.0](preprint_study/paper/LICENSE). Third-party
provider data are excluded from both grants.
