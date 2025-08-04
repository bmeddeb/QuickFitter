add a Material Strategy layer

| Pain‑point today                                                                                     | How a strategy helps                                                                                                |
| ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Poor initial guesses** → slow or diverging optimisation                                            | Ship empirically good starting values, bounds, weighting rules, *per material family*.                              |
| **One‑size‑fits‑all residual scaling** causes bias between ε′ and ε″ for lossy vs. loss‑less samples | Strategy can override residual scaling / heteroscedastic weighting.                                                 |
| **Users must hand‑tune n‑poles / model choice**                                                      | Strategy can specify *model preference order* and hard/soft constraints (e.g. “never use Δε < 0 for dry ceramics”). |
| **Different noise characteristics** (e.g. VNA vs. TDR)                                               | Strategy can attach an uncertainty model or robust loss function.                                                   |
| **Meaningful QA rules differ** (e.g. tan δ thresholds)                                               | Strategy brings material‑specific evaluation thresholds and warning logic.                                          |
