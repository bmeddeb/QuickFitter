add a Material Strategy layer

| Pain‑point today                                                                                     | How a strategy helps                                                                                                |
| ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **Poor initial guesses** → slow or diverging optimisation                                            | Ship empirically good starting values, bounds, weighting rules, *per material family*.                              |
| **One‑size‑fits‑all residual scaling** causes bias between ε′ and ε″ for lossy vs. loss‑less samples | Strategy can override residual scaling / heteroscedastic weighting.                                                 |
| **Users must hand‑tune n‑poles / model choice**                                                      | Strategy can specify *model preference order* and hard/soft constraints (e.g. “never use Δε < 0 for dry ceramics”). |
| **Different noise characteristics** (e.g. VNA vs. TDR)                                               | Strategy can attach an uncertainty model or robust loss function.                                                   |
| **Meaningful QA rules differ** (e.g. tan δ thresholds)                                               | Strategy brings material‑specific evaluation thresholds and warning logic.                                          |

2 Conceptual architecture:
┌───────────────┐            ┌────────────────┐
│  UI / Front   │ ◀──JSON──▶ │  Fitting Core  │
└───────────────┘            │   (lmfit etc.) │
          ▲                  └────────┬───────┘
          │                           │
          │                 uses      ▼
          │                   ┌───────────────────────┐
          └─ user selects ──▶ │   MaterialStrategy    │
                              └───────────────────────┘
2.1 Core responsibilities of MaterialStrategy:
| Method                       | Purpose (override in subclasses)                                                         |
| ---------------------------- | ---------------------------------------------------------------------------------------- |
| `initial_parameters(data)`   | Return a **dict** of good starting values *and* reasonable bounds.                       |
| `preprocess(data)`           | Optional denoising / resampling (e.g. median filter for water‑based sloshing).           |
| `weighting(real, imag)`      | Produce weights or scaling factors; may depend on ε″ magnitude.                          |
| `preferred_model_order()`    | Ranked list: `["debye_multi", "sarkar", "drude"]…`.                                      |
| `n_poles_hint(data)`         | Suggest pole count or fix it.                                                            |
| `fit_evaluator_thresholds()` | Override FitEvaluator thresholds (RMS, χ²) for fair scoring.                             |
| `post_validate(result)`      | Material‑specific sanity checks (e.g. ε∞ ≥ 1, ∆ε ≥ 0 for non‑ferroelectric dielectrics). |
Keep the interface thin; new strategies only override what they need:
from abc import ABC, abstractmethod

class MaterialStrategy(ABC):
    name: str
    @abstractmethod
    def initial_parameters(self, f_ghz, eps): ...
    # optional hooks with default fall‑backs:
    def preprocess(self, f_ghz, eps): return f_ghz, eps
    def weighting(self, eps): return None       # means “use global default”
    def preferred_model_order(self): return ["debye_multi"]
    def n_poles_hint(self, f_ghz, eps): return "auto"
    def fit_evaluator_thresholds(self): return {}
    def post_validate(self, fit_result): return []

class LowLossLaminate(MaterialStrategy):
    name = "low_loss_laminate"
    def initial_parameters(...):
        # e.g. Rogers 5880 family heuristics
        return {"eps_inf": 2.0, "delta_eps": 0.1, ...}
    def weighting(self, eps):
        # emphasise accuracy on ε′, down‑weight ε″ if <1e‑3
        w_real = np.ones_like(eps)
        w_imag = 0.3 * np.ones_like(eps)
        return w_real, w_imag

class PureWater(MaterialStrategy):
    name = "pure_water_25C"
    def initial_parameters(...): ...
    def preferred_model_order(self): return ["debye_single", "sarkar"]
    def n_poles_hint(...): return 1
    def fit_evaluator_thresholds(self):
        return {"rms_imag": (0.005, 0.01, 0.02, "Add Cole‑Cole α term")}
        keep these strategies as:

        Hard‑coded subclasses (fastest).

        External YAML/JSON registry – so power‑users can author recipes without touching code.

        Plug‑in system (entry_points) for third‑party additions.

3 Algorithmic integration points
3.1 In the optimiser wrapper (run_analysis):
strategy = get_strategy(user_choice)           # default = GenericMaterial
f_ghz, eps = strategy.preprocess(f_ghz, eps)

init_params   = strategy.initial_parameters(f_ghz, eps)
n_poles_hint  = strategy.n_poles_hint(f_ghz, eps)
weights       = strategy.weighting(eps)

model_list = strategy.preferred_model_order()
for model_name in model_list:
    result = fit_model(model_name, init_params, weights, ...)
    warnings = strategy.post_validate(result)
    if result.success and not warnings:
        break                                   # accept first “good” model
3.2 Fit evaluation thresholds:
Provide a simple merge:
base_thresholds = FitEvaluator.DEFAULT_THRESHOLDS
overrides = strategy.fit_evaluator_thresholds()
evaluator = FitEvaluator(thresholds={**base_thresholds, **overrides})
3.3 UI wiring:
Dropdown or autocomplete + rich cards:
"Select material family" → shows dielectric constant range, typical loss, notes.

On change, JS serialises strategy_name along with CSV and sends to Python via run_analysis(csv, strategy, method).

After fit, show strategy‑specific warnings or suggestions returned by post_validate.

4 Populating the strategy catalogue:
| Family                                                     | Key heuristics                                                           | Typical bounds                                                |
| ---------------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------- |
| **Low‑loss RF laminates** (PTFE‑glass, Rogers RO, Taconic) | ε∞ ≈ 1.9‑2.4, Δε < 0.3, ω₁ in MHz, ω₂ few GHz, very low tan δ            | `delta_eps > 0, < 0.5`                                        |
| **FR‑4**                                                   | ε∞ ≈ 3.8, broad loss plateau; tan δ \~ 0.02 at 1 GHz                     | allow Δε negative *or* positive to catch dispersion inversion |
| **Water / aqueous**                                        | Single Debye pole dominates, τ around 8 ps (≈ 20 GHz); enormous ε₀ \~ 78 | `n_poles=1`, weight ε″ strongly                               |
| **Ceramics (BaTiO₃)**                                      | Very high ε, multiple relaxations, possible ferroelectric anomalies      | start with Δε big, allow negative lobes                       |
| **Conductive coatings / CNT**                              | Drude‑like: add σ term, maybe switch to Extended Lorentz model           | strategy signals alternative model class                      |
Source this table from published databases (IPC‑TM‑650, IT’RS, NIST water permittivity etc.).
5 “Smart default” fallback
If the user leaves Material = “Auto”:

Quick fingerprint the data:

median ε′ level → rough family (low, mid, high‑k).

slope sign of ε′ vs. f → positive dispersion indicates Δε < 0.

peak of ε″ / ε′ → locate relaxation.

Pick closest strategy.
(Simple heuristic tree now; later a tiny classifier.)

6 Risks & mitigation:
| Risk                                                               | Mitigation                                                                                                                                      |
| ------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **Strategy mismatch** – user picks wrong material → misleading fit | Surface a “material consistency score” post‑fit; if poor, suggest different strategies.                                                         |
| **Explosion of subclasses**                                        | Allow composition: Strategy = Base + Mix‑in decorators (NoiseModelMixin, HighFreqMixin, etc.).                                                  |
| **Overconstrained bounds hinder valid solutions**                  | Keep *soft* bounds (e.g. lmfit’s `vary` or larger ranges) and provide a *“Relax bounds”* toggle in UI.                                          |
| **Maintenance burden**                                             | Store strategies in external YAML with checksum versioning; testing harness loads each, fits reference synthetic data, CI fails on regressions. |
7 Next steps
Pick 3 anchor strategies (Low‑loss laminate, Water, Generic‑High‑k) – cover 80 % of current use‑cases.

Create a material_registry.py that maps name → StrategyClass and a very small factory.

Expose dropdown in UI; Auto option runs the heuristic chooser.

Write unit tests with your synthetic data sets to ensure each strategy’s initial_parameters leads to convergence ≤ 5 iterations in ≥ 95 % of cases.

After baseline is stable, discuss:

user‑editable YAML recipes,

Bayesian hyper‑prior learning (update recommended bounds with user fits),

community contribution pipeline (small JSON + reference data).

8 Summary
Strategy pattern cleanly decouples material knowledge from numerical engine.

Begin with a thin interface; override only what differs.

Offer user selection + auto detection; always allow manual override.

Keep strategies data‑driven – make them easy to update without touching core code.

Bake in validation & warnings to avoid silent misuse.
