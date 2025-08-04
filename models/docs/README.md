# Dielectric Models and KK Validation

This directory collects documentation snippets for the available dielectric
models in *QuickFitter* and demonstrates how to run the Kramers–Kronig
causality check on a fitted model.

## Example

```python
import numpy as np
from models.havriliak_negami import HavriliakNegamiModel
from models.kk_validator import KramersKronigValidator

f = np.geomspace(1, 110, 601)  # GHz
true_p = dict(eps_inf=2.2, delta_eps=3.1, tau=8e-11, alpha=0.9, beta=0.85)
model = HavriliakNegamiModel()
params0 = model.guess(true_p, f_ghz=f)
result = model.fit(true_p, params0, f_ghz=f)

validator = KramersKronigValidator.from_model(
    model,
    result,
    f_ghz=f,
    loss_repr="eps_imag",
    method="auto",
)
validator.validate()
print(validator.get_report())
```

The validator accepts either measured data passed directly to the constructor or
model-generated data via :py:meth:`KramersKronigValidator.from_model`. The
``loss_repr`` argument specifies whether the provided loss column is the
imaginary permittivity (``"eps_imag"``) or tangent-delta (``"tan_delta"``).
