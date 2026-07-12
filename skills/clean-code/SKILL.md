---
name: clean-code
description: Apply the clean-code Python readability style when editing this repository, especially after removing type hints, reorganizing training modules, cleaning dense numerical/vectorized code, fixing cramped blank-line layout, or fixing ugly Black line breaks. Use for requests about making code less cramped, keeping short calls on one line, splitting genuinely long calls vertically, introducing local helpers for repeated long argument groups, removing top-of-file module docstrings, preserving __all__, and keeping training/simulation code readable without changing behavior.
---

# Clean Code

## Style Rules

Use this style when editing `fighter_rl/**/*.py`.

- Do not add top-of-file module docstrings such as `"""Shared utilities."""`.
- Keep public package exports explicit with `__all__` in every `__init__.py`.
- Do not use function parameter or return type annotations.
- Do not leave type-removal spacing artifacts such as `model                  ,` or `)                      :`.
- Keep short signatures on one line when they fit, for example `def detach_state(state):`.
- Use Google-style `Args:` docstrings only where documentation is genuinely useful.
- Do not compress multiple statements with semicolons.
- Preserve semantic state while cleaning style. Do not remove identifiers such
  as `bucket_ids`, recurrent state, masks, report fields, or experiment metadata
  just to shorten code.

## Call Layout Rules

Prefer one-line calls when the complete call fits cleanly under Black's line
length. Do not leave ugly two- or three-line calls where only the closing
parenthesis moved.

Prefer this:

```python
own_alt = bucket_uniform("altitude_m", 7000.0)
ata_abs = bucket_abs("ata_deg", 0.0, axis="ata")
own_speed = apply_bucket_dv(target_speed, own_speed)
```

Avoid this:

```python
own_speed = self._bucket_uniform(
    target_cfg, buckets, bucket_ids, "own_speed_mps", 285.0
)
```

When a block repeats the same long helper arguments, introduce a small local
helper that captures the repeated context and keeps call sites short.

Prefer this:

```python
def bucket_uniform(name, default):
    return self._bucket_uniform(target_cfg, buckets, bucket_ids, name, default)

def bucket_abs(name, default, axis=None):
    return self._bucket_abs(target_cfg, buckets, bucket_ids, name, default, axis=axis)

own_alt = bucket_uniform("altitude_m", 7000.0)
distance = bucket_abs("distance_m", 700.0, axis="distance")
```

Use vertical argument layout when a call is genuinely long and cannot be made
shorter without hiding important information.

Prefer this:

```python
self.init_bucket_require_feasible = self._bucket_bool(
    target_cfg,
    buckets,
    bucket_ids,
    "ensure_initial_feasible",
    True,
)
```

Do not force long calls into one line if Black will immediately split them.

## Blank Line Rules

Insert blank lines between logical groups, not mechanically after every line.

- Separate validation/guard clauses from the following main logic.
- Separate setup, sampling, calculation, mutation, and return blocks.
- In numerical code, group related variables:
  - fixed constants
  - reference values
  - mass or unit conversions
  - offsets/deltas
  - derived tensors
  - final state updates
- In dense math formulas, keep multiplication/division tight and use spaces
  around `+` and `-` to separate terms, for example
  `u*ct*cs + v*(sp*st*cs - cp*ss)`. Wrap those blocks with `# fmt: off`
  and `# fmt: on` so Black does not undo the formula layout.
- In training loops, separate:
  - batch extraction
  - early exits
  - normalization
  - loop setup
  - model forward pass
  - loss computation
  - finite checks
  - optimizer step
  - logging/checkpoint updates
- In validation/reporting code, separate execution blocks such as
  `with torch.no_grad()` or `env.step(...)` from `report[...]`, manifest,
  and log payload construction.
- In environment steps, separate:
  - action normalization
  - target/ownship simulation
  - geometry recomputation
  - reward components
  - done/valid masks
  - info dictionaries
- In curriculum sampling helpers, separate:
  - config extraction
  - probability/fraction clamping
  - random selector tensors
  - uniform/easy/boundary sample tensors
  - final `torch.where` selection

## Example

Prefer this:

```python
self.dry_weight = 17630.0
self.dry_cgx, self.dry_cgz = -194.868149247237, -5.03345098112748

self.tank_x, self.tank_y, self.tank_z = -174.4, 65.0, 5.0

full_fuel = 6971.963548061171
full_cgx, full_cgz = -189.06766929517926, -2.1900658483050157

dry_mass = self.dry_weight / 32.174
fuel_mass = full_fuel / 32.174

ddx = (full_cgx - self.dry_cgx) / 12
ddz = (full_cgz - self.dry_cgz) / 12

tx = (full_cgx - self.tank_x) / 12
ty = self.tank_y / 12
tz = (full_cgz - self.tank_z) / 12
```

Do not collapse those groups into one uninterrupted block.

## Validation

After style edits, run:

```bash
.venv/bin/python -m black --check fighter_rl
env PYTHONPYCACHEPREFIX=/tmp/fighter_rl_pycache .venv/bin/python -m py_compile $(find fighter_rl -name '*.py' -print)
```

For training-facing edits, also run the preflight config used in this repo when available.
