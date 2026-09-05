# toxsim

`toxsim` generates context-aware synthetic poisoned-patient presentations. It
ports the latent-severity behavior from INTOXICATE: intoxication class affects
severity, which in turn influences GCS, respiratory failure, dysrhythmia, heart
rate, and systolic blood pressure.

## Installation

```shell
python -m pip install toxsim
# Include optional progress and rich CLI output:
python -m pip install "toxsim[cli]"
```

## Model data

The package includes the canonical clinical model configuration as package
data, including `predictive_variables.yml`, each variable's score file, and
`model.schema.yml`. `load_model_data()` uses these resources by default and
never depends on the current working directory.

```text
toxsim/data/
  model.schema.yml
  model/
  predictive_variables.yml
  <variable-name>_score.yml
```

Pass a directory to `load_model_data()` or `toxsim-generate
--model-data-dir` to override the bundled model with a compatible
configuration. An override directory contains `predictive_variables.yml` and
one `<variable-name>_score.yml` file for every predictive variable.

## Python API

```python
from toxsim import create_patients, load_model_data

model = load_model_data()
patients = create_patients(model, count=100, seed=2026)
```

`create_patient(model, seed=...)` creates one patient. A seed produces the same
patient data across runs, including the generated patient ID. Each presentation
entry has `name`, `value`, and `score`; continuous variables also include the
model range and whether its underlying value was within that range. The
patient-level `risk` equals the sum of presentation scores.

## CLI

```shell
toxsim-generate \
  --count 1000 \
  --destination ./generated \
  --format both \
  --seed 2026
```

`--format` accepts `json`, `jsonl`, or `both` (the default). JSON output is
written as `patients.json`; JSONL output is written as `patients.jsonl`. Add
`--model-data-dir /path/to/model-data` to use a compatible override.
