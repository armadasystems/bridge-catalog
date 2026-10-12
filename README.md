# Catalog of applications for Kubernetes®

This repository contains the configuration and manifests for deploying and managing applications for Kubernetes.

---

## Overview

This repository serves as a central hub for managing and deploying applications for Kubernetes
---

## Applications

- [KAI Scheduler](kai-scheduler/README.md)



## Catalog snapshot (Pulse dashboard)

Every merge to `staging` runs `.github/workflows/catalog-snapshot.yml`, which exports all
models (`models/v1`) and apps (`apps/v1`) to a GitHub Actions artifact named
`bridge-catalog-snapshot` containing one Parquet file, `bridge_catalog_snapshot.parquet`: one row
per model or app (`entity_type` = `Model` / `App`), with the snapshot info (`snapshot_id`,
`commit_sha`, `branch`, `generated_at`, `schema_version`) on every row. Each file is a full
snapshot. The data team loads it into StarRocks for the Pulse/Omni dashboard. Dates come from
git automatically:
`start_date` is when the YAML was first committed (on any branch), `merged_date` is when it was
merged into `staging`.

Optional `tracking` block, for the dashboard only (Bridge ignores it):

```yaml
tracking:
  status: live          # in-progress | live | deprecated
  liveDate: 2026-09-10  # date it went live
  note: "needs ray-llm w/ vLLM 0.28.0"
  # rarely needed overrides
  source: "Mistral AI"  # models: Source column
  partner: "SecurIn"    # apps: Partner column (default: displayName)
  type: "API"           # apps: Type column (default: App)
  appName: "Securin Model Security Scan"  # apps: App Name column (default: displayName)
```

Run locally: `pip install -r scripts/requirements.txt && python scripts/export_catalog.py --out out`.
View the result as a table:
`python -c "import pyarrow.parquet as pq; print(pq.read_table('out/bridge_catalog_snapshot.parquet').to_pandas().to_string())"`
(needs `pip install pandas`), or open it in any Parquet viewer.
