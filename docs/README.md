# HFlow documentation

HFlow turns landed physical-AI recordings into canonical, quality-checked,
queryable datasets. Start with the path that matches what you are trying to do.

```text
MCAP recordings -> transform -> quality checks -> enrichments
                                        |
                                        v
canonical MCAP + provenance + Parquet catalog -> curated manifest
```

The documentation follows the [Diátaxis](https://diataxis.fr/) model: tutorials
teach through a complete experience, how-to guides solve a specific task,
reference pages define exact contracts, and explanation pages describe design
and tradeoffs.

## Start here

- **Run a pipeline locally:** follow the
  [five-minute quickstart](../examples/README.md#five-minute-quickstart).
- **Understand the project first:** read the [architecture](./ARCHITECTURE.md),
  especially [what is implemented](./ARCHITECTURE.md#implementation-status).
- **Bring existing processing code:** use the [porting guide](./PORTING.md).
- **Schedule a working pipeline:** continue with the [runtime guide](./RUNTIME.md).
- **Find code you can run:** browse the [examples catalog](../examples/README.md).
- **Understand where HFlow fits:** see the [robotics data stack](./INTEGRATIONS.md).
- **Get a direct answer:** check the [frequently asked questions](./FAQ.md).

## Tutorials

Tutorials are complete learning paths. They make choices for you so you can see
the whole workflow before adapting it.

- [Five-minute local quickstart](../examples/README.md#five-minute-quickstart):
  synthesize a small episode and run transform, checks, and reporting without
  Docker.
- [Egocentric factory corpus](../examples/egocentric/README.md): prepare real
  head-mounted video, inject known defects, run locally or in Airflow, inspect
  the results, and cut a curated manifest.

## How-to guides

Use these when you already know the outcome you need.

- [Enable the built-in quality checks](./how-to/enable-built-in-checks.md)
- [Port existing processing and quality-check code](./PORTING.md)
- [Serve a workspace over HTTP](./SERVE.md)
- [Call an OpenAI vision endpoint from a step](./how-to/call-openai-vision.md)
- [Run and operate the local Airflow runtime](./RUNTIME.md)
- [Deploy into an existing Airflow environment](./RUNTIME.md#bring-your-own-airflow-hflow-deploy)
- [Query quality evidence and create a manifest](./CATALOG.md)
- [Export a portable dataset snapshot](./how-to/export-dataset-snapshot.md)
- [Find and reprocess stale episodes](./CATALOG.md#finding-stale-episodes-to-reprocess)
- [Load a large manifest with memory mapping](./how-to/load-manifest-mmap.md)
- [Inspect episodes in Foxglove](./how-to/inspect-episodes-in-foxglove.md)
- [Add a native-video provider](./PROVIDERS.md)
- [Develop from Windows with WSL2](./how-to/wsl2.md)

## Reference

Reference pages define stable inputs, outputs, configuration, and stored-data
contracts.

- [Canonical episode format](./FORMAT.md)
- [Catalog tables and curation API](./CATALOG.md)
- [Portable dataset snapshot](./how-to/export-dataset-snapshot.md#format-contract)
- [Runtime commands and configuration](./RUNTIME.md)
- [Native-video provider protocol](./PROVIDERS.md)

## Explanation

These pages explain why the system has its current boundaries.

- [Architecture and implementation status](./ARCHITECTURE.md)
- [Hosting HFlow: the data-plane contract for operating workspaces for others](./HOSTING.md)
- [Benchmark methodology and scale limits](./BENCHMARKS.md)
- [Why steps use ordinary Python and small adapters](./PORTING.md#the-pattern)
- [How HFlow fits with MCAP, Airflow, visualization tools, and training formats](./INTEGRATIONS.md)
- [Frequently asked questions](./FAQ.md)

## Documentation contract

Every project-owned page is linked from this index. Every runnable example is
linked from the [examples catalog](../examples/README.md), and focused how-to
guides link to the code that implements their task. New pages must state one
primary user goal instead of mixing a tutorial, operational recipe, API listing,
and design essay into one narrative.
