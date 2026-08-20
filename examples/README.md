# HFlow runnable examples

Examples are executable documentation. Each one names its prerequisites, gives
one command to run, and says what successful output looks like.

## Five-minute quickstart

**Use it for:** seeing the complete in-process lifecycle on a small multimodal
episode.

**Prerequisites:** the normal development environment from the repository root;
no Docker, scheduler, network service, or API key.

```bash
uv sync --locked --all-extras
uv run python examples/quickstart.py
```

The example synthesizes `data/sample/episode_0001.mcap`, writes a canonical
episode under `data/test-runs/`, and prints the measurements from timestamp,
motion, and camera-health checks. Pass an existing MCAP path as the first
argument to process your own recording.

Code: [`quickstart.py`](./quickstart.py)

## OpenAI vision check

**Use it for:** sampling an episode, making a contact sheet, and calling an
OpenAI vision model from an HFlow check.

**Prerequisites:** an OpenAI API key and the optional `openai` dependency. This
example sends image data to the configured endpoint and may incur API charges.

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-5.4-mini"
uv run --extra openai python examples/openai_vision/pipeline.py
```

The example synthesizes a small episode when no MCAP path is provided and runs
two contact-sheet checks through the Responses API: an activity description,
and a hand-visibility fraction (Dyna's "missing/occluded hand positions"
quality issue, answered as model evidence). Each check is one API call;
`OPENAI_BASE_URL` can point the same client at another endpoint that
implements the Responses API.

Guide: [Call an OpenAI vision endpoint from a step](../docs/how-to/call-openai-vision.md)  
Code: [`openai_vision/pipeline.py`](./openai_vision/pipeline.py)

## Egocentric factory corpus

**Use it for:** a complete corpus workflow on real data, covering Hugging Face
download, declared fault injection, local processing, Airflow scheduling,
inspection with Foxglove and DuckDB, and SQL curation.

**Prerequisites:** accepted Hugging Face dataset terms, `hf` authentication,
network access, and about 2.5 GB of disk (a ~1 GB pinned download plus the
generated corpus). Docker is needed only for the Airflow path.

```bash
uv run python examples/egocentric/prepare.py
uv run python examples/egocentric/pipeline.py data/egocentric/landing/*.mcap
```

The preparation step writes 96 deterministic episodes, six with camera faults
declared in the tracked manifest (and editable there). The pipeline accepts
the 90 healthy episodes, quarantines the six faulty ones, and records
contact-sheet artifacts for the accepted set.

Guide and expected artifacts: [`egocentric/README.md`](./egocentric/README.md)

## Example requirements

Keep examples small enough to read, but complete enough to execute from the
repository root. A new example must include:

- prerequisites and external side effects, including network calls or cost;
- an exact command that works in the locked uv environment;
- the observable files or terminal result produced on success; and
- a link from this catalog and from the relevant how-to guide.
