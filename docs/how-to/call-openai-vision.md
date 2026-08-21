# Call an OpenAI vision endpoint from a step

Use an ordinary OpenAI client inside an HFlow check when a visual judgment
cannot be expressed as a deterministic local measurement. HFlow owns frame
extraction, scheduling, and result recording; your step owns the prompt, model,
credentials, sampling policy, and interpretation of the answer.

The complete example is
[`examples/openai_vision/pipeline.py`](../../examples/openai_vision/pipeline.py).
It follows the official OpenAI
[images and vision guide](https://developers.openai.com/api/docs/guides/images-vision):
the image is sent to the Responses API as a base64 data URL and the text result
is read from `response.output_text`. The example ships two such checks: the
activity description walked through below, and a hand-visibility check
(Dyna's "missing/occluded hand positions" issue) that turns the model's tile
count into a `hands_visible_fraction` measurement.

## 1. Install the optional client

From the repository root:

```bash
uv sync --locked --extra openai
```

The OpenAI SDK is an example-only dependency. It is not installed with the
HFlow core package because pipeline authors may use a different client or no
model endpoint at all.

## 2. Configure credentials and the model

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-5.4-mini"
```

The example defaults `OPENAI_BASE_URL` to `https://api.openai.com/v1`. Set it
explicitly only when the target implements the Responses API:

```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

Do not put API keys in a pipeline file, runtime bundle, or committed `.env`
file. The local process and Airflow worker should receive them through their
secret-management environment.

## 3. Declare and use the endpoint

The App maps a stable capability name to the configured URL:

```python
app = hflow.App(
    "openai-vision-example",
    data_root="./data/openai-vision",
    endpoints={"vision": OPENAI_BASE_URL},
)
```

The check declares `uses="vision"`. Preflight can then reject a missing alias
before processing any episode. At run start an exported
`HFLOW_ENDPOINT_VISION` variable overrides (or supplies) the alias, so the
same file can point at a different endpoint without editing it. Inside the
step, the endpoint remains an ordinary OpenAI SDK option:

```python
@app.check(uses="vision")
def describe_activity(episode: hflow.Episode) -> hflow.CheckResult:
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=app.endpoints["vision"],
    )
    response = client.responses.create(
        model=os.environ["OPENAI_MODEL"],
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": ACTIVITY_PROMPT},
                    {"type": "input_image", "image_url": contact_sheet_data_url},
                ],
            }
        ],
    )
    return hflow.CheckResult(measurements={"activity_description": response.output_text})
```

## 4. Run it

```bash
uv run --extra openai python examples/openai_vision/pipeline.py
```

Pass an MCAP path to use your own episode. The example samples frames at 0.5
FPS, caps the request at 12 tiles, and makes one model request per episode. Those
choices are part of the measurement definition: change them deliberately and
version the step explicitly when external model configuration cannot be hashed
from the function.

## Keep model output as evidence

Model output is probabilistic. Record the answer, model name, and sampling
parameters as measurements or labels; avoid making a model response a critical
gate until its error modes and thresholds are validated on your own data. The
[porting guide](../PORTING.md#dialect-3-jpeg-frames-and-your-own-vlm-client)
shows the lower-level per-frame pattern and an OpenAI-compatible Chat
Completions client for locally hosted models.
