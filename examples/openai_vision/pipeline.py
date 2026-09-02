"""Call an OpenAI vision model from an hflow quality-check step.

Run from the repository root with:

    OPENAI_API_KEY=... OPENAI_MODEL=gpt-5.4-mini \
      uv run --extra openai python examples/openai_vision/pipeline.py [episode.mcap]
"""

import base64
import os
import sys
from pathlib import Path

from openai import OpenAI

import hflow

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
ACTIVITY_PROMPT = (
    "Describe the physical activity across this timestamped sequence in one concise sentence. "
    "State when the visual evidence is insufficient."
)
HAND_VISIBILITY_PROMPT = (
    "This image is a grid of {tile_count} frames sampled from an egocentric recording. "
    "In how many of the {tile_count} frames are the operator's hands clearly visible "
    "(not missing from view, not occluded by objects)? Answer with a single integer."
)

app = hflow.App(
    "openai-vision-example",
    data_root="./data/openai-vision",
)


def _image_data_url(image_path: Path) -> str:
    encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded_image}"


@app.check(requires=("vision-model",), version="responses-contact-sheet-v1")
def describe_activity(episode: hflow.Episode) -> hflow.CheckResult:
    contact_sheet = hflow.ffmpeg.contact_sheet(
        episode.frames(fps=0.5),
        episode.workdir / "openai-vision-contact-sheet.jpg",
        columns=4,
        max_tiles=12,
    )
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=OPENAI_BASE_URL,
    )
    model_name = os.environ["OPENAI_MODEL"]
    response = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": ACTIVITY_PROMPT},
                    {
                        "type": "input_image",
                        "image_url": _image_data_url(contact_sheet.path),
                    },
                ],
            }
        ],
    )
    return hflow.CheckResult(
        measurements={
            "activity_description": response.output_text,
            "vision_model": model_name,
            "sampled_frame_count": len(contact_sheet.tile_log_times_ns),
        }
    )


@app.check(requires=("vision-model",), version="hand-visibility-contact-sheet-v1")
def hand_visibility(episode: hflow.Episode) -> hflow.CheckResult:
    """Missing/occluded hand positions, a quality issue named in Dyna's article.

    Hand visibility is a model judgment, not a signal statistic, so it lives
    on the VLM extension surface: sample frames, ask the model, record the
    fraction as evidence. The keep/drop threshold stays a curation query.
    """
    contact_sheet = hflow.ffmpeg.contact_sheet(
        episode.frames(fps=0.5),
        episode.workdir / "hand-visibility-contact-sheet.jpg",
        columns=4,
        max_tiles=12,
    )
    tile_count = len(contact_sheet.tile_log_times_ns)
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=OPENAI_BASE_URL,
    )
    response = client.responses.create(
        model=os.environ["OPENAI_MODEL"],
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": HAND_VISIBILITY_PROMPT.format(tile_count=tile_count),
                    },
                    {
                        "type": "input_image",
                        "image_url": _image_data_url(contact_sheet.path),
                    },
                ],
            }
        ],
    )
    answer_text = response.output_text.strip()
    measurements: dict[str, hflow.MeasurementValue] = {
        "hands_visible_raw_answer": answer_text,
        "hands_sampled_frame_count": tile_count,
    }
    # An unparsable answer is an unknown, not zero visible hands: record the
    # raw text and a tag instead of inventing a fraction.
    if answer_text.isdigit():
        measurements["hands_visible_fraction"] = min(int(answer_text), tile_count) / tile_count
        return hflow.CheckResult(measurements=measurements)
    return hflow.CheckResult(measurements=measurements, tags=["hands_visible_unparsed"])


def main() -> None:
    if len(sys.argv) > 1:
        episode_path = Path(sys.argv[1])
    else:
        episode_path = Path("./data/openai-vision/sample.mcap")
        if not episode_path.is_file():
            print(f"synthesizing a sample episode at {episode_path} ...")
            hflow.testing.synthesize_episode(episode_path)

    app.test(episode_path)


if __name__ == "__main__":
    main()
