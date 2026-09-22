# AutoJev-27B

Built with autonomous agents, from research and data generation to training, evaluation, and deployment. The user set the goals and refined the scope; agents executed the work.

AutoJev-27B is a multimodal decision model based on Qwen3.8-27B. It returns probabilities over supplied choices in one forward pass per question, with a TypeSafe-compatible API and browser playground.

[Code](https://github.com/denis-pplx/autojev) · [Model weights](https://huggingface.co/denis-pplx/autojev-27b)

## Results

![Benchmark accuracy: Qwen3.8-27B, AutoJev-27B and Jev](assets/selected-accuracy.png)

| Model | Overall accuracy ↑ | ECE ↓ | Brier ↓ |
|---|---:|---:|---:|
| Qwen3.8-27B | 69.83% | 0.06483 | 0.40834 |
| **AutoJev-27B** | **84.60%** | **0.04282** | **0.22027** |
| Jev | 82.79% | 0.05274 | 0.25400 |

## Training

**One H200 · full-weight SFT · 73,000 unique training examples · 286 updates.** The released model is checkpoint 200. Training uses cross-entropy; calibration fits a scalar temperature separately.

![Training loss](assets/training-progress.png)

Run `bash configs/train.sh --help` for training arguments. The exact curated training corpus is not bundled.

## Run

Python 3.12+, [uv](https://docs.astral.sh/uv/) and a GPU with space for approximately 49 GiB of BF16 weights plus runtime overhead.

```bash
git clone https://github.com/denis-pplx/autojev.git
cd autojev
uv sync --frozen --python 3.12
uv run hf download denis-pplx/autojev-27b --local-dir checkpoints/selected
AUTOJEV_CHECKPOINT=checkpoints/selected uv run autojev-serve
```

Open **http://localhost:8000** for the playground or `/docs` for the API. `POST /v1/systemone` supports `choice`, `noul`, `score`, and optional base64 `images`. Set `AUTOJEV_API_KEY` to enable authentication. While the weights are private, authenticate with `uv run hf auth login` before downloading.

Use the included `DecisionModel` loader or server. Published benchmarks measure text decisions; image support is not a natural-image accuracy claim.

[Code: MIT](LICENSE) · [Weights: Apache 2.0](https://huggingface.co/denis-pplx/autojev-27b/blob/main/LICENSE) · Independent implementation inspired by Jev.
