# SpatialUAV: Benchmarking Spatial Intelligence for Low-Altitude UAV Perception, Collaboration, and Motion

[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-yellow)](<https://huggingface.co/datasets/Hyu-Zhang/SpatialUAV>)
[![Paper](https://img.shields.io/badge/arXiv-2606.27876-b31b1b.svg)](https://arxiv.org/abs/2606.27876)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**SpatialUAV** is a benchmark for evaluating spatial intelligence in real
low-altitude UAV scenarios. It covers perception, spatial relation reasoning,
aerial-aerial collaboration, aerial-ground collaboration, and UAV motion
understanding under a unified visual-question-answer format.

<p align="center">
  <img src="assets/example_vis.png" width="100%" alt="SpatialUAV examples">
</p>

## Highlights

- **4,331 curated instances** from real low-altitude UAV images, videos, and metadata.
- **14 task types** across semantic discrimination, spatial relations,
  aerial-aerial collaboration, aerial-ground collaboration, and motion understanding.
- **7 visual input configurations** covering single images, paired views,
  candidate-view selection, annotated images, and ordered video frames.
- **9 answer formats**, including option labels, region IDs, region pairs,
  bounding boxes, angle-distance values, movement directions, and free-form text.
- **Task-specific evaluation** for heterogeneous outputs instead of relying on
  one generic text metric.

## Benchmark Overview

<p align="center">
  <img src="assets/data_pipeline.png" width="100%" alt="SpatialUAV data pipeline">
</p>

SpatialUAV standardizes every sample into a visual-input, question, and answer
record. Its construction pipeline combines detector-assisted regions, depth
supervision, metadata-derived rules, manual annotation, blind filtering, and
multi-round validation.

| Group | Instances | Task Types | Main Capability |
| --- | ---: | --- | --- |
| Semantic Discrimination | 599 | Region Recognition, Anomaly Detection | Recognize queried objects and safety-critical regions |
| Spatial Relation | 716 | Direction Recognition, Distance Comparison | Infer direction and relative depth from UAV views |
| Aerial-Aerial Collaboration | 1,231 | Collaboration Recognition, Shared Association, Object Matching, Camera Transformation, Occlusion Removal | Match and reason across multiple UAV viewpoints |
| Aerial-Ground Collaboration | 785 | Shared Association, Collaboration Recognition, View Translation, Path Planning | Align aerial and ground observations |
| Motion Understanding | 1,000 | Global Motion | Describe UAV/camera motion over ordered frames |

<p align="center">
  <img src="assets/task_distribution.png" width="55%" alt="Task distribution">
</p>

<p align="center">
  <img src="assets/answer_formats.png" width="90%" alt="Answer format distribution">
</p>

## Data Format

The annotation file is JSONL. Each line is one benchmark instance:

```json
{
  "id": "Region_Recognition_00001",
  "image": ["./SpatialUAV/samples_Single_Image/img0001.jpg"],
  "conversations": [
    {
      "from": "human",
      "value": "Which regions in the image contain a parking lot? Answer with only the region labels, formatted exactly like `Region 1, 2`. No explanation."
    }
  ],
  "source": "SpatialUAV",
  "GT": "Region 3, 4."
}
```

Expected dataset layout after download:

```text
SpatialUAV/
  annotations.jsonl
  annotations_subset_20pct_per_task.jsonl
  samples_Single_Image/
  samples_A2A_Pured/
  samples_A2A_detected/
  samples_A2A_Occlusion_Removal/
  samples_A2G_Pured/
  samples_A2G_detected/
  samples_A2G_Path_Planning/
  samples_Motion_Understanding_Frames/
```

Use `annotations_subset_20pct_per_task.jsonl` for quick checks and
`annotations.jsonl` for full benchmark evaluation.

## Installation

Clone the code and install the common dependencies:

```bash
git clone https://github.com/Hyu-Zhang/SpatialUAV.git
cd SpatialUAV
pip install -r requirements.txt
```

Download the benchmark with Git LFS into the layout expected by the scripts:

```bash
git lfs install
git clone https://huggingface.co/datasets/Hyu-Zhang/SpatialUAV SpatialUAV
```

The full image/frame archive is distributed through Hugging Face rather than
stored in this GitHub repository. Do not commit the local `SpatialUAV/` data
directory; it is excluded by `.gitignore`.

Local model backends require their own model environments, for example
`torch`, `transformers`, `accelerate`, `torchvision`, and `qwen-vl-utils`.
Install the dependencies recommended by the corresponding model repository.

## Inference

All model families are launched through one script:

```bash
python run_spatialuav_inference.py \
  --backend qwen \
  --model /path/to/model \
  --annotations /path/to/SpatialUAV/annotations.jsonl \
  --output predictions/predictions_qwen.jsonl
```

Supported backends:

| Backend | Usage |
| --- | --- |
| `autodl` | OpenAI, Anthropic, or Gemini compatible API inference |
| `cambrian` | Local Cambrian-S style checkpoints |
| `internvl35` | Local InternVL3.5 checkpoints |
| `qwen` | Local Qwen text or vision-language checkpoints |
| `spatialvlm` | Qwen2.5-VL compatible SpatialVLM-style checkpoints |
| `vst` | Local VST-7B-SFT checkpoints |

Common options:

```bash
--task-prefix Region_Recognition A2A_Object_Matching
--limit 100
--offset 0
--image-limit 8
--resume
--continue-on-error
--max-tokens 512
--temperature 0
```

Examples:

```bash
python run_spatialuav_inference.py --backend cambrian --model /path/to/Cambrian-S-7B
python run_spatialuav_inference.py --backend internvl35 --model /path/to/InternVL3_5-8B
python run_spatialuav_inference.py --backend vst --model /path/to/VST-7B-SFT --no-motion-as-video
python run_spatialuav_inference.py --backend autodl --provider openai --model <served-model-name>
```

## Evaluation

Set the API key for the GPT-5.4-mini semantic judge and evaluate predictions
with the same metric configuration used in the paper:

```bash
export AUTODL_API_KEY=<your-api-key>

python eval_spatialuav.py \
  --predictions predictions/predictions_qwen.jsonl \
  --annotations /path/to/SpatialUAV/annotations.jsonl \
  --motion-judge-mode gpt_54_mini \
  --motion-judge-provider openai \
  --motion-judge-model gpt-5.4-mini \
  --output-json results/summary_qwen.json
```

The evaluator uses task-specific metrics:

| Output Type | Evaluation |
| --- | --- |
| Region IDs | set matching / partial-region score |
| Option labels | exact option accuracy |
| Region pairs | pair-level precision, recall, and F1 |
| Bounding boxes | IoU and center-aware geometry score |
| Camera transformation | angle and distance error thresholds |
| Path planning direction | exact direction accuracy |
| Motion description | GPT-5.4-mini semantic similarity judge |

Global Motion predictions are scored for semantic similarity against the
reference descriptions by GPT-5.4-mini, producing a normalized score in
`[0, 1]`. Judge responses are cached in `eval_cache/` so interrupted evaluations
can resume without rescoring completed samples. Use `--motion-judge-base-url`
when evaluating through a different OpenAI-compatible endpoint.

## Results

<p align="center">
  <img src="assets/reasoning_group_scores.png" width="100%" alt="Reasoning group scores">
</p>

SpatialUAV is designed as a diagnostic benchmark. Current VLMs show stronger
performance on recognition-style tasks, while cross-view association,
structured grounding, geometric transformation, and temporal UAV motion remain
substantially harder.

## Paper

**SpatialUAV: Benchmarking Spatial Intelligence for Low-Altitude UAV
Perception, Collaboration, and Motion** ([arXiv:2606.27876](https://arxiv.org/abs/2606.27876))

```bibtex
@article{zhang2026spatialuav,
  title   = {SpatialUAV: Benchmarking Spatial Intelligence for Low-Altitude UAV Perception, Collaboration, and Motion},
  author  = {Zhang, Haoyu and Liu, Meng and Xiang, Qianlong and Wang, Kun and Wang, Yaowei and Nie, Liqiang},
  journal = {arXiv preprint arXiv:2606.27876},
  year    = {2026}
}
```

## License

The code in this repository is released under the [MIT License](LICENSE).
The benchmark data is derived from multiple source datasets and remains subject
to the terms of those sources; the MIT License does not relicense the data.
