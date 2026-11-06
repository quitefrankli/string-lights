# SAM model changes for hand masking

Researched 2026-11-01 against Meta's repositories and model card.

## What changes

| Version | Main change | Relevance here |
| --- | --- | --- |
| SAM 2.1 | Improved SAM 2 checkpoints; the SAM 2 family already supports visual prompts and video segmentation. | We currently use its image model independently on every frame, with GroundingDINO supplying hand boxes. Video tracking would be a pipeline change even without upgrading models. |
| SAM 3 | Adds concept segmentation: text or image exemplars identify all matching objects. Combines a detector and tracker sharing a vision encoder. | A prompt such as `hands` could replace our separate GroundingDINO stage; video inference could maintain masks across frames. |
| SAM 3.1 | Adds Object Multiplex and optimized inference, grouping tracked objects into shared-memory buckets. | Primarily a multi-object video efficiency update, with some accuracy improvements; its headline speedup is not a prediction for our two hands. |

Sources: [SAM 2 release notes](https://github.com/facebookresearch/sam2/blob/main/RELEASE_NOTES.md), [SAM 3 architecture and usage](https://github.com/facebookresearch/sam3), [SAM 3.1 release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md). Current implementation: `string_lights/masking.py`, `string_lights/config.py`, and `string_lights/pipeline.py`.

## How much better is 3.1?

Meta reports roughly **7× faster inference at 128 objects on one H100**, compared with the original SAM 3 release. This does not establish the speedup for two hands on our RTX 5070 Ti. Text-prompt video benchmark results are mixed: some improve and some decline. Visual-prompt video object segmentation improves on six of seven reported benchmarks, including MOSEv2 from 60.3 to 62.3. These are general benchmarks, not a guarantee about finger boundaries in guitar footage. [Release measurements](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)

## Migration requirements

The official SAM 3 installation requires Python ≥3.12, PyTorch ≥2.7, and a CUDA GPU with CUDA ≥12.6; 3.1 requires updated repository code. Our Python requirement and locked PyTorch 2.11 satisfy the version floors, but CUDA compatibility and memory use still need checking. [Installation](https://github.com/facebookresearch/sam3#installation)

The SAM 3.1 checkpoint is gated: Hugging Face account access/conditions and authentication are required. Its model card explicitly says there is **no Hugging Face Transformers integration** for these checkpoints. Our current `Sam2Model`/`Sam2Processor` path therefore cannot be upgraded by substituting a model ID; use Meta's video predictor API. [Official model card](https://huggingface.co/facebook/sam3.1)

## Recommendation

Keep the existing SAM 2.1 cache as a baseline. Evaluate SAM 3.1 video inference on the same seven seconds and retain the existing binary mask cache format. Compare missed fingertips, gaps between fingers, inclusion of guitar pixels, temporal flicker, runtime, and peak GPU memory. Reduced flicker is a plausible benefit of temporal tracking, not a measured result for this clip. The sources reviewed do not establish hand-specific superiority.
