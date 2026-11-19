# Project collaboration requirements

- Work in this repository. Do not make another checkout or project copy unless the user requests it.
- Preserve the user's data scope, source, scientific task, split rules and training budget. A slow download or limited resources do not authorize a smaller or selectively downloaded source.
- For the current PDEBench experiment, obtain the complete official HDF5 and verify its published MD5 before building QA. Never present a selected-frame cache, cropped subset or unfinished download as the complete source.
- Explain proposed dataset or protocol substitutions, their compatibility requirements and scientific consequences before applying them. Suggestions alone do not authorize changing the experiment.
- Distinguish the complete source dataset from the crops and QA sampled for an experiment, and distinguish smoke/pilot checks from completed training.
- Report only checks actually performed, including skips, failures and platform limitations. Local tiny-model tests do not establish remote Qwen 14B readiness, GPU memory use, speed or accuracy.
- The remote Linux server is operated by the user through pasted commands unless a verified connection is available. Do not claim to have run server commands or completed its downloads.
- When correcting a mistake, audit related code and instructions, preserve unrelated work, and clearly state the correction and the remaining uncertainty.
