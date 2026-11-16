"""Prepare fully audited, immutable training arrays without loading LLM weights.

On Windows with conflicting OpenMP installations, set MKL_THREADING_LAYER=SEQUENTIAL
and OMP_NUM_THREADS=1 before starting Python. Do not use KMP_DUPLICATE_LIB_OK.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))

from scripts import train_point_readout as trainer
from tensor_compression.downstream.point_readout_data import load_config, resolve_path
from tensor_compression.downstream.readout_training_cache import (
    prepare_training_cache, training_token_identity,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/field_to_llm_multisource_v2.yaml"))
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), required=True)
    parser.add_argument("--model-dir")
    parser.add_argument("--qa-dir")
    parser.add_argument("--output-dir", required=True)
    cli = parser.parse_args()
    config = load_config(cli.config, cli.profile)
    args = trainer.model_args(config, cli.model_dir, checkpointing=False)
    trainer.core.apply_runtime_environment(args)
    tokenizer = trainer.core.load_tokenizer(args)
    identity = training_token_identity(tokenizer, text_encoding_version=trainer.TEXT_ENCODING_VERSION,
                                       system_message=trainer.SYSTEM_MESSAGE)
    result = prepare_training_cache(resolve_path(cli.qa_dir or config["data"]["qa_dir"]),
        resolve_path(cli.output_dir), tokenizer, config["training"],
        encode_record=trainer.encode_training_record, token_identity=identity,
        progress=lambda event: print(json.dumps(event, ensure_ascii=False), flush=True))
    print(json.dumps({"cache_dir": str(resolve_path(cli.output_dir)), "questions": result["questions"],
                      "source_audit": result["source_audit"], "token_report": result["token_report"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
