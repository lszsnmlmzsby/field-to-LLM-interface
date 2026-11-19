"""Command-line interface for field data, training, evaluation and inference."""
from importlib import import_module
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Each operation has one implementation; evaluation shares the training loader.
COMMANDS = {
    "data": ("build_point_readout_qa", ()),
    "prepare": ("prepare_readout_training_cache", ()),
    "train": ("train_point_readout", ()),
    "evaluate": ("train_point_readout", ("--evaluate-only",)),
    "infer": ("chat_point_readout", ()),
    "score": ("score_point_readout", ()),
    "baseline": ("benchmark_point_readout", ()),
    "benchmark": ("benchmark_readout_efficiency", ()),
    "analyze": ("analyze_field_predictions", ()),
    "compare-blocks": ("compare_block_readout", ()),
    "environment": ("check_field_qa_environment", ()),
    "download-ins": ("download_ins_tracer", ()),
    "download-well": ("download_the_well", ()),
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__ + "\n\nUsage: python field.py COMMAND [OPTIONS]\n")
        print("Commands: " + ", ".join(COMMANDS))
        print("\nUse COMMAND --help for operation-specific options.")
        return
    command = args.pop(0)
    if command not in COMMANDS:
        raise SystemExit(f"Unknown command {command!r}; use --help to list commands.")
    module, fixed = COMMANDS[command]
    sys.argv = [f"field.py {command}", *args, *fixed]
    import_module(f"tensor_compression.workflows.{module}").main()


if __name__ == "__main__":
    main()
