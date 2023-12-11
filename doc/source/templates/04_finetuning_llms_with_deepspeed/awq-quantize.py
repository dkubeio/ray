from pathlib import Path
import argparse
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

quant_config = { "zero_point": True, "q_group_size": 128, "w_bit": 4 }

parser = argparse.ArgumentParser(description="quantize the merged model")

parser.add_argument(
    "--quant-path",
    type=str,
    help="Path to quant output directory.",
    required=True,
)

parser.add_argument("--model-path", required=False, type=str, default="", help="merged model path")
args = parser.parse_args()

if not Path(args.model_path).exists():
    raise ValueError(f"Merged model {args.model_path} does not exist.")

if not args.quant_path:
    args.quant_path = Path(args.model_path) / "_awq"
    print(f"Quant output path not specified. Using {args.quant_path}")

Path(args.quant_path).mkdir(parents=True, exist_ok=True)
# Load model
model = AutoAWQForCausalLM.from_pretrained(args.model_path, safetensors=True)
tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, safetensors=True)

# Quantize
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model
model.save_quantized(args.quant_path)
tokenizer.save_pretrained(args.quant_path)
