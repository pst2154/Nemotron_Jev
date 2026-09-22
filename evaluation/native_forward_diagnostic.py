"""Isolate BF16 single-pass versus cached-mask execution in the native model.

This is a numerical diagnostic, not a serving optimization or accuracy benchmark.
"""
import argparse
import json
from pathlib import Path

from compare_backends import workload
from scoring import candidate_codes, evaluate, prepare_questions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError('Existing diagnostic results are preserved')

    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.checkpoint, trust_remote_code=True, dtype=torch.bfloat16).to('cuda').eval()
    payload, _ = workload(tokenizer, 100, 1000)
    payload['questions'] = {key: payload['questions'][key] for key in ['2', '97', '98']}
    reference = evaluate(model, tokenizer, payload)
    codes = candidate_codes(tokenizer)
    layers = [layer.self_attn for layer in model.encoder.layers]
    previous = [layer.diffusion_lm for layer in layers]
    rows = []
    try:
        with torch.inference_mode():
            for layer in layers:
                layer.diffusion_lm = False
            for key, _, labels, _, ids in prepare_questions(tokenizer, payload):
                inputs = torch.tensor([ids + [model.mask_token_id]], device=model.device)
                hidden = model.encoder(input_ids=inputs, use_cache=False,
                                       use_causal_mask=True, is_training=False).last_hidden_state
                logits = model.diffusion_head(hidden[:, -1:])[0, 0].float()
                probs = logits[[token for _, token in codes[:len(labels)]]].softmax(-1).cpu().tolist()
                rows.append({'question': key,
                             'cached_mask': reference['answers'][key]['probabilities'],
                             'single_pass': dict(zip(labels, probs))})
    finally:
        for layer, value in zip(layers, previous):
            layer.diffusion_lm = value
    destination.write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
