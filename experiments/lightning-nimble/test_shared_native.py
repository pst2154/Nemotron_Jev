"""CPU hybrid-cache equivalence tests; no full checkpoint required."""
import torch
import argparse
from dataclasses import replace
from transformers import NemotronHConfig, NemotronHForCausalLM
from nimble.scoring.parallel_schema import PreparedPrompts
from shared_native import NativeScorer, align_prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(17)
    config = NemotronHConfig(
        vocab_size=128, hidden_size=64,
        layers_block_type=["linear_attention", "moe", "full_attention", "linear_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        intermediate_size=32, mamba_num_heads=8, mamba_head_dim=16,
        n_groups=2, ssm_state_size=8, conv_kernel=4, chunk_size=8,
        n_routed_experts=4, n_shared_experts=1, moe_intermediate_size=32,
        moe_shared_expert_intermediate_size=32, num_experts_per_tok=2,
        n_group=1, topk_group=1, max_position_embeddings=256,
        num_nextn_predict_layers=0, use_mamba_kernels=False,
    )
    model = NemotronHForCausalLM(config).to(args.device).eval()
    versions = [p._version for p in model.parameters()]
    scorer = NativeScorer(model)
    for prefix_length in (8, 17, 65):
        prefix = torch.randint(3, 120, (prefix_length,)).tolist()
        suffixes = [torch.randint(3, 120, (n,)).tolist() for n in (1, 3, 3, 7, 7)]
        candidates = [[5, 8], [5, 8, 11], [5, 8], [5, 8, 11], [5, 8]]
        p = PreparedPrompts(list(range(5)), [list(range(len(c))) for c in candidates], prefix, suffixes,
                            [prefix+s for s in suffixes], candidates)
        reference = scorer.independent(p)
        for batch in (1, 2, 8):
            for a, b in zip(scorer.batched_independent(p,batch),reference):
                torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
            for optimized in (False, True):
                for repeat in range(2):
                    actual = scorer.shared(p, batch, optimized=optimized)
                    for a, b in zip(actual, reference):
                        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
        single = replace(p, names=p.names[:1], choices=p.choices[:1], suffix_ids=p.suffix_ids[:1],
                         full_ids=p.full_ids[:1], candidate_ids=p.candidate_ids[:1])
        torch.testing.assert_close(scorer.shared(single, optimized=True)[0], reference[0], atol=0, rtol=0)
        p.names.reverse(); p.suffix_ids.reverse(); p.full_ids.reverse(); p.candidate_ids.reverse()
        for a, b in zip(scorer.shared(p, 8), reversed(reference)):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
        aligned = align_prompts(p, 8)
        assert aligned.full_ids == p.full_ids
        for a, b in zip(scorer.shared(aligned, 8, optimized=True), reversed(reference)):
            torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    assert versions == [p._version for p in model.parameters()], 'Weights mutated'
    print("PASS: hybrid-cache equivalence, mixed lengths, branch order, repeated calls, batches 1/2/8")


if __name__ == "__main__":
    main()
