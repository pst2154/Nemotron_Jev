"""SemIf-style execution with unchanged Nimble prompts and native hybrid caches.

No generation, weight edits, or persistent cross-request cache. Equal-length
suffix buckets avoid padding-dependent recurrent-state changes.
"""
import copy
from collections import defaultdict
from dataclasses import replace
import torch


def load_native_model(checkpoint, adapter=None):
    """Load BF16 without silently accepting missing or mismatched weights."""
    from transformers import AutoModelForCausalLM
    model, info = AutoModelForCausalLM.from_pretrained(
        checkpoint, dtype=torch.bfloat16, device_map='cuda',
        attn_implementation='sdpa', output_loading_info=True,
    )
    failures = {k: info[k] for k in ('missing_keys', 'mismatched_keys', 'error_msgs') if info.get(k)}
    unexpected = [k for k in info.get('unexpected_keys', []) if not k.startswith('mtp.')]
    if failures or unexpected:
        raise RuntimeError(f'Checkpoint mismatch: {failures}, unexpected={unexpected}')
    model.set_experts_implementation('grouped_mm')
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return model.eval()


def align_prompts(prepared, multiple):
    """Move only the split point; full prompt tokens are unchanged."""
    if multiple < 1:
        raise ValueError('Alignment must be positive')
    length = len(prepared.prefix_ids) // multiple * multiple
    if not length:
        return prepared
    return replace(prepared, prefix_ids=prepared.prefix_ids[:length],
                   suffix_ids=[ids[length:] for ids in prepared.full_ids])


class NativeScorer:
    def __init__(self, model):
        self.base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.base.eval()
        self.device = self.base.get_input_embeddings().weight.device

    def project(self, hidden, candidates):
        ids = torch.tensor(candidates, device=self.device)
        head = self.base.get_output_embeddings().weight.index_select(0, ids)
        return hidden.float() @ head.float().T

    @torch.inference_mode()
    def independent(self, prepared):
        results = []
        for ids, candidates in zip(prepared.full_ids, prepared.candidate_ids):
            hidden = self.base.model(input_ids=torch.tensor([ids], device=self.device),
                                     use_cache=False).last_hidden_state[0, -1]
            results.append(self.project(hidden, candidates))
        return results

    @torch.inference_mode()
    def batched_independent(self, prepared, batch_size=8):
        """Full prompts batched together, without prefix reuse or padding."""
        if batch_size < 1:
            raise ValueError('Positive batch size required')
        groups = defaultdict(list)
        for i, ids in enumerate(prepared.full_ids):
            groups[len(ids)].append(i)
        results = [None] * len(prepared.names)
        for indices in groups.values():
            for offset in range(0, len(indices), batch_size):
                batch = indices[offset:offset+batch_size]
                tokens = torch.tensor([prepared.full_ids[i] for i in batch], device=self.device)
                hidden = self.base.model(input_ids=tokens, use_cache=False).last_hidden_state[:, -1]
                for row, i in enumerate(batch):
                    results[i] = self.project(hidden[row], prepared.candidate_ids[i])
        return results

    @torch.inference_mode()
    def shared(self, prepared, batch_size=8, optimized=False):
        if batch_size < 1 or not prepared.prefix_ids:
            raise ValueError("Positive batch size and nonempty prefix required")
        if any(not s for s in prepared.suffix_ids):
            raise ValueError("Each question requires a nonempty suffix")
        if any(prepared.prefix_ids + suffix != full
               for suffix, full in zip(prepared.suffix_ids, prepared.full_ids)):
            raise ValueError("Shared prefix/suffix must reproduce the independent prompt exactly")
        if optimized and len(prepared.names) == 1:
            # Splitting a lone question into two forwards has no reuse benefit.
            return self.independent(prepared)
        prefix = torch.tensor([prepared.prefix_ids], device=self.device)
        cached = self.base.model(input_ids=prefix, use_cache=True).past_key_values
        groups = defaultdict(list)
        for i, suffix in enumerate(prepared.suffix_ids):
            groups[len(suffix)].append(i)
        results = [None] * len(prepared.names)
        jobs = [(length, indices[offset:offset + batch_size])
                for length, indices in groups.items()
                for offset in range(0, len(indices), batch_size)]
        for job, (length, batch) in enumerate(jobs):
            # Native reorder forks BOTH attention KV and Mamba recurrent/conv state.
            # The last batch may consume this request-local cache without copying it.
            branch = cached if optimized and job == len(jobs)-1 else copy.deepcopy(cached)
            if not optimized or len(batch) != 1:
                branch.reorder_cache(torch.zeros(len(batch), dtype=torch.long, device=self.device))
            tokens = torch.tensor([prepared.suffix_ids[i] for i in batch], device=self.device)
            positions = torch.arange(prefix.shape[1], prefix.shape[1] + length,
                                     device=self.device).unsqueeze(0).expand(len(batch), -1)
            hidden = self.base.model(input_ids=tokens, past_key_values=branch,
                                     position_ids=positions, use_cache=True).last_hidden_state[:, -1]
            if optimized:
                candidate_groups = defaultdict(list)
                for row, i in enumerate(batch):
                    candidate_groups[tuple(prepared.candidate_ids[i])].append((row, i))
                for candidates, rows in candidate_groups.items():
                    projected = self.project(hidden[[row for row, _ in rows]], candidates)
                    for j, (_, i) in enumerate(rows):
                        results[i] = projected[j]
            else:
                for row, i in enumerate(batch):
                    results[i] = self.project(hidden[row], prepared.candidate_ids[i])
            del branch, hidden
        return results
