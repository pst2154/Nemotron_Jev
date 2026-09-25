"""Explicit decision accounting; legacy output_tokens is not text generation."""


def decision_usage(input_tokens, questions, *, sampled_tokens=0):
    return {
        'input_tokens': input_tokens,
        # Preserve the existing API convention, including benchmark accounting.
        'output_tokens': questions,
        'output_tokens_basis': 'scored_decision_positions',
        'decision_positions': questions,
        'generated_text_tokens': 0,
        'discarded_sampled_tokens': sampled_tokens,
    }
