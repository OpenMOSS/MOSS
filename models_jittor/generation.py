import jittor as jt

def generate(moss, input_str, tokenizer, method, **kwargs):
    """
    Choose different methods to generate sentences.

    :param input_str: The input text.
    :param tokenizer: Tokenizer.
    :param method: Generation method. Should be one of: ['greedy', 'sample']
    :param kwargs: Other parameters used for generation.
        - max_gen_len: int. Maximum generate length. Used in all methods.
        - temperature: float. Used in ``sample``.
        - top_p: float. Used in ``sample``.
        - top_k: int. Used in ``sample``.
    """
    if method == "greedy":
        return greedy_search(moss, input_str, tokenizer, **kwargs)
    elif method == "sample":
        return sample(moss, input_str, tokenizer, **kwargs)
    else:
        raise NotImplementedError(
            f"Unsupported generation method {method}"
        )

def greedy_search(model, input_str, tokenizer, max_gen_len,
                  eos_token_id=None, pad_token_id=None):
    model.eval()
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    if pad_token_id is None and eos_token_id is not None:
        pad_token_id = eos_token_id
    eos_token_id_tensor = jt.Var(eos_token_id)

    tokenized = tokenizer(input_str, return_tensors='np')
    sentence_ids = jt.Var(tokenized['input_ids'])
    attention_mask = jt.Var(tokenized['attention_mask'])
    unfinished_sequences = sentence_ids.new(sentence_ids.shape[0]).fill_(1)
    past_key_values = None
    while True:
        # set input
        if past_key_values:
            input_ids = sentence_ids[:, -1].unsqueeze(-1)
        else:
            input_ids = sentence_ids
            
        outputs = model(input_ids, past_key_values=past_key_values,
                        attention_mask=attention_mask)
        # caculate probs
        next_token_logits = outputs['logits'][:, -1, :].float()
        next_tokens = jt.argmax(next_token_logits, dim=-1)[0]

        # concat sentence
        next_tokens = next_tokens * unfinished_sequences + \
            pad_token_id * (1 - unfinished_sequences)
        sentence_ids = jt.cat([sentence_ids, next_tokens[:, None]], dim=-1)
        # update input
        past_key_values = outputs['past_key_values']
        attention_mask = jt.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

        # if eos_token was found in one sentence, set sentence to finished
        next_tokens.repeat(eos_token_id_tensor.shape[0], 1)
        unfinished_sequences = unfinished_sequences.mul(
            next_tokens.repeat(eos_token_id_tensor.shape[0], 1) \
                       .not_equal(eos_token_id_tensor.unsqueeze(1)) \
                       .prod(dim=0)
        )

        jt.sync_all()

        if unfinished_sequences.max() == 0 or sentence_ids.shape[-1] >= max_gen_len:
            break

    return sentence_ids.reshape([-1,]).tolist()[tokenized['input_ids'].shape[1]:]

def sample(model, input_str, tokenizer, max_gen_len, temperature, top_p, top_k,
           eos_token_id=None, pad_token_id=None):
    model.eval()
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    if pad_token_id is None and eos_token_id is not None:
        pad_token_id = eos_token_id
    eos_token_id_tensor = jt.Var(eos_token_id)

    tokenized = tokenizer(input_str, return_tensors='np')
    sentence_ids = jt.Var(tokenized['input_ids'])
    attention_mask = jt.Var(tokenized['attention_mask'])
    unfinished_sequences = sentence_ids.new(sentence_ids.shape[0]).fill_(1)
    past_key_values = None

    while True:

        # set input
        if past_key_values:
            input_ids = sentence_ids[:, -1].unsqueeze(-1)
        else:
            input_ids = sentence_ids
        outputs = model(input_ids, past_key_values=past_key_values,
                        attention_mask=attention_mask)

        next_token_logits = outputs['logits'][:, -1, :].float()

        # sample
        # temperature
        scores = next_token_logits / temperature
        # top_k
        scores = sample_top_k(scores, top_k)
        # top_p
        scores = sample_top_p(scores, top_p)

        probs = jt.nn.softmax(scores, dim=-1)
        next_tokens = jt.multinomial(probs, num_samples=1).squeeze(1)
        # concat sentence
        next_tokens = next_tokens * unfinished_sequences + \
            pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        sentence_ids = jt.cat([sentence_ids, next_tokens[:, None]], dim=-1)
        past_key_values = outputs['past_key_values']
        attention_mask = jt.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

        # if eos_token was found in one sentence, set sentence to finished
        next_tokens.repeat(eos_token_id_tensor.shape[0], 1)
        unfinished_sequences = unfinished_sequences.mul(
            next_tokens.repeat(eos_token_id_tensor.shape[0], 1) \
                       .not_equal(eos_token_id_tensor.unsqueeze(1)) \
                       .prod(dim=0)
        )

        jt.sync_all()

        if unfinished_sequences.max() == 0 or sentence_ids.shape[-1] >= max_gen_len:
            break

    return sentence_ids.reshape([-1,]).tolist()[tokenized['input_ids'].shape[1]:]

def sample_top_k(scores, top_k):
    top_k = min(top_k, scores.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = scores < jt.topk(scores, top_k)[0][..., -1, None]
    scores = scores.masked_fill(indices_to_remove, -float("Inf"))

    return scores

def sample_top_p(scores, top_p):
    sorted_logits, sorted_indices = jt.sort(scores, descending=False)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - top_p)

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    scores = scores.masked_fill(indices_to_remove, -float("Inf"))
    
    return scores
