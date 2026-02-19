import numpy as np
import torch
from rapidfuzz import fuzz


def contains_word(s, w):
    return w in s


def computeMaximumAttentionScore(activity: np.ndarray) -> np.ndarray:
    """
    Compute the Maximum Attention Score (MAS) given self-attention values.
    Shape expected: (num_layers, num_heads, num_tokens)

    Returns
    -------
    ndarray
        MAS values in range [0,1]
    """
    MAS_res = np.zeros(activity.shape[2])

    for x in range(activity.shape[0]):
        for y in range(activity.shape[1]):
            max_idx = np.argmax(activity[x, y, :])
            max_val = activity[x, y, max_idx]
            MAS_res[max_idx] += max_val

    total = np.sum(MAS_res)
    if total > 0:
        MAS_res /= total
    return MAS_res


def analyzeAttentionSingleTupleDecoy(
    model,
    tokenizer,
    data,
    guid_dict=None,
    select_guid=None,
    num_layers=12,
    num_heads=12,
    do_debug=False,
):
    """
    Extract attention of target vs decoy words.
    """
    problem_list = set([])
    activity = np.zeros((num_layers, num_heads))
    attention_matrix = None

    print(f"{guid_dict=}, {select_guid=}, {len(data)=}")

    if select_guid is not None and select_guid not in guid_dict:
        problem_list.add(select_guid)
        if do_debug:
            print(f"GUID {select_guid} not found in guid_dict.")
        return activity, attention_matrix

    if select_guid is None:
        elements = range(len(data))
    else:
        assert guid_dict is not None
        elements = [guid_dict[select_guid]]

    for idx in elements:
        sentence_a = data[idx].text_a
        sentence_b = data[idx].text_b
        groundtruth = data[idx].groundtruth
        # guid = data[idx].guid
        decoy = data[idx].decoy
        reference_idx = data[idx].reference_idx

        if groundtruth is None:
            continue

        # Use transformers to get attention
        inputs = tokenizer.encode_plus(
            sentence_a, sentence_b, return_tensors="pt", add_special_tokens=True
        )
        input_ids = inputs["input_ids"]
        token_type_ids = inputs["token_type_ids"]

        # Get model outputs with attention
        with torch.no_grad():
            outputs = model(input_ids, token_type_ids=token_type_ids, output_attentions=True)
            all_attentions = outputs.attentions  # Tuple of (batch, heads, seq_len, seq_len) per layer

        # Convert to numpy and format: (num_layers, num_heads, seq_len, seq_len)
        attn = torch.stack([layer_attn.squeeze(0) for layer_attn in all_attentions]).numpy()

        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        # Split tokens into sentence_a and sentence_b based on token_type_ids
        tokens_a = [t for t, tid in zip(tokens, token_type_ids[0]) if tid == 0 and t not in ['[CLS]', '[SEP]', '']]
        tokens_b = [t for t, tid in zip(tokens, token_type_ids[0]) if tid == 1 and t not in ['[CLS]', '[SEP]', '']]

        # Find the separator position to map indices correctly
        sep_positions = [i for i, t in enumerate(tokens) if t == '[SEP]']
        if len(sep_positions) >= 1:
            # First [SEP] marks end of sentence_a
            sentence_a_end = sep_positions[0]
        else:
            sentence_a_end = len(tokens_a) + 1  # +1 for [CLS]

        # Map reference_idx to actual token position in the full sequence
        actual_reference_idx = sentence_a_end + 1 + reference_idx  # +1 to skip first [SEP]

        groundtruth_tokens = tokenizer.tokenize(groundtruth)

        activity = np.zeros((num_layers, num_heads, len(decoy) + 1))
        attention_matrix = np.zeros((num_layers, num_heads, len(decoy) + 1))

        if (
            tokenizer.tokenize(groundtruth)[0] not in sentence_a
            and tokenizer.tokenize(groundtruth)[0] not in sentence_b
        ):
            if do_debug:
                print(f"Wrong annotation: {sentence_a} | {groundtruth} | {sentence_b}")
            continue

        for layer_id in range(num_layers):
            for head_id in range(num_heads):
                # Get attention from all tokens to the reference token
                attention_to_ref = attn[layer_id, head_id, :, actual_reference_idx]

                correct_activity = 0
                decoy_attention = []

                # Correct word attention
                if contains_word(sentence_a, groundtruth_tokens[0]):
                    for k in decoy:
                        if len(groundtruth_tokens) == 1:
                            ratios = np.array(
                                [fuzz.ratio(k, token) for token in tokens_a]
                            )
                            best_match_idx = int(np.argmax(ratios))
                            actual_best_idx = 1 + best_match_idx  # +1 for [CLS]
                            correct_activity = attention_to_ref[actual_best_idx]
                        else:
                            local_attention = []
                            for f in groundtruth_tokens:
                                if f in tokens_a:
                                    token_idx = tokens_a.index(f)
                                    actual_idx = 1 + token_idx  # +1 for [CLS]
                                    local_attention.append(
                                        attention_to_ref[actual_idx]
                                    )
                            if local_attention:
                                correct_activity = np.max(local_attention)
                else:
                    if len(groundtruth_tokens) == 1:
                        if groundtruth in tokens_b:
                            token_idx = tokens_b.index(groundtruth)
                            actual_idx = sentence_a_end + 1 + token_idx
                            correct_activity = attention_to_ref[actual_idx]
                    else:
                        local_attention = []
                        for f in groundtruth_tokens:
                            if f in tokens_b:
                                token_idx = tokens_b.index(f)
                                actual_idx = sentence_a_end + 1 + token_idx
                                local_attention.append(
                                    attention_to_ref[actual_idx]
                                )
                        if local_attention:
                            correct_activity = np.max(local_attention)

                # Decoy attention
                if contains_word(sentence_a, groundtruth_tokens[0]):
                    for k in decoy:
                        if len(tokenizer.tokenize(k)) == 1:
                            ratios = [fuzz.ratio(k, token) for token in tokens_a]
                            best_match_idx = int(np.argmax(ratios))
                            actual_best_idx = 1 + best_match_idx  # +1 for [CLS]
                            decoy_attention.append(
                                attention_to_ref[actual_best_idx]
                            )
                        else:
                            local_attention = []
                            for f in tokenizer.tokenize(k):
                                if f in tokens_a:
                                    token_idx = tokens_a.index(f)
                                    actual_idx = 1 + token_idx  # +1 for [CLS]
                                    local_attention.append(
                                        attention_to_ref[actual_idx]
                                    )
                            decoy_attention.append(
                                np.max(local_attention) if local_attention else 0
                            )
                else:
                    for k in decoy:
                        if len(tokenizer.tokenize(k)) == 1:
                            if k in tokens_b:
                                token_idx = tokens_b.index(k)
                                actual_idx = sentence_a_end + 1 + token_idx
                                decoy_attention.append(
                                    attention_to_ref[actual_idx]
                                )
                        else:
                            local_attention = []
                            for f in tokenizer.tokenize(k):
                                if f in tokens_b:
                                    token_idx = tokens_b.index(f)
                                    actual_idx = sentence_a_end + 1 + token_idx
                                    local_attention.append(
                                        attention_to_ref[actual_idx]
                                    )
                            decoy_attention.append(
                                np.max(local_attention) if local_attention else 0
                            )

                attn_vals = [correct_activity] + decoy_attention
                activity[head_id, layer_id, np.argmax(np.array(attn_vals))] += 1
                attention_matrix[head_id, layer_id, :] = np.asarray(attn_vals)

    if do_debug and problem_list:
        print("Problems with GUIDs:", problem_list)

    return activity, attention_matrix

