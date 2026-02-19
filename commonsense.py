#
# SPDX-FileCopyrightText: 2019-2020 SAP SE or an SAP affiliate company
#
# SPDX-License-Identifier: Apache-2.0
#


import argparse

import numpy as np
import torch
from rapidfuzz import fuzz
from tqdm import trange
from transformers import BertModel, BertTokenizer

import data_processors as processors


def contains_word(s, w):
    return s.find(w) > -1


def computeMaximumAttentionScore(activity):
    """
    Compute the Maximum Attention Score (MAS) given the self-attention of a transformer network architecture.


    Parameters
    ----------
    activity : ndarray
        tensor of attentions across layers and heads

    Returns
    -------
    ndarray
        MAS values in range [0,1]

    """
    MAS_res = np.zeros((activity.shape[2]))
    # loop over layers and heads
    for x in range(0, activity.shape[0]):
        for y in range(0, activity.shape[0]):
            # get word with max attention and its attention
            max_idx = np.argmax(activity[x, y, :])
            max_val = np.max(activity[x, y, :])

            # mask out all non-max values
            MAS_res[max_idx] += max_val

    # now normalize the attentions
    MAS_res /= np.sum(MAS_res)
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
    Extracts the attention of target words, e.g. groundtruth and decoy word.

    Note: If target word is tokenized, we only consider the max attention here.


    Parameters
    ----------
    model : transformers.BertModel
        BERT model from transformers that provides access to attention
    tokenizer:  transformers.BertTokenizer
        BERT tokenizer
    data: InputItems[]
        List of InputItems containing the WNLI/PDP data
    guid_dict: dictionary
        Dictionary that maps unique ids to data indices. Default None
    select_guid: int
        GUID of example data, for which the attentions are to be extracted
    num_layers: int
        Number of layers. Default 12
    num_heads: int
        Number of attention heads. Default 12
    do_debug: boolean
        Toggle for printing debug information. Default False

    Returns
    -------

    activity : ndarray
        Count matrix, keeping track which layer and head is associated with max attention
    attention : ndarray
        Attention matrix (#layers, #heads, 2), containing for each head and layer the attention for true word and decoy, respectively

    """

    problem_list = set([])
    activity = np.zeros((num_layers, num_heads))

    if select_guid is None:
        elements = range(0, len(data))
    else:
        assert guid_dict is not None
        assert select_guid is not None
        elements = [guid_dict[select_guid]]

    for idx in elements:
        sentence_a = data[idx].text_a
        sentence_b = data[idx].text_b
        groundtruth = data[idx].groundtruth
        guid = data[idx].guid
        decoy = data[idx].decoy

        if groundtruth is not None:
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
            tokens_a = [t for t, tid in zip(tokens, token_type_ids[0]) if tid == 0 and t not in ['[CLS]', '[SEP]', '[PAD]']]
            tokens_b = [t for t, tid in zip(tokens, token_type_ids[0]) if tid == 1 and t not in ['[CLS]', '[SEP]', '[PAD]']]

            # Find the separator position to map indices correctly
            sep_positions = [i for i, t in enumerate(tokens) if t == '[SEP]']
            if len(sep_positions) >= 1:
                # First [SEP] marks end of sentence_a
                sentence_a_end = sep_positions[0]
            else:
                sentence_a_end = len(tokens_a) + 1  # +1 for [CLS]

            groundtruth_tokens = tokenizer.tokenize(data[idx].groundtruth)
            activity = np.zeros((num_layers, num_heads, len(decoy) + 1))
            attention_matrix = np.zeros((num_layers, num_heads, len(decoy) + 1))
            reference_idx = data[idx].reference_idx

            # Map reference_idx to actual token position in the full sequence
            # reference_idx is in tokens_b, need to map to position in full token sequence
            # Full sequence: [CLS] + tokens_a + [SEP] + tokens_b + [SEP]
            actual_reference_idx = sentence_a_end + 1 + reference_idx  # +1 to skip first [SEP]

            if (
                tokenizer.tokenize(groundtruth)[0] not in sentence_a
                and tokenizer.tokenize(groundtruth)[0] not in sentence_b
            ):
                print(
                    "Wrong annotation: "
                    + sentence_a
                    + " | "
                    + groundtruth
                    + " | "
                    + sentence_b
                )
                continue

            for layer_id in range(0, num_layers):
                for head_id in range(0, num_heads):
                    # attention shape: (layer, head, seq_len, seq_len)
                    # Get attention from all tokens to the reference token
                    attention_to_ref = attn[layer_id, head_id, :, actual_reference_idx]

                    correct_activity = 0
                    indices = []

                    # determine attention for the correct word

                    # check if correct word is in sentence_a OR sentence_b
                    if contains_word(sentence_a, groundtruth_tokens[0]):
                        # check if target is single or multi-token
                        if len(tokenizer.tokenize(groundtruth)) == 1:
                            # some answers might not be perfect match or misspellings, e.g. plural piece(s), so fuzzy matching necessary
                            ratios = [
                                fuzz.ratio(groundtruth, token) for token in tokens_a
                            ]
                            best_match_idx = ratios.index(max(ratios))

                            # Map to actual position in full sequence
                            actual_best_idx = 1 + best_match_idx  # +1 for [CLS]
                            correct_activity = attention_to_ref[actual_best_idx]
                            indices.append(best_match_idx)

                        # target streches over multiple tokens
                        else:
                            groundtruth_split = tokenizer.tokenize(groundtruth)
                            local_attention = []
                            for f in groundtruth_split:
                                if len(f) > 1:
                                    try:
                                        token_idx = tokens_a.index(f)
                                        actual_idx = 1 + token_idx  # +1 for [CLS]
                                        local_attention.append(
                                            attention_to_ref[actual_idx]
                                        )
                                        indices.append(token_idx)
                                    except:
                                        problem_list.add(guid)
                                        pass
                            # keep max attention
                            if len(local_attention) > 0:
                                correct_activity = np.max(local_attention)

                    else:
                        # check if target is single or multi-token
                        if len(tokenizer.tokenize(groundtruth)) == 1:
                            try:
                                token_idx = tokens_b.index(groundtruth)
                                actual_idx = sentence_a_end + 1 + token_idx
                                correct_activity = attention_to_ref[actual_idx]
                                indices.append(token_idx)
                            except:
                                problem_list.add(guid)
                                pass

                        # target stretches over multiple tokens
                        else:
                            groundtruth_split = tokenizer.tokenize(groundtruth)
                            local_attention = []
                            for f in groundtruth_split:
                                if len(f) > 1:
                                    try:
                                        token_idx = tokens_b.index(f)
                                        actual_idx = sentence_a_end + 1 + token_idx
                                        local_attention.append(
                                            attention_to_ref[actual_idx]
                                        )
                                        indices.append(token_idx)
                                    except:
                                        problem_list.add(guid)
                                        pass
                            # keep max attention
                            if len(local_attention) > 0:
                                correct_activity = np.max(local_attention)

                    # determine attention for the decoy word

                    decoy_attention = []

                    if contains_word(sentence_a, groundtruth_tokens[0]):
                        for k in decoy:
                            # check if target is single or multi-token
                            if len(tokenizer.tokenize(k)) == 1:
                                # some answers might not be perfect match or misspellings, e.g. plural piece(s), so fuzzy matching necessary
                                ratios = [fuzz.ratio(k, token) for token in tokens_a]
                                best_match_idx = ratios.index(max(ratios))
                                actual_best_idx = 1 + best_match_idx  # +1 for [CLS]
                                decoy_attention.append(
                                    attention_to_ref[actual_best_idx]
                                )

                                indices.append(best_match_idx)
                            else:
                                decoy_split = tokenizer.tokenize(k)
                                local_attention = []

                                for f in decoy_split:
                                    if len(f) > 1:
                                        try:
                                            token_idx = tokens_a.index(f)
                                            actual_idx = 1 + token_idx  # +1 for [CLS]
                                            local_attention.append(
                                                attention_to_ref[actual_idx]
                                            )
                                            indices.append(token_idx)
                                        except:
                                            problem_list.add(guid)
                                            pass

                                if len(local_attention) > 0:
                                    decoy_attention.append(np.max(local_attention))
                                else:
                                    decoy_attention.append(0)

                    else:
                        for k in decoy:
                            # check if target is single or multi-token
                            if len(tokenizer.tokenize(k)) == 1:
                                try:
                                    token_idx = tokens_b.index(k)
                                    actual_idx = sentence_a_end + 1 + token_idx
                                    decoy_attention.append(
                                        attention_to_ref[actual_idx]
                                    )
                                except:
                                    decoy_attention.append(0)
                                    problem_list.add(guid)
                            else:
                                decoy_split = tokenizer.tokenize(k)
                                local_attention = []
                                for f in decoy_split:
                                    if len(f) > 1:
                                        try:
                                            token_idx = tokens_b.index(f)
                                            actual_idx = sentence_a_end + 1 + token_idx
                                            # some answers might not be perfect match or misspellings, e.g. plural piece(s), so fuzzy matching necessary
                                            ratios = [
                                                fuzz.ratio(f, token) for token in tokens_b
                                            ]
                                            best_match_idx = ratios.index(max(ratios))
                                            actual_best_idx = sentence_a_end + 1 + best_match_idx

                                            local_attention.append(
                                                attention_to_ref[actual_best_idx]
                                            )
                                            indices.append(best_match_idx)
                                        except:
                                            problem_list.add(guid)
                                            pass

                                if len(local_attention) > 0:
                                    decoy_attention.append(np.max(local_attention))
                                else:
                                    decoy_attention.append(0)

                    attn_vals = [correct_activity] + decoy_attention

                    activity[head_id, layer_id, np.argmax(attn_vals)] += 1
                    attention_matrix[head_id, layer_id, :] = np.asarray(attn_vals[:])

    if do_debug and len(problem_list) > 0:
        print("Problems with following guids: " + str(problem_list))
    return activity, attention_matrix


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Shou5ld contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--bert_model",
        default=None,
        type=str,
        required=True,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
        "bert-base-multilingual-cased, bert-base-chinese.",
    )
    parser.add_argument(
        "--task_name",
        choices=["PDP", "MNLI", "pdp", "mnli"],
        required=True,
        help="The name of the task to train, pdp or WNLI.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Set this flag if you are want to print debug infos.",
    )

    args = parser.parse_args()

    if args.task_name.lower() == "pdp":
        processor = processors.XMLPDPProcessor()
        tokenizer = BertTokenizer.from_pretrained(
            args.bert_model, do_lower_case=args.do_lower_case
        )
        all_ambigious, guid_dict = processor.get_train_items(
            args.data_dir, select_type="ambigious"
        )
        all_guids = set([])
        for i in range(0, len(all_ambigious)):
            all_guids.add(all_ambigious[i].guid)

    elif args.task_name.lower() == "mnli":
        processor = processors.XMLMnliProcessor()
        tokenizer = BertTokenizer.from_pretrained(
            args.bert_model, do_lower_case=args.do_lower_case
        )
        all_ambigious, guid_dict = processor.get_train_items(
            args.data_dir, select_type="ambigious"
        )
        all_guids = set([])
        for i in range(0, len(all_ambigious)):
            all_guids.add(all_ambigious[i].guid)

    model = BertModel.from_pretrained(args.bert_model)

    MAS_list = []
    counter = np.zeros((2))

    for i in trange(0, len(all_guids)):
        _, attention = analyzeAttentionSingleTupleDecoy(
            model,
            tokenizer,
            all_ambigious,
            guid_dict,
            select_guid=i,
            do_debug=args.debug,
        )
        ref_word = tokenizer.tokenize(all_ambigious[i].text_b)[
            all_ambigious[i].reference_idx
        ]
        MAS = computeMaximumAttentionScore(attention)

        MAS_list.append(MAS[0])
        # now count how many time MAX is assigned either the true word or the decoy
        if np.argmax(MAS) == 0:
            counter[0] += 1
        else:
            counter[1] += 1

        if args.debug:
            print(
                str(all_ambigious[i].guid)
                + " | "
                + str(MAS)
                + " | "
                + all_ambigious[i].text_a
                + " "
                + all_ambigious[i].text_b
                + " || >"
                + ref_word
                + "<, "
                + str(all_ambigious[i].groundtruth)
                + ", "
                + str(all_ambigious[i].decoy)
            )

    print(args.task_name.upper() + " Accuracy: " + str(counter[0] / np.sum(counter)))


if __name__ == "__main__":
    main()

