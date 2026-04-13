from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import LlamaConfig

from open_instruct import dataset_transformation as dt
from open_instruct.llopa_adapter import PREFILL_LOWER_SYSTEM_LEN_KEY, install_llopa_modeling


class DummyTokenizer:
    class _Encoding:
        def __init__(self, input_ids: torch.Tensor):
            self.input_ids = input_ids

    def _render(self, conversation, add_generation_prompt: bool = False) -> str:
        parts = []
        for msg in conversation:
            parts.append(f"<{msg['role']}>{msg['content']}</{msg['role']}>")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def apply_chat_template(
        self,
        conversation,
        tokenize=False,
        return_tensors="pt",
        padding=False,
        truncation=False,
        max_length=None,
        add_generation_prompt=False,
    ):
        assert return_tensors == "pt"
        assert padding is False
        text = self._render(conversation, add_generation_prompt=add_generation_prompt)
        if not tokenize:
            return text
        ids = [ord(ch) + 1 for ch in text]
        if truncation and max_length is not None:
            ids = ids[: int(max_length)]
        return torch.tensor([ids], dtype=torch.long)

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        assert not add_special_tokens
        assert return_tensors == "pt"
        ids = torch.tensor([[ord(ch) + 1 for ch in text]], dtype=torch.long)
        return self._Encoding(ids)


def _tri_module():
    repo_root = Path(__file__).resolve().parents[2]
    modeling_path = repo_root / "Capsule" / "tri_llama3_modeling.py"
    install_llopa_modeling(str(modeling_path), model_family="llama")
    import transformers.models.llama.modeling_llama as tri_llama

    return tri_llama


def _llopa_inference_module():
    import importlib.util

    repo_root = Path(__file__).resolve().parents[2]
    capsule_root = repo_root / "Capsule"
    if str(capsule_root) not in sys.path:
        sys.path.insert(0, str(capsule_root))
    module_path = repo_root / "Capsule" / "llopa_inference.py"
    spec = importlib.util.spec_from_file_location("capsule_test_llopa_inference", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load llopa_inference module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_tiny_model():
    tri_llama = _tri_module()
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        hidden_act="silu",
        attention_bias=False,
        mlp_bias=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = tri_llama.LlamaForCausalLM(config)
    model.config._attn_implementation = "eager"
    model.model.config._attn_implementation = "eager"
    model.eval()
    return tri_llama, model


def _make_single_turn_batch():
    input_ids = torch.tensor([[10, 11, 20, 21, 30, 31, 40, 41]], dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :6] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "assistant_header_start": torch.tensor([4], dtype=torch.long),
        dt.ASSISTANT_HEADER_STARTS_KEY: torch.tensor([[4]], dtype=torch.long),
        dt.ASSISTANT_HEADER_START_MASK_KEY: torch.tensor([[True]], dtype=torch.bool),
        PREFILL_LOWER_SYSTEM_LEN_KEY: torch.tensor([2], dtype=torch.long),
    }


def _make_multi_turn_batch():
    # [system x2][user1 x2][assistant1 header x2][assistant1 content x2][user2 x2][assistant2 header x2][assistant2 content x2]
    input_ids = torch.tensor([[10, 11, 20, 21, 30, 31, 40, 41, 50, 51, 60, 61, 70, 71]], dtype=torch.long)
    labels = input_ids.clone()
    labels[:, :6] = -100
    labels[:, 8:12] = -100
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "assistant_header_start": torch.tensor([10], dtype=torch.long),
        dt.ASSISTANT_HEADER_STARTS_KEY: torch.tensor([[4, 10]], dtype=torch.long),
        dt.ASSISTANT_HEADER_START_MASK_KEY: torch.tensor([[True, True]], dtype=torch.bool),
        PREFILL_LOWER_SYSTEM_LEN_KEY: torch.tensor([2], dtype=torch.long),
    }


def _run_lower_stack(tri_llama, model, batch, lower_k: int):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"].to(dtype=torch.long)
    inputs_embeds = model.model.embed_tokens(input_ids)
    cache_position = tri_llama._tri_arange(0, input_ids.size(1), input_ids.device)
    position_ids = tri_llama._llopa_position_ids_from_mask(attention_mask)
    lower_mask = tri_llama.create_causal_mask(
        config=model.model.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids,
    )
    hidden_states = inputs_embeds
    position_embeddings = model.model.rotary_emb(hidden_states, position_ids)
    for layer_idx in range(lower_k):
        hidden_states = model.model.layers[layer_idx](
            hidden_states,
            attention_mask=lower_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
    return hidden_states, position_ids


def _prepare_multiturn_upper_batch(tri_llama, model, batch, lower_k: int, system_prefill: str = "full"):
    hidden_states, position_ids = _run_lower_stack(tri_llama, model, batch, lower_k=lower_k)
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]
    valid_lens = attention_mask.sum(dim=1, dtype=torch.long)
    system_lens = batch[PREFILL_LOWER_SYSTEM_LEN_KEY]
    header_starts = batch[dt.ASSISTANT_HEADER_STARTS_KEY]
    header_mask = batch[dt.ASSISTANT_HEADER_START_MASK_KEY]

    turn_sample_rows = []
    turn_starts = []
    turn_ends = []
    for row in range(labels.size(0)):
        spans = tri_llama._tri_resolve_assistant_turn_spans(
            labels_row=labels[row],
            valid_len=int(valid_lens[row].item()),
            assistant_header_starts=header_starts[row],
            assistant_header_start_mask=header_mask[row],
        )
        for start, end in spans:
            turn_sample_rows.append(row)
            turn_starts.append(start)
            turn_ends.append(end)

    turn_sample_ids = torch.tensor(turn_sample_rows, dtype=torch.long)
    turn_starts = torch.tensor(turn_starts, dtype=torch.long)
    turn_ends = torch.tensor(turn_ends, dtype=torch.long)
    turn_system_lens = system_lens.index_select(0, turn_sample_ids)
    upper_gather_idx, upper_valid_mask, _ = tri_llama._tri_build_prefill_lower_multiturn_index_batch(
        turn_starts=turn_starts,
        turn_ends=turn_ends,
        system_lens=turn_system_lens,
        system_prefill=system_prefill,
        device=hidden_states.device,
    )
    upper_hidden, _ = tri_llama._tri_pack_indexed_tensor(
        hidden_states.index_select(0, turn_sample_ids),
        gather_idx=upper_gather_idx,
        valid_mask=upper_valid_mask,
        pad_value=0.0,
    )
    upper_position_ids, _ = tri_llama._tri_pack_indexed_tensor(
        position_ids.index_select(0, turn_sample_ids),
        gather_idx=upper_gather_idx,
        valid_mask=upper_valid_mask,
        pad_value=0,
    )
    decode_labels, _ = tri_llama._tri_pack_indexed_tensor(
        labels.index_select(0, turn_sample_ids),
        gather_idx=upper_gather_idx,
        valid_mask=upper_valid_mask,
        pad_value=-100,
    )
    upper_attention_mask = upper_valid_mask.to(dtype=attention_mask.dtype)
    return upper_hidden, upper_position_ids, upper_attention_mask, decode_labels


def _run_upper_stack(tri_llama, model, upper_hidden, upper_position_ids, upper_attention_mask, lower_k: int):
    cache_position = tri_llama._tri_arange(0, upper_hidden.size(1), upper_hidden.device)
    upper_mask = tri_llama.create_causal_mask(
        config=model.model.config,
        input_embeds=upper_hidden,
        attention_mask=upper_attention_mask,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=upper_position_ids,
    )
    upper_pos_emb = model.model.rotary_emb(upper_hidden, upper_position_ids)
    hidden = upper_hidden
    for layer_idx in range(lower_k, len(model.model.layers)):
        hidden = model.model.layers[layer_idx](
            hidden,
            attention_mask=upper_mask,
            position_ids=upper_position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=upper_pos_emb,
        )
    return model.model.norm(hidden)


def test_sft_tulu_tokenize_tracks_valid_assistant_header_starts_and_truncation():
    tokenizer = DummyTokenizer()
    messages = [
        {"role": "user", "content": "question1"},
        {"role": "assistant", "content": "answer1"},
        {"role": "user", "content": "question2"},
        {"role": "assistant", "content": "answer2"},
    ]

    row = dt.sft_tulu_tokenize_and_truncate_v1({"messages": messages}, tokenizer, max_seq_length=10_000)
    expected_starts = [
        tokenizer.apply_chat_template(messages[:1], tokenize=True, return_tensors="pt").shape[1],
        tokenizer.apply_chat_template(messages[:3], tokenize=True, return_tensors="pt").shape[1],
    ]
    assert row[dt.ASSISTANT_HEADER_STARTS_KEY] == expected_starts

    second_header = expected_starts[1]
    truncated = dt.sft_tulu_tokenize_and_truncate_v1(
        {"messages": messages},
        tokenizer,
        max_seq_length=second_header + len("<assistant>"),
    )
    assert truncated[dt.ASSISTANT_HEADER_STARTS_KEY] == [expected_starts[0]]


def test_multiturn_upper_index_builder_respects_system_prefill_modes():
    tri_llama = _tri_module()
    turn_starts = torch.tensor([5, 12], dtype=torch.long)
    turn_ends = torch.tensor([9, 16], dtype=torch.long)
    system_lens = torch.tensor([2, 2], dtype=torch.long)

    expected = {
        "full": [[0, 1, 5, 6, 7, 8], [0, 1, 12, 13, 14, 15]],
        "no_system": [[0, 5, 6, 7, 8], [0, 12, 13, 14, 15]],
        "no_bos_system": [[5, 6, 7, 8], [12, 13, 14, 15]],
    }

    for mode, expected_rows in expected.items():
        gather_idx, valid_mask, _ = tri_llama._tri_build_prefill_lower_multiturn_index_batch(
            turn_starts=turn_starts,
            turn_ends=turn_ends,
            system_lens=system_lens,
            system_prefill=mode,
            device=torch.device("cpu"),
        )
        actual_rows = []
        for row_idx in range(gather_idx.size(0)):
            actual_rows.append(gather_idx[row_idx][valid_mask[row_idx]].tolist())
        assert actual_rows == expected_rows


def test_repack_upper_with_suffix_specials_inserts_fusion_tokens_before_decode():
    tri_llama, model = _make_tiny_model()
    model.config.capsule_num_suffix_specials = 2
    model.config.capsule_suffix_special_token_ids = [250, 251]

    hidden_size = model.config.hidden_size
    upper_hidden = torch.zeros((2, 5, hidden_size), dtype=torch.float32)
    upper_hidden[0, 0, 0] = 1.0
    upper_hidden[0, 1, 0] = 2.0
    upper_hidden[0, 2, 0] = 3.0
    upper_hidden[1, 0, 0] = 4.0
    upper_hidden[1, 1, 0] = 5.0
    upper_hidden[1, 2, 0] = 6.0
    upper_hidden[1, 3, 0] = 7.0
    upper_hidden[1, 4, 0] = 8.0

    upper_position_ids = torch.tensor(
        [
            [0, 10, 11, 0, 0],
            [0, 1, 12, 13, 14],
        ],
        dtype=torch.long,
    )
    upper_attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.long,
    )
    decode_labels = torch.tensor(
        [
            [-100, 100, 101, -100, -100],
            [-100, -100, 200, 201, 202],
        ],
        dtype=torch.long,
    )
    prefix_keep_lens = torch.tensor([1, 2], dtype=torch.long)
    split_starts = torch.tensor([10, 12], dtype=torch.long)

    repacked_hidden, repacked_position_ids, repacked_attention_mask, repacked_labels, repacked_valid_lens = (
        tri_llama._tri_repack_upper_with_suffix_specials(
            model,
            upper_hidden=upper_hidden,
            upper_position_ids=upper_position_ids,
            upper_attention_mask=upper_attention_mask,
            decode_labels=decode_labels,
            prefix_keep_lens=prefix_keep_lens,
            split_starts=split_starts,
        )
    )

    expected_specials = model.model.embed_tokens(torch.tensor([[250, 251]], dtype=torch.long)).squeeze(0)

    assert repacked_hidden.shape == (2, 7, hidden_size)
    assert repacked_valid_lens.tolist() == [5, 7]
    assert repacked_attention_mask[0].tolist() == [1, 1, 1, 1, 1, 0, 0]
    assert repacked_attention_mask[1].tolist() == [1, 1, 1, 1, 1, 1, 1]
    assert repacked_position_ids[0].tolist() == [0, 8, 9, 10, 11, 0, 0]
    assert repacked_position_ids[1].tolist() == [0, 1, 10, 11, 12, 13, 14]
    assert repacked_labels[0].tolist() == [-100, -100, -100, 100, 101, -100, -100]
    assert repacked_labels[1].tolist() == [-100, -100, -100, -100, 200, 201, 202]
    torch.testing.assert_close(repacked_hidden[0, 1:3], expected_specials)
    torch.testing.assert_close(repacked_hidden[1, 2:4], expected_specials)


def test_prefill_decode_with_suffix_specials_extends_only_upper_cache():
    tri_llama, model = _make_tiny_model()
    model.config.capsule_num_suffix_specials = 2
    model.config.capsule_suffix_special_token_ids = [250, 251]
    batch = _make_single_turn_batch()

    with torch.no_grad():
        outputs = model(
            **batch,
            use_cache=True,
            prefill_lower_layers=1,
            prefill_lower_attn="causal",
            prefill_lower_system_prefill="full",
        )

    assert outputs.past_key_values is not None
    assert tri_llama._layer_past_len(outputs.past_key_values, 0) == 8
    assert tri_llama._layer_past_len(outputs.past_key_values, 1) == 8


def test_insert_suffix_specials_inband_single_turn_updates_labels_and_split_start():
    tri_llama, _ = _make_tiny_model()
    batch = _make_single_turn_batch()

    (
        new_input_ids,
        new_attention_mask,
        new_labels,
        new_split_starts,
        new_header_starts,
        new_header_mask,
    ) = tri_llama._tri_insert_suffix_specials_inband(
        token_ids=[250, 251],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        split_starts=batch["assistant_header_start"],
        assistant_header_starts=batch[dt.ASSISTANT_HEADER_STARTS_KEY],
        assistant_header_start_mask=batch[dt.ASSISTANT_HEADER_START_MASK_KEY],
    )

    assert new_input_ids.tolist() == [[10, 11, 20, 21, 250, 251, 30, 31, 40, 41]]
    assert new_attention_mask.tolist() == [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    assert new_labels.tolist() == [[-100, -100, -100, -100, -100, -100, -100, -100, 40, 41]]
    assert new_split_starts.tolist() == [4]
    assert new_header_starts.tolist() == [[4]]
    assert new_header_mask.tolist() == [[True]]


def test_insert_suffix_specials_inband_multiturn_remaps_turn_starts():
    tri_llama, _ = _make_tiny_model()
    batch = _make_multi_turn_batch()

    (
        new_input_ids,
        new_attention_mask,
        new_labels,
        new_split_starts,
        new_header_starts,
        new_header_mask,
    ) = tri_llama._tri_insert_suffix_specials_inband(
        token_ids=[250, 251],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
        split_starts=batch["assistant_header_start"],
        assistant_header_starts=batch[dt.ASSISTANT_HEADER_STARTS_KEY],
        assistant_header_start_mask=batch[dt.ASSISTANT_HEADER_START_MASK_KEY],
    )

    assert new_input_ids.tolist() == [[10, 11, 20, 21, 250, 251, 30, 31, 40, 41, 50, 51, 250, 251, 60, 61, 70, 71]]
    assert new_attention_mask.tolist() == [[1] * 18]
    assert new_labels.tolist() == [[-100, -100, -100, -100, -100, -100, -100, -100, 40, 41, -100, -100, -100, -100, -100, -100, 70, 71]]
    assert new_split_starts.tolist() == [12]
    assert new_header_starts.tolist() == [[4, 12]]
    assert new_header_mask.tolist() == [[True, True]]


def test_unified_inband_single_turn_matches_manual_mutated_upper_path():
    tri_llama, model = _make_tiny_model()
    model.config.capsule_num_suffix_specials = 2
    model.config.capsule_suffix_special_token_ids = [250, 251]
    model.config.capsule_fusion_mode = "inband"
    batch = _make_single_turn_batch()

    with torch.no_grad():
        one_shot = model(
            **batch,
            use_cache=False,
            prefill_lower_layers=1,
            prefill_lower_attn="causal",
            prefill_lower_system_prefill="full",
        )
        (
            mutated_input_ids,
            mutated_attention_mask,
            mutated_labels,
            _,
            _,
            _,
        ) = tri_llama._tri_insert_suffix_specials_inband(
            token_ids=[250, 251],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            split_starts=batch["assistant_header_start"],
            assistant_header_starts=batch[dt.ASSISTANT_HEADER_STARTS_KEY],
            assistant_header_start_mask=batch[dt.ASSISTANT_HEADER_START_MASK_KEY],
        )
        mutated_batch = {
            "input_ids": mutated_input_ids,
            "attention_mask": mutated_attention_mask,
            "labels": mutated_labels,
            PREFILL_LOWER_SYSTEM_LEN_KEY: batch[PREFILL_LOWER_SYSTEM_LEN_KEY],
        }
        hidden_states, position_ids = _run_lower_stack(tri_llama, model, mutated_batch, lower_k=1)
        valid_lens = mutated_attention_mask.sum(dim=1, dtype=torch.long)
        split_starts = torch.tensor([4], dtype=torch.long)
        system_lens = mutated_batch[PREFILL_LOWER_SYSTEM_LEN_KEY]
        upper_gather_idx, upper_valid_mask, _ = tri_llama._tri_build_prefill_lower_upper_index_batch(
            split_starts=split_starts,
            valid_lens=valid_lens,
            system_lens=system_lens,
            system_prefill="full",
            device=hidden_states.device,
        )
        upper_hidden, _ = tri_llama._tri_pack_indexed_tensor(
            hidden_states,
            gather_idx=upper_gather_idx,
            valid_mask=upper_valid_mask,
            pad_value=0.0,
        )
        upper_position_ids, _ = tri_llama._tri_pack_indexed_tensor(
            position_ids,
            gather_idx=upper_gather_idx,
            valid_mask=upper_valid_mask,
            pad_value=0,
        )
        decode_labels, _ = tri_llama._tri_pack_indexed_tensor(
            mutated_labels,
            gather_idx=upper_gather_idx,
            valid_mask=upper_valid_mask,
            pad_value=-100,
        )
        upper_attention_mask = upper_valid_mask.to(dtype=mutated_attention_mask.dtype)
        batched_hidden = _run_upper_stack(
            tri_llama,
            model,
            upper_hidden,
            upper_position_ids,
            upper_attention_mask,
            lower_k=1,
        )
        batched_logits = model.lm_head(batched_hidden)
        batched_loss = model.loss_function(logits=batched_logits, labels=decode_labels, vocab_size=model.config.vocab_size)

    torch.testing.assert_close(one_shot.loss, batched_loss)


def test_prefill_decode_with_inband_fusion_uses_mutated_sequence_without_upper_only_cache_write():
    tri_llama, model = _make_tiny_model()
    model.config.capsule_num_suffix_specials = 2
    model.config.capsule_suffix_special_token_ids = [250, 251]
    model.config.capsule_fusion_mode = "inband"
    batch = _make_single_turn_batch()

    with torch.no_grad():
        outputs = model(
            **batch,
            use_cache=True,
            prefill_lower_layers=1,
            prefill_lower_attn="causal",
            prefill_lower_system_prefill="full",
        )

    assert outputs.past_key_values is not None
    assert not bool(getattr(outputs.past_key_values, "_capsule_suffix_specials_written", False))
    assert tri_llama._layer_past_len(outputs.past_key_values, 0) == 10
    assert tri_llama._layer_past_len(outputs.past_key_values, 1) == 8


def test_unified_generate_inband_prefill_matches_runtime_prefill_reference():
    _, model = _make_tiny_model()
    model.config.capsule_num_suffix_specials = 2
    model.config.capsule_suffix_special_token_ids = [250, 251]
    model.config.capsule_fusion_mode = "inband"
    tokenizer = DummyTokenizer()
    llopa_inference = _llopa_inference_module()

    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    bundle = llopa_inference._build_unified_prefill_lower_prompt_bundle(
        tokenizer,
        prompt_messages=messages,
        prompt_add_generation_prompt=True,
        structured_prompt_segments=None,
        device=torch.device("cpu"),
    )

    with torch.no_grad():
        unified_out = llopa_inference._direct_llopa_generate_impl(
            model,
            tokenizer,
            prompt_messages=messages,
            prompt_add_generation_prompt=True,
            structured_prompt_segments=bundle["segments"],
            input_ids=bundle["prompt_ids"],
            attention_mask=bundle["attention_mask"],
            lower_k=1,
            prefill_attn="causal",
            system_prefill="full",
            user_prefill="full",
            no_upper_attn=False,
            max_new_tokens=1,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
            use_cache=True,
        )
        direct_scores = unified_out.scores
        assert direct_scores is not None and len(direct_scores) == 1
        direct_logits = direct_scores[0]

        seed = model.llopa_reference_prefill_seed(
            system_ids=bundle["segments"]["system_ids"],
            user_ids=bundle["segments"]["user_ids"],
            assistant_ids=bundle["segments"]["assistant_prefill_ids"],
            lower_k=1,
            prefill_attn="causal",
            system_prefill="full",
            no_upper_attn=False,
        )
        assert seed is not None
        pkv, S, U, tri_logits = seed
        assert pkv is not None
        assert S >= 0 and U >= 0
        tri_logits = tri_logits.to(torch.float32)

    torch.testing.assert_close(direct_logits, tri_logits)


def test_unified_generate_inband_returns_effective_prompt_prefix():
    tri_llama, model = _make_tiny_model()
    model.config.capsule_num_suffix_specials = 2
    model.config.capsule_suffix_special_token_ids = [250, 251]
    model.config.capsule_fusion_mode = "inband"
    tokenizer = DummyTokenizer()
    llopa_inference = _llopa_inference_module()

    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
        {"role": "assistant", "content": "Answer:"},
    ]
    bundle = llopa_inference._build_unified_prefill_lower_prompt_bundle(
        tokenizer,
        prompt_messages=messages,
        prompt_add_generation_prompt=False,
        structured_prompt_segments=None,
        device=torch.device("cpu"),
    )
    mutated_input_ids, mutated_attention_mask, _, _, _, _ = tri_llama._tri_insert_suffix_specials_inband(
        token_ids=[250, 251],
        input_ids=bundle["prompt_ids"],
        attention_mask=bundle["attention_mask"],
        labels=None,
        split_starts=bundle["prefill_lower_split_start"],
        assistant_header_starts=bundle["assistant_header_starts"],
        assistant_header_start_mask=bundle["assistant_header_start_mask"],
    )

    with torch.no_grad():
        out = llopa_inference._direct_llopa_generate_impl(
            model,
            tokenizer,
            prompt_messages=messages,
            prompt_add_generation_prompt=False,
            structured_prompt_segments=bundle["segments"],
            input_ids=mutated_input_ids,
            attention_mask=mutated_attention_mask,
            lower_k=1,
            prefill_attn="causal",
            system_prefill="full",
            user_prefill="full",
            no_upper_attn=False,
            max_new_tokens=1,
            do_sample=False,
            output_scores=False,
            return_dict_in_generate=True,
            use_cache=True,
        )

    assert out is not None
    torch.testing.assert_close(
        out.sequences[0, : mutated_input_ids.size(1)],
        mutated_input_ids[0],
    )


def test_unified_generate_inband_multiturn_uses_provided_input_ids_prefix():
    _, model = _make_tiny_model()
    model.config.capsule_num_suffix_specials = 2
    model.config.capsule_suffix_special_token_ids = [250, 251]
    model.config.capsule_fusion_mode = "inband"
    tokenizer = DummyTokenizer()
    llopa_inference = _llopa_inference_module()

    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "Answer:"},
    ]
    bundle = llopa_inference._build_unified_prefill_lower_prompt_bundle(
        tokenizer,
        prompt_messages=messages,
        prompt_add_generation_prompt=False,
        structured_prompt_segments=None,
        device=torch.device("cpu"),
    )
    split_start = int(bundle["prefill_lower_split_start"][0].item())
    suffix_ids = torch.tensor([[250, 251]], dtype=bundle["prompt_ids"].dtype)
    mutated_input_ids = torch.cat(
        [
            bundle["prompt_ids"][:, :split_start],
            suffix_ids,
            bundle["prompt_ids"][:, split_start:],
        ],
        dim=1,
    )
    mutated_attention_mask = torch.ones_like(mutated_input_ids, dtype=torch.long)

    with torch.no_grad():
        out = llopa_inference._direct_llopa_generate_impl(
            model,
            tokenizer,
            prompt_messages=messages,
            prompt_add_generation_prompt=False,
            structured_prompt_segments=bundle["segments"],
            input_ids=mutated_input_ids,
            attention_mask=mutated_attention_mask,
            lower_k=1,
            prefill_attn="causal",
            system_prefill="full",
            user_prefill="full",
            no_upper_attn=False,
            max_new_tokens=1,
            do_sample=False,
            output_scores=False,
            return_dict_in_generate=True,
            use_cache=True,
        )

    assert out is not None
    torch.testing.assert_close(
        out.sequences[0, : mutated_input_ids.size(1)],
        mutated_input_ids[0],
    )


def test_single_turn_metadata_does_not_change_prefill_lower_loss():
    _, model = _make_tiny_model()
    batch = _make_single_turn_batch()

    with torch.no_grad():
        old_path = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            use_cache=False,
            prefill_lower_layers=1,
            prefill_lower_attn="causal",
            prefill_lower_system_prefill="full",
            assistant_header_start=batch["assistant_header_start"],
            prefill_lower_system_len=batch[PREFILL_LOWER_SYSTEM_LEN_KEY],
        )
        new_path = model(
            **batch,
            use_cache=False,
            prefill_lower_layers=1,
            prefill_lower_attn="causal",
            prefill_lower_system_prefill="full",
        )

    torch.testing.assert_close(new_path.loss, old_path.loss)


def test_multiturn_one_shot_matches_sequential_upper_loss():
    tri_llama, model = _make_tiny_model()
    batch = _make_multi_turn_batch()

    with torch.no_grad():
        one_shot = model(
            **batch,
            use_cache=False,
            prefill_lower_layers=1,
            prefill_lower_attn="causal",
            prefill_lower_system_prefill="full",
        )
        upper_hidden, upper_position_ids, upper_attention_mask, decode_labels = _prepare_multiturn_upper_batch(
            tri_llama,
            model,
            batch,
            lower_k=1,
            system_prefill="full",
        )

        batched_hidden = _run_upper_stack(
            tri_llama,
            model,
            upper_hidden.clone(),
            upper_position_ids,
            upper_attention_mask,
            lower_k=1,
        )
        batched_logits = model.lm_head(batched_hidden)
        batched_loss = model.loss_function(logits=batched_logits, labels=decode_labels, vocab_size=model.config.vocab_size)

        token_weighted_sum = torch.zeros((), dtype=batched_loss.dtype)
        total_effective_tokens = 0
        for row_idx in range(upper_hidden.size(0)):
            row_hidden = _run_upper_stack(
                tri_llama,
                model,
                upper_hidden[row_idx : row_idx + 1].clone(),
                upper_position_ids[row_idx : row_idx + 1],
                upper_attention_mask[row_idx : row_idx + 1],
                lower_k=1,
            )
            row_logits = model.lm_head(row_hidden)
            row_labels = decode_labels[row_idx : row_idx + 1]
            row_loss = model.loss_function(logits=row_logits, labels=row_labels, vocab_size=model.config.vocab_size)
            effective_tokens = int((row_labels[:, 1:] != -100).sum().item())
            token_weighted_sum = token_weighted_sum + row_loss * effective_tokens
            total_effective_tokens += effective_tokens

        sequential_loss = token_weighted_sum / total_effective_tokens

    torch.testing.assert_close(one_shot.loss, batched_loss)
    torch.testing.assert_close(one_shot.loss, sequential_loss, rtol=1e-5, atol=1e-6)


def test_single_turn_prefill_lower_gradients_reach_prefix_tokens_and_both_stacks():
    _, model = _make_tiny_model()
    model.train()
    batch = _make_single_turn_batch()

    outputs = model(
        **batch,
        use_cache=False,
        prefill_lower_layers=1,
        prefill_lower_attn="causal",
        prefill_lower_system_prefill="full",
    )
    outputs.loss.backward()

    embed_grad = model.model.embed_tokens.weight.grad
    assert embed_grad is not None
    assert float(embed_grad[10].norm().item()) > 0.0  # system token
    assert float(embed_grad[20].norm().item()) > 0.0  # user token
    assert float(embed_grad[30].norm().item()) > 0.0  # assistant header token
    assert float(embed_grad[40].norm().item()) > 0.0  # supervised assistant token
    assert float(model.model.layers[0].self_attn.q_proj.weight.grad.norm().item()) > 0.0
    assert float(model.model.layers[1].self_attn.q_proj.weight.grad.norm().item()) > 0.0


def test_multiturn_prefill_lower_gradients_reach_earlier_prefix_tokens():
    _, model = _make_tiny_model()
    model.train()
    batch = _make_multi_turn_batch()

    outputs = model(
        **batch,
        use_cache=False,
        prefill_lower_layers=1,
        prefill_lower_attn="causal",
        prefill_lower_system_prefill="full",
    )
    outputs.loss.backward()

    embed_grad = model.model.embed_tokens.weight.grad
    assert embed_grad is not None
    assert float(embed_grad[10].norm().item()) > 0.0  # system token
    assert float(embed_grad[20].norm().item()) > 0.0  # first user turn
    assert float(embed_grad[30].norm().item()) > 0.0  # first assistant header
    assert float(embed_grad[50].norm().item()) > 0.0  # second user turn
    assert float(embed_grad[60].norm().item()) > 0.0  # final assistant header
    assert float(model.model.layers[0].self_attn.q_proj.weight.grad.norm().item()) > 0.0
    assert float(model.model.layers[1].self_attn.q_proj.weight.grad.norm().item()) > 0.0


def test_upper_only_fusion_specials_receive_training_gradients():
    _, model = _make_tiny_model()
    model.train()
    model.config.capsule_num_suffix_specials = 2
    model.config.capsule_suffix_special_token_ids = [250, 251]
    model.config.capsule_fusion_mode = "upper_only"
    batch = _make_single_turn_batch()

    outputs = model(
        **batch,
        use_cache=False,
        prefill_lower_layers=1,
        prefill_lower_attn="causal",
        prefill_lower_system_prefill="full",
    )
    outputs.loss.backward()

    embed_grad = model.model.embed_tokens.weight.grad
    assert embed_grad is not None
    assert float(embed_grad[250].norm().item()) > 0.0
    assert float(embed_grad[251].norm().item()) > 0.0


def test_last_layer_module_none_matches_default_unified_forward():
    _, model = _make_tiny_model()
    model.eval()
    batch = _make_single_turn_batch()

    with torch.no_grad():
        default_out = model(
            **batch,
            use_cache=False,
            prefill_lower_layers=1,
            prefill_lower_attn="causal",
            prefill_lower_system_prefill="full",
        )
        none_out = model(
            **batch,
            use_cache=False,
            prefill_lower_layers=1,
            prefill_lower_attn="causal",
            prefill_lower_system_prefill="full",
            prefill_lower_replay_module="none",
        )

    torch.testing.assert_close(default_out.loss, none_out.loss)
    torch.testing.assert_close(default_out.logits, none_out.logits)


def test_replay_layer_index_schedule_matches_upper_layer_counting_rule():
    tri_llama, _ = _make_tiny_model()

    assert tri_llama._tri_replay_layer_index_set(upper_layer_indices=[27, 28, 29, 30, 31], replay_per_layers=-1) == {31}
    assert tri_llama._tri_replay_layer_index_set(upper_layer_indices=[27, 28, 29, 30, 31], replay_per_layers=1) == {27, 28, 29, 30, 31}
    assert tri_llama._tri_replay_layer_index_set(upper_layer_indices=[27, 28, 29, 30, 31], replay_per_layers=2) == {28, 30}
    assert tri_llama._tri_replay_layer_index_set(upper_layer_indices=[27, 28, 29, 30, 31], replay_per_layers=4) == {30}


def test_last_layer_module_self_and_cross_support_training_backward():
    batch = _make_single_turn_batch()

    for mode in ("self", "cross"):
        _, model = _make_tiny_model()
        model.train()
        outputs = model(
            **batch,
            use_cache=False,
            prefill_lower_layers=1,
            prefill_lower_attn="causal",
            prefill_lower_system_prefill="full",
            prefill_lower_replay_module=mode,
        )

        assert outputs.loss is not None
        assert torch.isfinite(outputs.loss)

        outputs.loss.backward()

        final_q_grad = model.model.layers[-1].self_attn.q_proj.weight.grad
        assert final_q_grad is not None
        assert torch.isfinite(final_q_grad).all()
        if mode == "cross":
            final_layer = model.model.layers[-1]
            assert final_layer.replay_cross_attn.q_proj.weight is not final_layer.self_attn.q_proj.weight
            cross_q_grad = final_layer.replay_cross_attn.q_proj.weight.grad
            assert cross_q_grad is not None
            assert torch.isfinite(cross_q_grad).all()


def test_replay_cross_attention_is_seeded_from_self_attention_on_load():
    _, src_model = _make_tiny_model()
    with torch.no_grad():
        src_model.model.layers[-1].self_attn.q_proj.weight.fill_(0.125)
        src_model.model.layers[-1].self_attn.o_proj.weight.fill_(0.25)

    state_dict = src_model.state_dict()
    stripped_state_dict = {
        key: value
        for key, value in state_dict.items()
        if ".replay_cross_attn." not in key
    }

    _, dst_model = _make_tiny_model()
    missing, unexpected = dst_model.load_state_dict(stripped_state_dict, strict=False)
    assert all(".replay_cross_attn." not in key for key in missing)
    assert not unexpected

    final_layer = dst_model.model.layers[-1]
    torch.testing.assert_close(
        final_layer.replay_cross_attn.q_proj.weight,
        final_layer.self_attn.q_proj.weight,
    )
    torch.testing.assert_close(
        final_layer.replay_cross_attn.o_proj.weight,
        final_layer.self_attn.o_proj.weight,
    )


def test_last_layer_module_vanilla_prefill_decode_avoids_duplicate_lower_pass():
    tri_llama, model = _make_tiny_model()
    model.eval()
    batch = _make_single_turn_batch()
    original = tri_llama._tri_prefill_lower_prompt_hidden

    def _unexpected_second_pass(*args, **kwargs):
        raise AssertionError("duplicate lower hidden replay pass should not run")

    tri_llama._tri_prefill_lower_prompt_hidden = _unexpected_second_pass
    try:
        with torch.no_grad():
            outputs = model(
                **batch,
                use_cache=False,
                prefill_lower_layers=1,
                prefill_lower_attn="causal",
                prefill_lower_system_prefill="full",
                prefill_lower_replay_module="self",
            )
    finally:
        tri_llama._tri_prefill_lower_prompt_hidden = original

    assert outputs.loss is not None
    assert torch.isfinite(outputs.loss)


def test_last_layer_module_self_skips_replay_branch():
    tri_llama, model = _make_tiny_model()
    batch = _make_single_turn_batch()
    original_replay = tri_llama._tri_replay_attention_forward

    def _unexpected_replay(*args, **kwargs):
        raise AssertionError("self mode should not invoke the separate replay branch")

    tri_llama._tri_replay_attention_forward = _unexpected_replay
    try:
        outputs = model(
            **batch,
            use_cache=False,
            prefill_lower_layers=1,
            prefill_lower_attn="causal",
            prefill_lower_system_prefill="full",
            prefill_lower_replay_module="self",
        )
    finally:
        tri_llama._tri_replay_attention_forward = original_replay

    assert outputs.loss is not None
    assert torch.isfinite(outputs.loss)


def test_last_layer_module_integrated_self_and_cross_flash_attention_dispatch_shapes():
    tri_llama, model = _make_tiny_model()
    attn = model.model.layers[-1].self_attn
    attn.config._attn_implementation = "flash_attention_2"
    attn.layer_idx = model.config.num_hidden_layers - 1

    hidden_states = torch.randn(1, 4, model.config.hidden_size)
    memory_hidden_states = torch.randn(1, 3, model.config.hidden_size)
    query_position_ids = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    memory_position_ids = torch.tensor([[2, 3, 4]], dtype=torch.long)
    query_pos_emb = model.model.rotary_emb(hidden_states, query_position_ids)
    memory_pos_emb = model.model.rotary_emb(memory_hidden_states, memory_position_ids)
    memory_key_states, memory_value_states = tri_llama._tri_project_memory_kv(
        attn_module=attn,
        memory_hidden_states=memory_hidden_states,
        memory_position_embeddings=memory_pos_emb,
        target_device=hidden_states.device,
        target_dtype=hidden_states.dtype,
    )

    original_flash = tri_llama.ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
    original_resolve = tri_llama._resolve_attn_impl
    calls = []

    def _fake_flash(module, query, key, value, attention_mask, dropout=0.0, scaling=None, **kwargs):
        calls.append(
            {
                "attention_mask_shape": None if attention_mask is None else tuple(attention_mask.shape),
                "has_cu_q": "cu_seq_lens_q" in kwargs,
                "has_cu_k": "cu_seq_lens_k" in kwargs,
                "max_length_q": kwargs.get("max_length_q"),
                "max_length_k": kwargs.get("max_length_k"),
                "is_causal": bool(module.is_causal),
            }
        )
        return query.transpose(1, 2).new_zeros((query.size(0), query.size(2), query.size(1), query.size(3))), None

    tri_llama.ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = _fake_flash
    tri_llama._resolve_attn_impl = lambda config: "flash_attention_2"
    try:
        self_out, _ = attn(
            hidden_states=hidden_states,
            position_embeddings=query_pos_emb,
            attention_mask=None,
            past_key_values=None,
            cache_position=torch.arange(hidden_states.size(1), dtype=torch.long),
            position_ids=query_position_ids,
            extra_prefix_kv=(memory_key_states, memory_value_states),
            extra_prefix_valid_mask=torch.ones((1, 3), dtype=torch.bool),
            extra_prefix_query_mask=torch.ones((1, 4), dtype=torch.bool),
            extra_prefix_local_valid_mask=torch.ones((1, 4), dtype=torch.bool),
        )
        cross_out = tri_llama._tri_replay_attention_forward(
            attn_module=attn,
            hidden_states=hidden_states,
            position_embeddings=query_pos_emb,
            local_valid_mask=torch.ones((1, 4), dtype=torch.bool),
            query_replay_mask=torch.ones((1, 4), dtype=torch.bool),
            memory_hidden_states=memory_hidden_states,
            memory_position_embeddings=memory_pos_emb,
            memory_valid_mask=torch.ones((1, 3), dtype=torch.bool),
            module_type="cross",
        )
    finally:
        tri_llama.ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = original_flash
        tri_llama._resolve_attn_impl = original_resolve

    assert self_out.shape == hidden_states.shape
    assert cross_out.shape == hidden_states.shape
    assert len(calls) == 2
    assert calls[0]["attention_mask_shape"] is None
    assert calls[0]["has_cu_q"] is False
    assert calls[0]["is_causal"] is True
    assert calls[1]["attention_mask_shape"] is None
    assert calls[1]["has_cu_q"] is True
    assert calls[1]["has_cu_k"] is True
    assert calls[1]["max_length_q"] == 4
    assert calls[1]["max_length_k"] == 3
    assert calls[1]["is_causal"] is False
