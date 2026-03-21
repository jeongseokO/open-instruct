from __future__ import annotations

from pathlib import Path

import torch
from transformers import LlamaConfig

from open_instruct import dataset_transformation as dt
from open_instruct.llopa_adapter import PREFILL_LOWER_SYSTEM_LEN_KEY, install_llopa_modeling


class DummyTokenizer:
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


def _tri_module():
    repo_root = Path(__file__).resolve().parents[2]
    modeling_path = repo_root / "Capsule" / "tri_llama3_modeling.py"
    install_llopa_modeling(str(modeling_path), model_family="llama")
    import transformers.models.llama.modeling_llama as tri_llama

    return tri_llama


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
