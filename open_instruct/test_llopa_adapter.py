import types

import torch

from open_instruct import llopa_adapter


class _DummyEncoding:
    def __init__(self, input_ids: torch.Tensor):
        self.input_ids = input_ids


class DummyTokenizer:
    pad_token_id = 0
    chat_template = "dummy"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert not tokenize
        parts = []
        for msg in messages:
            parts.append(f"<{msg['role']}>{msg['content']}</{msg['role']}>")
        if add_generation_prompt:
            parts.append("<assistant>")
        return "".join(parts)

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        assert not add_special_tokens
        assert return_tensors == "pt"
        ids = torch.tensor([[ord(ch) + 1 for ch in text]], dtype=torch.long)
        return _DummyEncoding(ids)


def _sample_messages():
    return [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "final answer"},
    ]


def _sample_messages_2():
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "longer question text"},
        {"role": "assistant", "content": "different answer"},
    ]


def _sample_messages_no_system():
    return [
        {"role": "user", "content": "question only"},
        {"role": "assistant", "content": "answer only"},
    ]


def _base_batch(messages_batch):
    batch_size = len(messages_batch)
    return {
        "input_ids": torch.ones((batch_size, 4), dtype=torch.long),
        "attention_mask": torch.ones((batch_size, 4), dtype=torch.long),
        "labels": torch.ones((batch_size, 4), dtype=torch.long),
        "messages": messages_batch,
    }


def test_build_last_turn_example_masks_assistant_header_only():
    tokenizer = DummyTokenizer()
    example = llopa_adapter._build_llopa_last_turn_example(
        tokenizer,
        _sample_messages(),
        system_prefill="no_bos_system",
        user_prefill="full",
    )

    assert example is not None
    assistant_ids = example[llopa_adapter.LLOPA_ASSISTANT_IDS_KEY]
    labels = example[llopa_adapter.LLOPA_LABELS_KEY]
    prefix_len = int((labels == -100).sum().item())

    assert prefix_len > 0
    assert torch.equal(labels[:, prefix_len:], assistant_ids[:, prefix_len:])
    assert torch.all(labels[:, :prefix_len] == -100)


def test_compute_llopa_batch_loss_batched_last_turn_matches_slow_path(monkeypatch):
    tokenizer = DummyTokenizer()
    messages_batch = [_sample_messages(), _sample_messages_2()]
    slow_batch = _base_batch(messages_batch)
    fast_batch = dict(slow_batch)
    fast_batch.update(
        llopa_adapter._batch_llopa_last_turn_examples(
            tokenizer,
            messages_batch,
            system_prefill="no_bos_system",
            user_prefill="full",
        )
    )

    def fake_step_fn(**kwargs):
        labels = kwargs["labels"]
        valid_mask = labels != -100
        per_sample = (labels.float() * valid_mask.float()).sum(dim=1) / 1000.0
        loss = per_sample.mean()
        return types.SimpleNamespace(loss=loss, logits=None, hidden_states=None)

    monkeypatch.setattr(llopa_adapter, "_get_llopa_step_fn", lambda model: fake_step_fn)

    slow_loss = llopa_adapter.compute_llopa_batch_loss(
        model=object(),
        tokenizer=tokenizer,
        batch=slow_batch,
        lower_k=28,
        prefill_mode="lower",
        prefill_attn="causal",
        system_prefill="no_bos_system",
        user_prefill="full",
        no_upper_attn=False,
        loss_scope="last_turn",
    )
    fast_loss = llopa_adapter.compute_llopa_batch_loss(
        model=object(),
        tokenizer=tokenizer,
        batch=fast_batch,
        lower_k=28,
        prefill_mode="lower",
        prefill_attn="causal",
        system_prefill="no_bos_system",
        user_prefill="full",
        no_upper_attn=False,
        loss_scope="last_turn",
    )

    assert torch.isclose(slow_loss, fast_loss)


def test_compute_llopa_batch_loss_warns_once_when_batched_tensors_missing(monkeypatch):
    tokenizer = DummyTokenizer()
    batch = _base_batch([_sample_messages()])
    warnings = []

    def fake_warning(message, *args, **kwargs):
        warnings.append(message % args if args else message)

    def fake_step_fn(**kwargs):
        labels = kwargs["labels"]
        valid_mask = labels != -100
        per_sample = (labels.float() * valid_mask.float()).sum(dim=1) / 1000.0
        loss = per_sample.mean()
        return types.SimpleNamespace(loss=loss, logits=None, hidden_states=None)

    monkeypatch.setattr(llopa_adapter.logger, "warning", fake_warning)
    monkeypatch.setattr(llopa_adapter, "_get_llopa_step_fn", lambda model: fake_step_fn)
    monkeypatch.setattr(llopa_adapter, "_WARNED_BATCHED_LAST_TURN_FALLBACK", False)

    llopa_adapter.compute_llopa_batch_loss(
        model=object(),
        tokenizer=tokenizer,
        batch=batch,
        lower_k=28,
        prefill_mode="lower",
        prefill_attn="causal",
        system_prefill="no_bos_system",
        user_prefill="full",
        no_upper_attn=False,
        loss_scope="last_turn",
    )
    llopa_adapter.compute_llopa_batch_loss(
        model=object(),
        tokenizer=tokenizer,
        batch=batch,
        lower_k=28,
        prefill_mode="lower",
        prefill_attn="causal",
        system_prefill="no_bos_system",
        user_prefill="full",
        no_upper_attn=False,
        loss_scope="last_turn",
    )

    assert len(warnings) == 1
    assert "prebatched segment tensors" in warnings[0]


def test_get_prefill_lower_system_len_matches_tokenized_system_prefix():
    tokenizer = DummyTokenizer()
    expected = llopa_adapter._tokens_from_messages(
        tokenizer,
        [{"role": "system", "content": "system prompt"}],
        torch.device("cpu"),
        add_generation_prompt=False,
    ).size(1)

    system_len = llopa_adapter.get_prefill_lower_system_len(
        tokenizer,
        _sample_messages(),
        split_start=999,
        sequence_len=999,
    )
    clamped = llopa_adapter.get_prefill_lower_system_len(
        tokenizer,
        _sample_messages(),
        split_start=3,
        sequence_len=999,
    )
    no_system = llopa_adapter.get_prefill_lower_system_len(
        tokenizer,
        _sample_messages_no_system(),
        split_start=999,
        sequence_len=999,
    )

    assert system_len == expected
    assert clamped == 3
    assert no_system == 0


def test_build_prefill_lower_upper_indices_supports_system_prefill_modes():
    device = torch.device("cpu")

    full_idx = llopa_adapter.build_prefill_lower_upper_indices(
        sequence_len=10,
        split_start=7,
        system_len=4,
        system_prefill="full",
        device=device,
    )
    no_system_idx = llopa_adapter.build_prefill_lower_upper_indices(
        sequence_len=10,
        split_start=7,
        system_len=4,
        system_prefill="no_system",
        device=device,
    )
    no_bos_idx = llopa_adapter.build_prefill_lower_upper_indices(
        sequence_len=10,
        split_start=7,
        system_len=4,
        system_prefill="no_bos_system",
        device=device,
    )

    assert torch.equal(full_idx, torch.tensor([0, 1, 2, 3, 7, 8, 9], dtype=torch.long))
    assert torch.equal(no_system_idx, torch.tensor([0, 7, 8, 9], dtype=torch.long))
    assert torch.equal(no_bos_idx, torch.tensor([7, 8, 9], dtype=torch.long))
