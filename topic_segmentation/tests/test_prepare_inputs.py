"""Window making: output shape, worker serialization, seeded augmentation, and error handling."""
import pickle
import random
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch.nn import functional as F

from topic_segmentation.objectives import CombinedObjective, TSSP
from topic_segmentation.prepare_inputs import WindowMaker, pack_windows

LABEL_IDS = {"B-EOP": 0, "O": 1}


class TinyTokenizer:
    """Whitespace tokenizer with the attributes WindowMaker reads."""
    bos_token = "[BOS]"
    bos_token_id = 1
    cls_token_id = 2
    pad_token_id = 0

    def __init__(self):
        self.vocab = {}

    def __call__(self, documents, **kwargs):
        result = {name: [] for name in ("input_ids", "token_type_ids", "attention_mask")}
        for index, sentences in enumerate(documents):
            tokens = " ".join(sentences).replace(self.bos_token, self.bos_token + " ").split()
            ids = [1 if token == self.bos_token else
                   self.vocab.setdefault(token, 1000 + len(self.vocab)) for token in tokens]
            result["input_ids"].append(ids)
            result["token_type_ids"].append([index % 2] * len(ids))
            result["attention_mask"].append([1] * len(ids))
        return result


def window_maker(seed=42, max_length=16, tokenizer=None):
    return WindowMaker(tokenizer or TinyTokenizer(), LABEL_IDS, {1}, max_length, seed)


def batch(first_id=10):
    # Topics of different sizes exercise replacement from other documents, truncation, padding, and overlap.
    documents = [
        (["a b", "c", "d e f", "g", "h i", "j"], ["O", "B-EOP", "O", "O", "B-EOP", "B-EOP"]),
        (["k l m n o p", "q r", "s", "t u v w"], ["B-EOP", "O", "O", "B-EOP"]),
        (["heading", "x y", "z a b", "c d e f", "g h"], ["-100", "O", "B-EOP", "O", "B-EOP"]),
        (["one two", "three", "four five", "six", "seven", "eight nine"], ["O", "B-EOP", "B-EOP", "O", "O", "B-EOP"]),
    ]
    return {"sentences": [d[0] for d in documents], "labels": [d[1] for d in documents],
            "example_id": list(range(first_id, first_id + len(documents)))}


def test_windows_carry_example_ids_and_fixed_length():
    output = window_maker(max_length=8)({"sentences": [["a", "b"]], "labels": [["O", "B-EOP"]], "example_id": [7]})
    assert output["example_id"] == [[7, 7]]
    assert len(output["input_ids"][0][0]) == 8


def test_window_maker_can_be_serialized_for_dataset_workers():
    prepare = window_maker()
    assert pickle.loads(pickle.dumps(prepare))(batch()) == prepare(batch())


def test_same_seed_gives_the_same_augmentation():
    prepare = window_maker(seed=42)
    first = prepare(batch())
    random.seed(0)  # a worker's global random state must not matter
    random.random()
    assert prepare(batch()) == first


def test_seed_and_batch_change_the_augmentation():
    expected = window_maker(seed=42)(batch())["sentences"]
    assert window_maker(seed=43)(batch())["sentences"] != expected
    assert window_maker(seed=42)(batch(first_id=20))["sentences"] != expected


def test_tokenizer_error_is_raised():
    tokenizer = mock.Mock(side_effect=ValueError("bad tokenization"), bos_token="[BOS]")
    with pytest.raises(ValueError, match="bad tokenization"):
        window_maker(tokenizer=tokenizer)(batch())


@pytest.mark.parametrize("lengths", [
    [12, 2, 3], [2, 12, 12, 3], [2, 3, 12], [12], [2, 3, 2, 3, 2],
])
def test_every_internal_boundary_is_scored_once(lengths):
    """Long singleton windows retain their boundary, except at the document end."""
    tokens, starts = [], []
    for length in lengths:
        starts.append(len(tokens))
        tokens.extend([1] + [9] * (length - 1))
    scored = []
    for window in pack_windows(tokens, starts, bos_token_id=1, max_length=8):
        for index in range(window.sentence_start, window.sentence_end):
            position = starts[index] - window.token_start + 1
            if position < 8 and position != window.masked_boundary:
                scored.append(index)
    assert scored == list(range(len(lengths) - 1))


def test_window_maker_preserves_internal_singleton_label():
    prepare = window_maker(max_length=8)
    inputs = {"sentences": [["a b c d e f g h i", "j", "k l m n o p q r s"]],
              "labels": [["B-EOP", "O", "B-EOP"]], "example_id": [0]}
    output = prepare(inputs)
    assert [label for pair in output["labels"] for label in pair[0] if label != -100] == [0, 1]
    inputs["labels"][0][0] = "-100"
    output = prepare(inputs)
    assert [label for pair in output["labels"] for label in pair[0] if label != -100] == [1]


@pytest.mark.parametrize("augmented", [False, True])
def test_emitted_final_singleton_has_finite_differentiable_loss(augmented):
    output = window_maker(max_length=8)({
        "sentences": [["a b c d e f g h i", "j k l m n o p q r"]],
        "labels": [["B-EOP", "B-EOP"]], "example_id": [0],
    })
    objective = CombinedObjective(SimpleNamespace(
        hidden_size=4, num_labels=2, num_tssp_labels=3, do_da_ts=True,
        ts_loss_weight=1.0, cl_loss_weight=0.0, tssp_loss_weight=0.0,
    ))
    assert [sum(label != -100 for label in pair[0]) for pair in output["labels"]] == [1, 0]
    for pair in output["labels"]:
        labels = torch.tensor([pair[0]])
        hidden = torch.randn(1, 8, 4, requires_grad=True)
        loss, logits = objective(hidden, labels, augmented=augmented)
        assert torch.isfinite(loss)
        if (labels != -100).any():
            torch.testing.assert_close(loss, F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1)),
                                       rtol=0, atol=0)
        else:
            assert loss.item() == 0
        loss.backward()
        assert torch.isfinite(hidden.grad).all()
        if not (labels != -100).any():
            assert torch.count_nonzero(hidden.grad) == 0


def test_tssp_with_no_sentence_pairs_has_zero_differentiable_loss():
    objective = TSSP(SimpleNamespace(hidden_size=4, num_tssp_labels=3))
    hidden = torch.randn(1, 8, 4, requires_grad=True)
    ignored = torch.full((1, 8), -100)
    loss = objective(hidden, ignored, ignored)
    assert loss.item() == 0
    loss.backward()
    assert torch.count_nonzero(hidden.grad) == 0
