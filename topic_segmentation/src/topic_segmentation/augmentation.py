"""Document augmentation (DA) and topic-aware sentence structure (TSSP) labels for the paired views."""
from dataclasses import dataclass, replace


@dataclass
class Unit:
    text: str
    token_ids: list[int]
    label: int


@dataclass
class AugmentedDocument:
    input_ids: list[int]
    sentences: list[str]
    token_labels: list[int]
    pair_orders: list[int]


def split_units(texts, labels, input_ids, boundary_ids):
    starts = [i for i, token in enumerate(input_ids) if token in boundary_ids]
    assert len(starts) == len(texts)
    ends = starts[1:] + [len(input_ids)]
    return [
        Unit(text, input_ids[start:end], label)
        for text, label, start, end in zip(texts, labels, starts, ends)
    ]


def section_spans(document, boundary_label):
    ends = [i for i, sentence in enumerate(document) if sentence.label == boundary_label]
    starts = [0] + [end + 1 for end in ends[:-1]]
    return list(zip(starts, ends))


def mix_sections(documents, topic_spans, document_index, rng):
    """Shuffle topics, optionally drawing replacements from another document."""
    topic_order = list(range(len(topic_spans[document_index])))
    rng.shuffle(topic_order)
    # Single-document batches also consume this draw before later shuffles.
    replace_topics = rng.random() > 0.5 and len(documents) > 1
    mixed = []
    for topic_index in topic_order:
        source_index = document_index
        prefix = f"{topic_index}-"
        if replace_topics and rng.random() > 0.5:
            candidates = [i for i in range(len(documents)) if i != document_index]
            source_index = rng.choice(candidates)
            topic_index = rng.choice(list(range(len(topic_spans[source_index]))))
            prefix = f"{source_index}-{topic_index}-"
        start, end = topic_spans[source_index][topic_index]
        for sentence_index in range(start, end + 1):
            sentence = documents[source_index][sentence_index]
            mixed.append(replace(sentence, text=f"{prefix}{sentence_index}-{sentence.text}"))
    return mixed


def pair_order(position, sentence_indices):
    """TSSP label of a shuffled sentence: 2 opens a topic, 0 follows its predecessor, 1 otherwise."""
    if position == 0:
        return 2
    return 0 if sentence_indices[position - 1] == sentence_indices[position] - 1 else 1


def shuffle_units(document, label_to_id, rng):
    output = AugmentedDocument(input_ids=[], sentences=[], token_labels=[], pair_orders=[])
    for start, end in section_spans(document, label_to_id["B-EOP"]):
        indices = list(range(start, end))
        rng.shuffle(indices)
        indices.append(end)  # The topic-final sentence always remains last.
        for position, sentence_index in enumerate(indices):
            sentence = document[sentence_index]
            label = label_to_id["B-EOP"] if position == len(indices) - 1 else label_to_id["O"]
            pair_label = pair_order(position, indices)
            ignored = [-100] * (len(sentence.token_ids) - 1)
            output.input_ids.extend(sentence.token_ids)
            output.sentences.append(f"{sentence_index}-{sentence.text}")
            output.token_labels.extend([label] + ignored)
            output.pair_orders.extend([pair_label] + ignored)
    return output


def augment(sentences, labels, tokenized, label_to_id, boundary_ids, rng):
    """Create one augmented view per document, drawing from `rng` in document order."""
    documents = [
        split_units(texts, [label_to_id.get(label, -100) for label in doc_labels], ids, boundary_ids)
        for texts, doc_labels, ids in zip(sentences, labels, tokenized["input_ids"])
    ]
    topic_spans = [section_spans(document, label_to_id["B-EOP"]) for document in documents]
    return [shuffle_units(mix_sections(documents, topic_spans, index, rng), label_to_id, rng)
            for index in range(len(documents))]
