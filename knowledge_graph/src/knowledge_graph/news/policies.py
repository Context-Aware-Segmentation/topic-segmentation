"""Combine spaCy and ReLiK mention spans and assign their entity links."""

COMBINED_MENTIONS = "relik_first_flat"  # the spaCy + ReLiK mentions used for NER and predicted-mention RE


def overlaps(left, right):
    return left[0] < right[1] and right[0] < left[1]


def select_without_overlap(ranked, selected=()):
    """Spans taken in order, skipping any that overlaps a span already selected."""
    selected = set(selected)
    for candidate in ranked:
        if not any(overlaps(candidate, kept) for kept in selected):
            selected.add(candidate)
    return selected


def longest_first(spans):
    return select_without_overlap(sorted(spans, key=lambda span: (-(span[1] - span[0]), span)))


def first_flat(preferred, other):
    """Preferred spans longest-first without overlaps, then the other system's spans in document order."""
    return select_without_overlap(sorted(other - preferred), longest_first(preferred))


def length_flat(spacy, relik, longest):
    """The union longest- or shortest-first, ReLiK first on equal length, without overlaps."""

    def rank(span):
        length = span[1] - span[0]
        return -length if longest else length, span not in relik, span

    return select_without_overlap(sorted(spacy | relik, key=rank))


def drop_boundary_conflicts(spacy, relik):
    """Exact agreements and spans that overlap no span of the other system, longest-first without overlaps."""
    exact = spacy & relik
    spacy_safe = {span for span in spacy - exact if not any(overlaps(span, other) for other in relik)}
    relik_safe = {span for span in relik - exact if not any(overlaps(span, other) for other in spacy)}
    return longest_first(exact | spacy_safe | relik_safe)


POLICIES = {
    "spacy_only": lambda spacy, relik: spacy,
    "relik_only": lambda spacy, relik: relik,
    "exact_agreement": lambda spacy, relik: spacy & relik,
    "candidate_union": lambda spacy, relik: spacy | relik,
    "spacy_first_flat": lambda spacy, relik: first_flat(spacy, relik),
    "relik_first_flat": lambda spacy, relik: first_flat(relik, spacy),
    "longest_flat": lambda spacy, relik: length_flat(spacy, relik, longest=True),
    "shortest_flat": lambda spacy, relik: length_flat(spacy, relik, longest=False),
    "drop_boundary_conflicts": drop_boundary_conflicts,
}

LINK_ASSIGNMENTS = {
    "relik_only": lambda spacy, relik: relik,
    "spacy_only": lambda spacy, relik: spacy,
    "relik_then_spacy": lambda spacy, relik: relik or spacy,
    "spacy_then_relik": lambda spacy, relik: spacy or relik,
    "qid_agreement": lambda spacy, relik: spacy & relik,
}


def default_links(policy):
    """A single system keeps its own links; combined policies prefer ReLiK, except spaCy-first flat."""
    return {"spacy_only": "spacy_only", "relik_only": "relik_only", "spacy_first_flat": "spacy_then_relik"}.get(
        policy, "relik_then_spacy"
    )
