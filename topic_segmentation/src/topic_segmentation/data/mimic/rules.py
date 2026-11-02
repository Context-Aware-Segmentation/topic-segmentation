"""Rules for splitting MIMIC notes into front matter, blocks, sections, and sentences."""
from __future__ import annotations

import re


COLON_HEADER_RE = re.compile(
    r"^\s*(?:[-*•]\s*)?(?P<label>[A-Za-z_][A-Za-z0-9 _/&(),.'+\-?]{1,100})\s*:\s*(?P<value>.*?)\s*$"
)
UNDERLINE_RE = re.compile(r"^\s*[=-]{3,}\s*$")
REJECT_ONE_WORD_LABELS = {
    "ABNORMAL", "AM", "LEFT", "MACHINE", "NO", "NORMAL", "PM", "RIGHT", "YES",
}
FRONT_MATTER_LABELS = {
    "ADMISSION DATE", "ATTENDING", "DATE OF BIRTH", "DISCHARGE DATE", "NAME", "SERVICE", "SEX", "UNIT", "UNIT NO",
    "___ NO",
}
SPECIAL_LABEL_ALIASES = {
    "___ COMPLAINT": "Chief Complaint",
    "___ ON ADMISSION": "Medications on Admission",
    "PHYSICAL ___": "Physical Exam",
    "___ DISPOSITION": "Discharge Disposition",
    "UNIT ___": "Unit No",
    "___ NO": "Unit No",
}
MAJOR_SECTION_LABELS = {
    "ACTIVE ISSUES", "ACTIVITY", "ACUTE ISSUES", "ADMISSION DIAGNOSES", "ADMISSION DIAGNOSIS", "ADMISSION EXAM",
    "ADMISSION LABS", "ADMISSION PHYSICAL EXAM", "ALLERGIES", "ASSESSMENT AND PLAN", "ASSESSMENT PLAN",
    "BRIEF HOSPITAL COURSE", "BRIEF SUMMARY", "CHIEF COMPLAINT", "CHRONIC AND STABLE ISSUES", "CHRONIC ISSUES", "DIET",
    "DISCHARGE CONDITION", "DISCHARGE DIAGNOSES", "DISCHARGE DIAGNOSIS", "DISCHARGE DISPOSITION", "DISCHARGE EXAM",
    "DISCHARGE INSTRUCTIONS", "DISCHARGE LABS", "DISCHARGE MEDICATIONS", "DISCHARGE PHYSICAL EXAM", "FACILITY",
    "FAMILY HISTORY", "FOLLOW UP", "FOLLOW UP APPOINTMENTS", "FOLLOW UP INSTRUCTIONS", "FOLLOWUP APPOINTMENTS",
    "FOLLOWUP INSTRUCTIONS", "FOLLOWUP PLAN", "GENERAL MEDICAL CONDITIONS", "HISTORY OF PRESENT ILLNESS",
    "HISTORY OF PRESENTING ILLNESS", "HOSPITAL COURSE", "HPI", "HPI EVENTS", "IMAGING", "IMAGING AND STUDIES",
    "IMAGING STUDIES", "LABORATORY STUDIES ON ADMISSION", "MAJOR PROCEDURE", "MAJOR SURGICAL OR INVASIVE PROCEDURE",
    "MEDICATION CHANGES", "MEDICATIONS", "MEDICATIONS ON ADMISSION", "MICRO", "MICROBIOLOGY", "ONCOLOGIC HISTORY",
    "OPAT ANTIMICROBIAL REGIMEN", "OPAT DIAGNOSIS", "OTHER PAST MEDICAL HISTORY", "PAIN MANAGEMENT", "PAST ILLNESSES",
    "PAST MEDICAL HISTORY", "PAST ONCOLOGIC HISTORY", "PAST SURGICAL HISTORY", "PATHOLOGY", "PATIENT INSTRUCTIONS",
    "PATIENT SUMMARY", "PERTINENT LABS", "PERTINENT RESULTS", "PHYSICAL EXAM", "PHYSICAL EXAMINATION",
    "PRIMARY DIAGNOSES", "PRIMARY DIAGNOSIS", "PROCEDURE IN DETAIL", "PSYCHIATRIC", "RELEVANT IMAGING", "RELEVANT LABS",
    "RELEVANT MICRO", "REVIEW OF SYSTEMS", "ROS", "SECONDARY DIAGNOSES", "SECONDARY DIAGNOSIS", "SOCIAL HISTORY",
    "SUMMARY STATEMENT", "SURGERY", "TRANSITIONAL ISSUES", "TRANSITIONS OF CARE", "TRANSPLANT SURGERY HOSPITAL COURSE",
    "WHAT HAPPENED IN THE HOSPITAL", "WORKING DIAGNOSIS", "WOUND CARE",
}
MEDICATION_LABEL_RE = re.compile(
    r"\b(SIG|TABLET|CAPSULE|NEBULIZATION|SUBCUTANEOUS|INTRAVENOUS|"
    r"INTRAMUSCULAR|TOPICAL|START|RESTART|DISP|DURATION|REFILLS?|"
    r"PRN|PO|BID|TID|QID|QHS|QAM|DAILY)\b",
    re.IGNORECASE,
)
SENTENCE_LABEL_STARTS = {
    "AFTER", "AT", "DURING", "EXAM", "HE", "HER", "HIS", "IF", "IMAGING", "IMMEDIATELY", "IN", "INITIAL", "IT", "LABS",
    "ON", "PATIENT", "PATIENTS", "PLEASE", "SHE", "STUDIES", "THE", "THERE", "THEY", "THIS", "VITALS", "WHILE", "YOU",
}
SENTENCE_LABEL_PHRASES = (
    " WAS ", " WERE ", " WAS GIVEN", " WAS TAKEN", " WERE GIVEN", " WHICH SHOWED", " SHOWED", " DEMONSTRATED",
    " GIVEN THAT", " NOTABLE ON ", " NOTABLE FOR", " SIGNIFICANT FOR", " DESCRIBES", " INCLUDE", " INCLUDES",
    " SUCH AS", " IF NO", " REASON WHY", " BEING D/C", " MEDICAL HISTORY IS", " HOSPITAL COURSE WAS",
)
FIELD_SPLITS = {
    "Unit No": ("Name",),
    "Unit ___": ("Name",),
    "___ No": ("Name",),
    "Discharge Date": ("Admission Date",),
    "Sex": ("Date of Birth",),
}
RIGHT_FIELD_RE = re.compile(
    r"^(?P<left>.*?\S)(?P<gap>[ \t]{2,})(?P<right>(?P<label>Unit No|Unit ___|___ No|Discharge Date|Sex)\s*:.*)$"
)
BODY_START_RE = re.compile(
    r"^\s*(?:"
    r"Chief Complaint|___ Complaint|Major Surgical or Invasive Procedure|"
    r"History of Present Illness|Past Medical History|Brief Hospital Course|"
    r"Hospital Course|Discharge Diagnosis|Discharge Condition"
    r")\s*:",
    re.IGNORECASE,
)
SENTENCE_SPLIT_RE = re.compile(
    r"<SPLIT>|"
    r"\s*//+\s*|"
    r"(?<=[.!?])\s+(?=(?:---+|-|o\s+)\s*[A-Z(])|"
    r"(?<=\))\s*(?=[A-Z])|"
    r"(?<=[.!?]\))\s+(?=[A-Z])|"
    r"(?<=[.!?])\s+(?=[A-Z0-9_(\[*?])|"
    r"(?<=[.!?])(?=[a-z])|"
    r"(?<=[.!?])(?=[A-Z])"
)
SEPARATOR_RE = re.compile(r"^[=_-]{3,}$")
LIST_ITEM_RE = re.compile(r"^\s*(?:\d+(?:\.(?!\d)|\))|#\d*:|#|[-*•]|\[[ xX]\])\s*|^\s*o\s+")
LEADING_NUMBER_MARKER_RE = re.compile(r"^(\s*\d+)\.\s+")
NUMBER_CONTINUATION_PREVIOUS_RE = re.compile(r"(?i)\b(?:series|image|slice|view|figure|fig)\s*$")
PRESERVE_FIELD_LABELS = {
    "ADMISSION DATE", "ATTENDING", "DATE OF BIRTH", "DISCHARGE DATE", "NAME", "SERVICE", "SEX", "UNIT NO",
}
ABBREVIATIONS = (
    "Mr.", "Ms.", "Mrs.", "Dr.", "Drs.", "M.D.", "M.S.", "R.N.", "O.R.", "E.C.", "Sust.Rel.", "P.Mibi", "y.o.", "A.M.",
    "P.M.", "a.m.", "p.m.", "Pt.", "Prof.", "Sr.", "Jr.", "St.", "vs.", "p.o.", "PO.", "max.", "min.", "bilat.", "U.S.",
    "e.g.", "i.e.", "etc.",
)
BARE_ABBREVIATIONS = {abbr for abbr in ABBREVIATIONS if len(abbr) <= 5}
UPPER_DOTTED_ABBREVIATION_RE = re.compile(r"\b(?:[A-Z]{1,3}\.){2,}")
LOWER_DOTTED_ABBREVIATION_RE = re.compile(r"\b(?:[a-z]\.){2,}(?:[a-z]\.?)?")
MEDICAL_DOTTED_TERM_RE = re.compile(
    r"\b([CEHSTIi])\.\s+(?=(?:[Dd]iff(?:icile)?|[Cc]oli|[Pp]ylori|[Aa]ureus|[Bb]ili|[Ee]\b))"
)
INLINE_NUMBER_MARKER_RE = re.compile(r"(^|[:;]\s+)(\d+)\.\s+(?=[A-Z])")
DECIMAL_POINT_RE = re.compile(r"(?<=\d)\.\s*(?=\d)")


def split_front_matter_line(line: str) -> list[str]:
    """Split a two-column front-matter line into fields; keep other lines as a single field."""
    match = RIGHT_FIELD_RE.match(line)
    left = match.group("left").rstrip() if match else ""
    if not match or not any(left.lstrip().startswith(f"{label}:") for label in FIELD_SPLITS[match.group("label")]):
        return [line]
    return [left, match.group("right").lstrip()]


def normalize_front_matter_doc(doc: dict, max_front_matter_lines: int = 30) -> dict:
    """Split two-column front matter into separate lines before the first body heading."""
    out, in_front_matter = [], True
    for line_index, raw_line in enumerate(doc["text"].splitlines(keepends=True)):
        line = raw_line.rstrip("\r\n")
        linebreak = raw_line[len(line):]
        if in_front_matter and BODY_START_RE.match(line):
            in_front_matter = False
        fields = split_front_matter_line(line) if in_front_matter and line_index < max_front_matter_lines else [line]
        if len(fields) == 1:
            out.append(raw_line)
        elif linebreak:
            out.extend(f"{field}{linebreak}" for field in fields)
        else:
            out.append("\n".join(fields))
    return dict(doc, text="".join(out))


def label_key(label: str) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", label.upper()).strip()


def clean_label(label: str) -> str:
    label = " ".join(label.split())
    label = label.strip("* ")
    label = re.sub(r"^\s*[-*•]\s*", "", label).strip()
    raw_key = label.upper()
    if raw_key in SPECIAL_LABEL_ALIASES:
        return SPECIAL_LABEL_ALIASES[raw_key]
    key = label_key(label)
    return SPECIAL_LABEL_ALIASES.get(key, label)


def is_major_label(label: str) -> bool:
    return label_key(clean_label(label)) in MAJOR_SECTION_LABELS


def looks_like_sentence_label(label: str) -> bool:
    key = f" {label_key(label)} "
    words = key.strip().split()
    if "." in label:
        return True
    if len(words) >= 2 and words[0] in {"HE", "SHE", "PATIENT", "PATIENTS", "THEY", "IT", "WE"}:
        return True
    if len(words) >= 4 and words[0] in SENTENCE_LABEL_STARTS:
        return True
    return any(phrase in key for phrase in SENTENCE_LABEL_PHRASES)


def looks_like_medication_label(label: str) -> bool:
    key = label_key(label)
    if key in {"SIG", "START", "DISP", "DURATION", "REASON FOR PRN DUPLICATE OVERRIDE"}:
        return True
    if "REASON FOR PRN DUPLICATE OVERRIDE" in key or "REASON FOR ORDERING" in key:
        return True
    if key.startswith(("SUBCUTANEOUS", "INTRAVENOUS", "INTRAMUSCULAR", "NEBULIZATION", "TABLET", "CAPSULE")):
        return True
    if key.startswith("PICC ") or " HEPARIN DEPENDENT" in key:
        return True
    return bool(MEDICATION_LABEL_RE.search(label))


def label_ok(label: str, *, allow_major: bool = False, allow_problem: bool = False) -> bool:
    label = clean_label(label)
    key = label_key(label)
    letters = sum(ch.isalpha() for ch in label)
    digits = sum(ch.isdigit() for ch in label)
    if letters < 2 or digits > max(4, letters * 2):
        return False
    if any(ch in label for ch in "()[]{}"):
        return False
    if "." in label:
        return False
    if "," in label and len(label) > 24 and label_key(label) not in MAJOR_SECTION_LABELS:
        return False
    first_alpha = next((ch for ch in label if ch.isalpha()), "")
    if first_alpha and first_alpha.islower():
        return False
    if key in REJECT_ONE_WORD_LABELS:
        return False
    if not allow_problem and looks_like_sentence_label(label):
        return False
    if not allow_major and looks_like_medication_label(label):
        return False
    return True


def normalized_heading_line(line: str) -> str:
    stripped = line.strip()
    if stripped.startswith("=") and stripped.endswith("="):
        stripped = stripped.strip("= ").strip()
    return stripped


def split_hash_label_value(text: str) -> tuple[str, str | None]:
    for index, char in enumerate(text):
        if char != ":":
            continue
        prev_char = text[index - 1] if index else ""
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if prev_char.isdigit() and next_char.isdigit():
            continue
        if next_char and not next_char.isspace():
            continue
        return text[:index].strip(), text[index + 1 :].strip() or None
    dash_match = re.search(r"\s*-\s+", text)
    if dash_match:
        label = text[: dash_match.start()].strip()
        value = text[dash_match.end() :].strip()
        if 1 <= len(label.split()) <= 8 and value:
            return label.rstrip(".").strip(), value
    return text.rstrip(".").strip(), None


def hash_header_parts(line: str) -> tuple[str, str | None] | None:
    stripped = line.strip()
    if not stripped.startswith("#"):
        return None
    after_hash = stripped.lstrip("#").lstrip()
    if after_hash.startswith("*"):
        return None
    after_hash = re.sub(r"^[).]\s*", "", after_hash).strip()
    label_text, value = split_hash_label_value(after_hash)
    label = clean_label(label_text)
    if "REFILL" in label_key(label):
        return None
    if value is None and len(label.split()) > 8:
        return None
    if not label_ok(label, allow_problem=True):
        return None
    return label, value


def colon_header_parts(line: str) -> tuple[str, str | None] | None:
    match = COLON_HEADER_RE.match(normalized_heading_line(line))
    if not match:
        return None
    label = clean_label(match.group("label"))
    value = match.group("value").strip()
    if label_key(label) in FRONT_MATTER_LABELS and not value:
        return None
    if value:
        if is_major_label(label) and label_ok(label, allow_major=True):
            return label, value
        return None
    return (label, None) if label_ok(label) else None


def subsection_parts(line: str) -> tuple[str, str | None] | None:
    return hash_header_parts(line) or colon_header_parts(line) or standalone_header_parts(line)


def standalone_header_parts(line: str) -> tuple[str, str | None] | None:
    label = clean_label(normalized_heading_line(line))
    if not label or ":" in label:
        return None
    key = label_key(label)
    if key not in MAJOR_SECTION_LABELS:
        return None
    if not label_ok(label, allow_major=True):
        return None
    return label, None


def underline_heading_label(line: str, next_line: str) -> str | None:
    if not UNDERLINE_RE.match(next_line):
        return None
    label = clean_label(line.strip())
    if not label or ":" in label:
        return None
    key = label_key(label)
    is_upper_heading = any(ch.isalpha() for ch in label) and label.upper() == label
    if not is_upper_heading and key not in MAJOR_SECTION_LABELS:
        return None
    if key in FRONT_MATTER_LABELS:
        return None
    return label if label_ok(label, allow_major=True) else None


def split_blocks(text: str) -> list[tuple[int, int, list[tuple[int, str]]]]:
    blocks: list[tuple[int, int, list[tuple[int, str]]]] = []
    current: list[tuple[int, str]] = []
    pending_blanks: list[tuple[int, str]] = []
    start_line: int | None = None

    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            if start_line is not None:
                pending_blanks.append((line_no, line))
            continue

        if len(pending_blanks) >= 2:
            if current:
                blocks.append((start_line or current[0][0], current[-1][0], current))
            current = [(line_no, line)]
            start_line = line_no
        else:
            if start_line is None:
                start_line = line_no
            current.extend(pending_blanks)
            current.append((line_no, line))
        pending_blanks = []

    if current:
        blocks.append((start_line or current[0][0], current[-1][0], current))
    return blocks


def split_subsections(block_lines: list[tuple[int, str]]) -> list[tuple[str | None, int | None, list[tuple[int, str]]]]:
    sections: list[tuple[str | None, int | None, list[tuple[int, str]]]] = []
    current_header: str | None = None
    current_header_line: int | None = None
    current_lines: list[tuple[int, str]] = []

    i = 0
    while i < len(block_lines):
        line_no, line = block_lines[i]
        if i + 1 < len(block_lines):
            underline_label = underline_heading_label(line, block_lines[i + 1][1])
            if underline_label is not None:
                if current_header is not None or current_lines:
                    sections.append((current_header, current_header_line, current_lines))
                current_header = underline_label
                current_header_line = line_no
                current_lines = []
                i += 2
                continue

        subsection = subsection_parts(line)
        if subsection is not None:
            label, value = subsection
            if current_header is not None and label_key(label) == label_key(current_header) and not current_lines:
                if value:
                    current_lines.append((line_no, value))
                i += 1
                continue
            if current_header is not None or current_lines:
                sections.append((current_header, current_header_line, current_lines))
            current_header = label
            current_header_line = line_no
            current_lines = []
            if value:
                current_lines.append((line_no, value))
            i += 2 if i + 1 < len(block_lines) and UNDERLINE_RE.match(block_lines[i + 1][1]) else 1
            continue
        current_lines.append((line_no, line))
        i += 1

    if current_header is not None or current_lines:
        sections.append((current_header, current_header_line, current_lines))
    return sections


def protect_sentence_text(text: str) -> str:
    protected = text
    for index, abbreviation in enumerate(ABBREVIATIONS):
        protected = protected.replace(abbreviation, abbreviation.replace(".", f"<ABBR{index}>"))
    protected = protected.replace("y.o", "y<DOT>o")
    protected = re.sub(r"\by\.\s+o\b", "y<DOT>o", protected)
    protected = MEDICAL_DOTTED_TERM_RE.sub(r"\1<DOT> ", protected)
    protected = UPPER_DOTTED_ABBREVIATION_RE.sub(
        lambda match: match.group(0).replace(".", "<DOT>"), protected
    )
    protected = LOWER_DOTTED_ABBREVIATION_RE.sub(
        lambda match: match.group(0).replace(".", "<DOT>"), protected
    )
    protected = re.sub(r"([a-z]{4,}\.)\s+(?=[a-z]{3,})", r"\1<SPLIT>", protected)
    protected = INLINE_NUMBER_MARKER_RE.sub(r"\1\2<NUMDOT> ", protected)
    protected = re.sub(r"^(\s*\d+)\.(?=[A-Za-z])", r"\1<NUMDOT> ", protected)
    protected = LEADING_NUMBER_MARKER_RE.sub(r"\1<NUMDOT> ", protected)
    protected = DECIMAL_POINT_RE.sub("<DECDOT>", protected)
    return protected


def restore_sentence_text(text: str) -> str:
    restored = text.replace("<NUMDOT>", ".").replace("<DECDOT>", ".").replace("<DOT>", ".")
    for index, abbreviation in enumerate(ABBREVIATIONS):
        restored = restored.replace(abbreviation.replace(".", f"<ABBR{index}>"), abbreviation)
    return restored


def split_paragraph(paragraph: str) -> list[str]:
    protected = protect_sentence_text(paragraph)
    parts = []
    for part in SENTENCE_SPLIT_RE.split(protected):
        restored = restore_sentence_text(part).strip()
        restored = re.sub(r"(?:\s*\.){2,}$", ".", restored)
        restored = re.sub(r"\s+\.$", ".", restored)
        parts.append(restored)
    return [part for part in parts if part and part not in {".", "·"} and not SEPARATOR_RE.match(part)]


def join_wrapped_lines(lines: list[str]) -> str:
    joined = ""
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not joined:
            joined = stripped
            continue
        joined += stripped if should_join_hyphenated_wrap(joined, stripped) else f" {stripped}"
    return joined


def should_join_hyphenated_wrap(previous: str, current: str) -> bool:
    if not re.search(r"[A-Za-z]-\s*$", previous):
        return False
    if re.match(r"(?i)(?:and|or|I:|II:|III:|IV:|V:|VI:|Alert\b)", current):
        return False
    previous_word = re.search(r"([A-Za-z]+)-\s*$", previous)
    current_word = re.match(r"([A-Za-z]+)", current)
    if not previous_word or not current_word:
        return False
    if current_word.group(1)[0].islower():
        return previous_word.group(1)[-1].islower() or previous_word.group(1).isupper()
    return False


def preserve_as_own_sentence(line: str) -> bool:
    if ":" not in line:
        return False
    label, value = line.split(":", 1)
    return label_key(label) in PRESERVE_FIELD_LABELS and bool(value.strip())


def number_continues_previous_line(paragraph: list[str], line: str) -> bool:
    return bool(paragraph and re.match(r"^\s*\d+\.\s+", line) and NUMBER_CONTINUATION_PREVIOUS_RE.search(paragraph[-1].strip()))


def split_sentences(lines: list[str]) -> list[str]:
    sentences: list[str] = []
    paragraph: list[str] = []

    def add_parts(parts: list[str]) -> None:
        for part in parts:
            if sentences and sentences[-1] in BARE_ABBREVIATIONS:
                sentences[-1] = f"{sentences[-1]} {part}"
            else:
                sentences.append(part)

    def flush() -> None:
        if paragraph:
            add_parts(split_paragraph(join_wrapped_lines(paragraph)))
            paragraph.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if stripped in {".", "·"} or SEPARATOR_RE.match(stripped):
            if stripped == "___" and paragraph and paragraph[-1].rstrip().endswith("Dr."):
                paragraph.append(stripped)
                continue
            flush()
            continue
        if preserve_as_own_sentence(stripped):
            flush()
            add_parts([stripped])
            continue
        if paragraph and re.search(r"\by\.$", paragraph[-1].strip()) and re.match(r"(?i)^o\b", stripped):
            paragraph.append(stripped)
            continue
        if LIST_ITEM_RE.match(stripped) and not number_continues_previous_line(paragraph, stripped):
            flush()
            paragraph.append(stripped)
            continue
        paragraph.append(stripped)

    flush()
    return merge_sentence_fragments(sentences)


def merge_sentence_fragments(sentences: list[str]) -> list[str]:
    merged: list[str] = []
    i = 0
    while i < len(sentences):
        current = sentences[i]
        i += 1
        while i < len(sentences) and should_merge_with_next(current, sentences[i]):
            current = f"{current} {sentences[i]}"
            i += 1
        merged.append(current)
    return merged


def should_merge_with_next(current: str, next_sentence: str) -> bool:
    if re.fullmatch(r"\d+[.)]", current):
        return True
    if re.search(r"\bSig:\s+\S+\s+\(\d+\)$", current):
        return True
    if re.search(r"\b(?:Sust|Ophth)\.$|\(E\.C\.\)$", current):
        return True
    if re.search(r"\b(?:every|times a|once a|by mouth every)$", current, re.IGNORECASE):
        return True
    if re.search(r"\(\d+$", current) and re.match(r"(?i)^(?:times?|day|hours?|tablet|capsule|ml|neb)\b", next_sentence):
        return True
    if current.count("(") > current.count(")") and re.fullmatch(r"[\d,;: ]+\)\.?", next_sentence):
        return True
    return False
