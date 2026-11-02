"""Clinical-10 entity labels, relation signatures, model prompts, and annotation schema."""

# Entity label -> (annotation description, model prompt).
_ENTITIES = {
    "CONDITION": ("diagnoses, diseases, disorders, complications", "medical condition or diagnosis"),
    "SYMPTOM": ("symptoms, signs, abnormal clinical findings", "clinical sign or symptom"),
    "DRUG": ("medications and biologic treatments; exclude dose, route, frequency", "drug or medication"),
    "PROCEDURE": ("therapeutic or diagnostic procedures and surgeries", "medical procedure"),
    "TEST": ("laboratory tests, imaging, cultures, measured investigations", "diagnostic test"),
    "ANATOMY": ("body parts, organs, tissues, anatomical sites", "anatomical structure"),
}
ENTITY_TYPES = list(_ENTITIES)
ENTITY_PROMPTS = {prompt: label for label, (_, prompt) in _ENTITIES.items()}
ASSERTIONS = ["present", "absent", "possible", "historical", "conditional"]
_EVENTS = tuple(label for label in ENTITY_TYPES if label != "ANATOMY")

# Relation label -> (model prompt, allowed head types, allowed tail types).
_RELATIONS = {
    "TREATS": ("treats", ("DRUG", "PROCEDURE"), ("CONDITION", "SYMPTOM")),
    "CAUSES": ("causes", ("DRUG", "PROCEDURE", "CONDITION"), ("CONDITION", "SYMPTOM")),
    "INDICATES": ("test result indicates or supports diagnosis", ("TEST",), ("CONDITION", "SYMPTOM")),
    "TEST_FOR": ("test performed to investigate", ("TEST",), ("CONDITION", "SYMPTOM")),
    "PROCEDURE_FOR": ("procedure performed to treat or investigate", ("PROCEDURE",), ("CONDITION", "SYMPTOM")),
    "LOCATED_IN": ("located in anatomical site", ("CONDITION", "SYMPTOM", "PROCEDURE", "TEST"), ("ANATOMY",)),
    "TEMPORALLY_BEFORE": ("happens before", _EVENTS, _EVENTS),
}
RELATION_PROMPTS = {label: prompt for label, (prompt, _, _) in _RELATIONS.items()}
PROMPT_TO_RELATION = {prompt: label for label, prompt in RELATION_PROMPTS.items()}
SIGNATURES = {label: (heads, tails) for label, (_, heads, tails) in _RELATIONS.items()}


def _type_names(types):
    return "non-ANATOMY" if types == _EVENTS else "|".join(types)


POLICY = {
    "entity_types": {label: description for label, (description, _) in _ENTITIES.items()},
    "assertions": ASSERTIONS,
    "relation_signatures": {
        label: f"{_type_names(heads)} -> {_type_names(tails)}" for label, (heads, tails) in SIGNATURES.items()
    },
}


def object_schema(properties):
    """JSON schema of an object with exactly these properties."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


STRING, INTEGER = {"type": "string"}, {"type": "integer"}
MENTION = object_schema(
    {
        "mention_id": STRING,
        "begin": INTEGER,
        "end": INTEGER,
        "text": STRING,
        "assertion": {"type": "string", "enum": ASSERTIONS},
    }
)
ENTITY = object_schema(
    {
        "entity_id": STRING,
        "type": {"type": "string", "enum": ENTITY_TYPES},
        "mentions": {"type": "array", "items": MENTION, "minItems": 1},
    }
)
RELATION = object_schema(
    {
        "relation_id": STRING,
        "head": STRING,
        "type": {"type": "string", "enum": list(SIGNATURES)},
        "tail": STRING,
        "evidence_mentions": {"type": "array", "items": STRING, "minItems": 2},
    }
)
SCHEMA = object_schema(
    {
        "note_id": STRING,
        "annotator": STRING,
        "entities": {"type": "array", "items": ENTITY},
        "relations": {"type": "array", "items": RELATION},
        "notes": {"type": "array", "items": STRING},
    }
)
