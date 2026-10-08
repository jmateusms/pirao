"""Ready-made analyses: a spec, its data and sampler settings, with a note.

Each example is one JSON file in ``pirao/examples``.  The files hold only
public data sets or numbers made up for teaching (``data_kind`` says which), so
any of them can be shown in a class.  Loading one validates it the same way a
user's own input is validated, so an example cannot drift out of step with the
registries without a test noticing.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path
from typing import Any

from .spec import ModelSpec, SamplerConfig

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"

#: Fields of an example file shown in a listing; the spec and rows are not.
LISTING_FIELDS = (
    "id",
    "title",
    "title_pt",
    "note",
    "note_pt",
    "source",
    "source_pt",
    "data_kind",
)


class ExampleNotFound(KeyError):
    pass


@cache
def _load_all() -> dict[str, dict[str, Any]]:
    examples = {}
    for path in sorted(EXAMPLES_DIR.glob("*.json")):
        example = json.loads(path.read_text(encoding="utf-8"))
        if example.get("id") != path.stem:
            raise ValueError(f"{path.name}: 'id' must match the file name")
        examples[path.stem] = example
    return dict(sorted(examples.items(), key=lambda kv: kv[1].get("order", 99)))


def list_examples() -> list[dict[str, Any]]:
    """Every example's title, note and source, in display order."""
    out = []
    for example in _load_all().values():
        entry = {k: example.get(k) for k in LISTING_FIELDS}
        entry["likelihood"] = example["spec"]["likelihood"]
        entry["n_rows"] = len(example["rows"])
        out.append(entry)
    return out


def get_example(example_id: str) -> dict[str, Any]:
    """One example in full, with its spec and sampler settings validated."""
    try:
        example = _load_all()[example_id]
    except KeyError:
        raise ExampleNotFound(example_id) from None
    spec = ModelSpec.model_validate(example["spec"])
    sampler = SamplerConfig.model_validate(example.get("sampler", {}))
    return {
        **example,
        "spec": spec.model_dump(mode="json"),
        "sampler": sampler.model_dump(mode="json"),
        "rows": [dict(row) for row in example["rows"]],
    }
