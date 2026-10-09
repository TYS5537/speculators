"""Pure schema-to-argparse conversion, without backend registration or resolution."""

import argparse
import types
from typing import Any, Literal, Union, get_args, get_origin

from pydantic.fields import FieldInfo


def _annotation_spec(annotation: Any) -> tuple[Any, bool, list[Any] | None]:
    """Reduce an optional, list or Literal annotation to type/nargs/choices."""
    origin = get_origin(annotation)
    if origin in (Union, types.UnionType):
        non_none = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(non_none) != 1:
            raise TypeError(
                f"CLI generation supports only single-type optionals; got "
                f"{annotation!r} with {len(non_none)} non-None arms."
            )
        annotation = non_none[0]
        origin = get_origin(annotation)
    if origin is list:
        (inner,) = get_args(annotation)
        return inner, True, None
    if origin is Literal:
        choices = list(get_args(annotation))
        elem_types = {type(choice) for choice in choices}
        base = elem_types.pop() if len(elem_types) == 1 else str
        return base, False, choices
    return annotation, False, None


def field_argument_kwargs(field: FieldInfo) -> dict[str, Any]:
    """Build fresh argparse kwargs; callers own defaults and dynamic registries.

    SUPPRESS lets the config CLI distinguish absent flags from explicit defaults.
    The legacy CLI replaces it with the field's raw default and its recipe overrides;
    neither caller needs to instantiate or validate a config during registration.
    """
    base, is_list, choices = _annotation_spec(field.annotation)
    kwargs: dict[str, Any] = {"default": argparse.SUPPRESS}
    if field.description:
        kwargs["help"] = field.description
    extra = field.json_schema_extra if isinstance(field.json_schema_extra, dict) else {}
    if base is bool and not is_list:
        optional = (
            extra.get("cli_bool") == "optional"
            or field.default is None
            or field.default is True
        )
        kwargs["action"] = argparse.BooleanOptionalAction if optional else "store_true"
    else:
        if is_list:
            kwargs["nargs"] = "+"
        kwargs["type"] = base
        if choices:
            kwargs["choices"] = choices
    return kwargs
