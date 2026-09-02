"""Resolved selection of registered checks, enrichments, and media steps.

Raw ``step_names`` iterables belong at public and serialized boundaries. Once
an :class:`hflow.App` has checked those names against its registry and the
enabled stages, internal execution and planning carry one of these variants so
``None`` never has to mean "all steps" in business logic.
"""

from dataclasses import dataclass
from typing import assert_never

from hflow.steps import Stage


@dataclass(frozen=True)
class AllRegisteredSteps:
    """Every registered step in the enabled stages."""


@dataclass(frozen=True)
class SelectedRegisteredSteps:
    """The names validated for one application's enabled stages."""

    names: frozenset[str]


RegisteredStepSelection = AllRegisteredSteps | SelectedRegisteredSteps

ALL_REGISTERED_STEPS = AllRegisteredSteps()


def registered_step_is_selected(
    selection: RegisteredStepSelection,
    step_name: str,
) -> bool:
    """Whether ``step_name`` belongs to a resolved selection."""
    match selection:
        case AllRegisteredSteps():
            return True
        case SelectedRegisteredSteps(names=selected_step_names):
            return step_name in selected_step_names
        case unreachable:
            assert_never(unreachable)


def selected_step_names_for_stage(
    selection: RegisteredStepSelection,
    stage: Stage,
    stage_by_step_name: dict[str, Stage],
) -> frozenset[str] | None:
    """Names for ``stage``; ``None`` means the complete registered stage."""
    match selection:
        case AllRegisteredSteps():
            return None
        case SelectedRegisteredSteps(names=selected_step_names):
            return frozenset(
                step_name
                for step_name in selected_step_names
                if stage_by_step_name[step_name] is stage
            )
        case unreachable:
            assert_never(unreachable)
