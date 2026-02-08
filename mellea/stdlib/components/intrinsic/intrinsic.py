"""Module for Intrinsics."""

import abc
from typing import Generic

from ....backends.adapters import AdapterType, fetch_intrinsic_metadata
from ....backends.adapters.adapter import Adapter
from ....core import CBlock, Component, ModelOutputThunk, S, TemplateRepresentation


class AdapterBackedComponent(Generic[S], Component[S], abc.ABC):
    """A component that is backed by an Adapter."""

    def __init__(
        self,
        adapter: Adapter | str,
        adapter_types: tuple[AdapterType, ...],
        adapter_kwargs: dict | None = None,
    ):
        """A component that is backed by an Adapter.

        `AdapterBackedComponent`s are special components that explicitly activate a model modality.
        These model modalities are usually explicit (activating an aLoRA or loading and using an LoRA).
        However, these model modalities could also be implicit, although that use is discouraged except as a compatibility shim.

        Args:
            adapter: the adapter that will be activated when this component is the `action` passed into a `Backend` generate call.
            adapter_types: the types of the adapter. Example: lora, alora, baked_in_mode
            adapter_kwargs: some adapters require kwargs when utilizing them; provide those here.
        """
        self._adapter = adapter
        self._adapter_types = adapter_types
        self._adapter_kwargs = adapter_kwargs

    @property
    def adapter_name(self) -> str:
        """Resolves and returns the name of the adapter."""
        match self._adapter:
            case str():
                return self._adapter
            case Adapter():
                return self._adapter.name
            case _:
                raise TypeError(
                    f"Expected Adapter | str but found {self._adapter} : {type(self._adapter)}"
                )

    @property
    def adapter_types(self) -> tuple[AdapterType, ...]:
        """Tuple of available adapter types that implement this intrinsic."""
        return self._adapter_types


class Intrinsic(Component[str]):
    """A component representing an intrinsic."""

    def __init__(
        self, intrinsic_name: str, intrinsic_kwargs: dict | None = None
    ) -> None:
        """A component for rewriting messages using intrinsics.

        Intrinsics are special components that transform a chat completion request.
        These transformations typically take the form of:
        - parameter changes (typically structured outputs)
        - adding new messages to the chat
        - editing existing messages

        An intrinsic component should correspond to a loaded adapter.

        Args:
            intrinsic_name: the user-visible name of the intrinsic; must match a known
                name in Mellea's intrinsics catalog.
            intrinsic_kwargs: some intrinsics require kwargs when utilizing them;
                provide them here
        """
        self.metadata = fetch_intrinsic_metadata(intrinsic_name)
        if intrinsic_kwargs is None:
            intrinsic_kwargs = {}
        self.intrinsic_kwargs = intrinsic_kwargs

    @property
    def intrinsic_name(self):
        """User-visible name of this intrinsic."""
        return self.metadata.name

    @property
    def adapter_types(self) -> tuple[AdapterType, ...]:
        """Tuple of available adapter types that implement this intrinsic."""
        return self.metadata.adapter_types

    def parts(self) -> list[Component | CBlock]:
        """The set of all the constituent parts of the `Intrinsic`.

        Will need to be implemented by subclasses since not all intrinsics are output
        as text / messages.
        """
        return []  # TODO revisit this.

    def format_for_llm(self) -> TemplateRepresentation | str:
        """`Intrinsic` doesn't implement `format_for_default`.

        Formats the `Intrinsic` into a `TemplateRepresentation` or string.

        Returns: a `TemplateRepresentation` or string
        """
        raise NotImplementedError(
            "`Intrinsic` doesn't implement format_for_llm by default. You should only "
            "use an `Intrinsic` as the action and not as a part of the context."
        )

    def _parse(self, computed: ModelOutputThunk) -> str:
        """Parse the model output. Returns string value for now."""
        return computed.value if computed.value is not None else ""
