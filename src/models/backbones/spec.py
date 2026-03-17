import argparse
from dataclasses import dataclass
from typing import Callable

import torch.nn as nn


def _noop_backbone_args_with_names(_parser: argparse.ArgumentParser) -> tuple[str, ...]:
    return ()


def _declared_arg_names(
    add_args_with_names: Callable[[argparse.ArgumentParser], tuple[str, ...]],
) -> tuple[str, ...]:
    return tuple(add_args_with_names(argparse.ArgumentParser(add_help=False)))


def _register_backbone_args(
    add_args_with_names: Callable[[argparse.ArgumentParser], tuple[str, ...]],
) -> Callable[[argparse.ArgumentParser], None]:
    def register(parser: argparse.ArgumentParser) -> None:
        add_args_with_names(parser)

    return register


@dataclass(frozen=True)
class BackboneSpec:
    supported_behaviors: tuple[str, ...]
    build: Callable[..., nn.Module]
    add_args: Callable[[argparse.ArgumentParser], None]
    arg_names: tuple[str, ...]


def make_backbone_spec(
    *,
    supported_behaviors: tuple[str, ...],
    build: Callable[..., nn.Module],
    add_args_with_names: Callable[[argparse.ArgumentParser], tuple[str, ...]] = _noop_backbone_args_with_names,
) -> BackboneSpec:
    return BackboneSpec(
        supported_behaviors=supported_behaviors,
        build=build,
        add_args=_register_backbone_args(add_args_with_names),
        arg_names=_declared_arg_names(add_args_with_names),
    )
