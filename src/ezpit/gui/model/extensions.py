# model/extensions.py
from __future__ import annotations

from collections.abc import Callable
from typing import Any

# hook_name -> list[callable(context)->dict]
_REGISTRY: dict[str, list[Callable[[dict], dict]]] = {}


def register_transform(hook: str):
    """
    Register a transform callable for the given hook.

      @register_transform("post_fq")
      def my_transform(context): ...
    """

    def deco(fn: Callable[[dict], dict]):
        _REGISTRY.setdefault(hook, []).append(fn)
        return fn

    return deco


def run_transforms(hook: str, context: dict) -> dict:
    """
    Hook 아래 등록된 변환을 모두 실행해 결과 dict를 병합해 반환합니다.

    예외는 삼켜서 메인 파이프라인을 깨지 않도록 합니다.
    """
    results: dict[str, Any] = {}
    for fn in _REGISTRY.get(hook, []):
        try:
            out = fn(context) or {}
            results.update(out)
        except Exception as e:
            results.setdefault("_errors", []).append(f"{fn.__name__}: {e}")
    return results
