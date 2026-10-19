from __future__ import annotations

from fastapi import Request


def dataset(request: Request):
    return request.app.state.dataset


def agent(request: Request):
    return request.app.state.agent


def run_store(request: Request):
    return request.app.state.store
