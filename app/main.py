from pathlib import Path

from fastapi import FastAPI

from app.api.routes import classification, evaluations
from app.core.config import Settings
from app.services.catalogue import Dataset
from app.providers.gemini import GeminiProvider
from app.repositories.database import RunStore, initialize, make_engine
from app.services.classification import ClaimAgent


def build_app(data_dir: Path | None = None, database_url: str | None = None) -> FastAPI:
    settings = Settings.from_environment()
    dataset = Dataset(data_dir or settings.data_dir)
    engine = make_engine(database_url or settings.database_url)
    app = FastAPI(title="Speaker Claim Evaluator")
    app.state.dataset = dataset
    app.state.store = RunStore(engine)
    app.state.agent = ClaimAgent(GeminiProvider(settings.gemini_model), dataset)
    app.include_router(classification.router)
    app.include_router(evaluations.router)

    @app.on_event("startup")
    def initialize_database() -> None:
        initialize(engine)

    return app


app = build_app()
