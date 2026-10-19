from fastapi import APIRouter, HTTPException, Request, status

from app.api.dependencies import agent, dataset, run_store
from app.schemas import EvaluationRequest, RunReport
from app.services.evaluation import EvaluationHarness

router = APIRouter(tags=["evaluations"])


@router.post("/evaluation-runs", status_code=status.HTTP_201_CREATED)
async def create_evaluation_run(request: EvaluationRequest, request_context: Request):
    store = run_store(request_context)
    run_id = await EvaluationHarness(dataset(request_context), agent(request_context), store).run(
        request.split, request.agent_version)
    return report(store.get_run(run_id))


@router.get("/evaluation-runs/{run_id}")
async def get_evaluation_run(run_id: str, request_context: Request):
    run = run_store(request_context).get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return {**report(run).model_dump(), "answers": [
        {"answer_id": item.answer_id, "status": item.status, "latency_ms": item.latency_ms,
         "error": item.error_json, "claims": [
             {"claim_id": claim.claim_id, "start_char": claim.start_char, "end_char": claim.end_char,
              "quote": claim.quote, "label": claim.label, "fact_ids": claim.fact_ids, "reason": claim.reason}
             for claim in sorted(item.claims, key=lambda record: record.start_char)]}
        for item in sorted(run.answers, key=lambda record: record.answer_id)]}


def report(run) -> RunReport:
    return RunReport(run_id=run.id, split=run.split, agent_version=run.agent_version,
                     status=run.status,
                     counts={"answers": len(run.answers), "errors": sum(item.status == "error" for item in run.answers)},
                     metrics=run.metrics_json)
