from fastapi import APIRouter, HTTPException, Request

from app.api.dependencies import agent, dataset
from app.core.errors import AgentErrorCode
from app.models import AgentInput
from app.schemas import ClassifyRequest
from app.services.classification import AgentFailure

router = APIRouter(tags=["classification"])


@router.post("/classify")
async def classify(request: ClassifyRequest, request_context: Request):
    ds, live_agent = dataset(request_context), agent(request_context)
    if request.brand_id not in ds.facts_by_brand:
        raise HTTPException(status_code=404, detail="unknown primary brand")
    try:
        return await live_agent.classify(
            AgentInput(answer_id=request.answer_id, brand_id=request.brand_id,
                       question=request.question, ai_answer=request.ai_answer),
            request.agent_version,
        )
    except AgentFailure as error:
        status_code = 504 if error.code == AgentErrorCode.PROVIDER_TIMEOUT else 502
        raise HTTPException(status_code=status_code, detail={
            "code": error.code.value,
            "message": "provider request could not be completed",
        }) from error
