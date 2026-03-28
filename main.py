from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
from calculator import ModelCalculator

class EstimationRequest(BaseModel):
    input_tokens: int = 10000
    cached_tokens: int = 0
    output_tokens: int = 500

class ModelEstimation(BaseModel):
    model_name: str
    provider: str
    input_cost: float
    cache_cost: float
    output_cost: float
    total_cost: float
    estimated_latency_sec: float
    release_date: str

class Subscription(BaseModel):
    id: str
    name: str
    provider: str
    monthly_price: float
    included_models: List[str]
    is_unlimited: bool
    limit_note: Optional[str] = None

class EstimationResponse(BaseModel):
    estimations: List[ModelEstimation]

# Global calculator instance
calculator = ModelCalculator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Optional: Fetch live pricing on startup
    calculator.refresh_catalog(use_live_pricing=True)
    yield 

app = FastAPI(title="AI Run Estimator API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/subscriptions", response_model=List[Subscription])
async def get_subscriptions():
    return [Subscription(**s) for s in calculator.get_subscriptions()]

@app.post("/estimate", response_model=EstimationResponse)
async def calculate_metrics(request: EstimationRequest):
    results = calculator.estimate(
        request.input_tokens, 
        request.cached_tokens, 
        request.output_tokens
    )
    
    # Map to API response model
    return EstimationResponse(estimations=[ModelEstimation(**res) for res in results])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
