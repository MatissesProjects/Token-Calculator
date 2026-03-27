from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# 1. Initialize the FastAPI app
app = FastAPI(
    title="AI Run Estimator API",
    description="A callable tool to estimate costs and latency for AI model runs.",
    version="1.0.0"
)

# 2. Define our input data structure
class EstimationRequest(BaseModel):
    input_tokens: int = 10000
    cached_tokens: int = 0
    output_tokens: int = 500

# 3. Define our output data structures
class ModelEstimation(BaseModel):
    model_name: str
    input_cost: float
    cache_cost: float
    output_cost: float
    total_cost: float
    estimated_latency_sec: float

class EstimationResponse(BaseModel):
    estimations: List[ModelEstimation]

# 4. Store our Model Data (Prices per 1 Million tokens)
MODELS_DATA = [
    {
        "name": "GPT-4o",
        "input_price": 5.00,
        "output_price": 15.00,
        "cache_read_price": 2.50,
        "speed_tps": 70
    },
    {
        "name": "Claude 3.5 Sonnet",
        "input_price": 3.00,
        "output_price": 15.00,
        "cache_read_price": 0.30,
        "speed_tps": 80
    },
    {
        "name": "Gemini 1.5 Pro",
        "input_price": 3.50,
        "output_price": 10.50,
        "cache_read_price": 0.88,
        "speed_tps": 60
    }
]

# 5. Create the callable endpoint
@app.post("/estimate", response_model=EstimationResponse)
async def calculate_metrics(request: EstimationRequest):
    results = []
    
    # Ensure we don't calculate more cached tokens than input tokens
    actual_cached = min(request.cached_tokens, request.input_tokens)
    actual_fresh_input = request.input_tokens - actual_cached

    for model in MODELS_DATA:
        # Math (divided by 1M since prices are per million)
        input_cost = (actual_fresh_input / 1_000_000) * model["input_price"]
        cache_cost = (actual_cached / 1_000_000) * model["cache_read_price"]
        output_cost = (request.output_tokens / 1_000_000) * model["output_price"]
        total_cost = input_cost + cache_cost + output_cost
        
        # Time estimate based on output tokens
        estimated_time_sec = round(request.output_tokens / model["speed_tps"], 1)

        results.append(ModelEstimation(
            model_name=model["name"],
            input_cost=round(input_cost, 6),
            cache_cost=round(cache_cost, 6),
            output_cost=round(output_cost, 6),
            total_cost=round(total_cost, 6),
            estimated_latency_sec=estimated_time_sec
        ))

    return EstimationResponse(estimations=results)

# Run instructions for local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
