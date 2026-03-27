from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import requests

app = FastAPI(title="AI Run Estimator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EstimationRequest(BaseModel):
    input_tokens: int = 10000
    cached_tokens: int = 0
    output_tokens: int = 500

class ModelEstimation(BaseModel):
    model_name: str
    input_cost: float
    cache_cost: float
    output_cost: float
    total_cost: float
    estimated_latency_sec: float

class EstimationResponse(BaseModel):
    estimations: List[ModelEstimation]

# --- 1. THE FALLBACK CATALOG ---
# If the live API fails or changes IDs, these models will STILL show up.
# Prices are per 1 Million tokens.
BASE_CATALOG = {
    # Google Gemini Models
    "google/gemini-1.5-pro": {"name": "Gemini 1.5 Pro", "input_price": 3.50, "output_price": 10.50, "speed_tps": 60, "cache_discount": 0.50},
    "google/gemini-1.5-flash": {"name": "Gemini 1.5 Flash", "input_price": 0.075, "output_price": 0.30, "speed_tps": 120, "cache_discount": 0.50},
    "google/gemini-1.5-flash-8b": {"name": "Gemini 1.5 Flash-8B", "input_price": 0.0375, "output_price": 0.15, "speed_tps": 150, "cache_discount": 0.50},
    
    # OpenAI Models
    "openai/gpt-4o": {"name": "GPT-4o", "input_price": 2.50, "output_price": 10.00, "speed_tps": 70, "cache_discount": 0.50},
    "openai/gpt-4o-mini": {"name": "GPT-4o Mini", "input_price": 0.15, "output_price": 0.60, "speed_tps": 130, "cache_discount": 0.50},
    "openai/o1-preview": {"name": "o1-preview", "input_price": 15.00, "output_price": 60.00, "speed_tps": 30, "cache_discount": 0.50},
    "openai/o1-mini": {"name": "o1-mini", "input_price": 3.00, "output_price": 12.00, "speed_tps": 50, "cache_discount": 0.50},

    # Anthropic Models
    "anthropic/claude-3.5-sonnet": {"name": "Claude 3.5 Sonnet", "input_price": 3.00, "output_price": 15.00, "speed_tps": 80, "cache_discount": 0.10},
    "anthropic/claude-3.5-haiku": {"name": "Claude 3.5 Haiku", "input_price": 0.80, "output_price": 4.00, "speed_tps": 140, "cache_discount": 0.10},
    "anthropic/claude-3-opus": {"name": "Claude 3 Opus", "input_price": 15.00, "output_price": 75.00, "speed_tps": 40, "cache_discount": 0.10},
}

# This will hold our final data (Live + Fallbacks)
ACTIVE_MODELS_DATA = []

def fetch_live_pricing():
    print("Fetching live pricing from OpenRouter...")
    
    # Start with a copy of our fallback catalog
    working_catalog = dict(BASE_CATALOG)
    
    try:
        response = requests.get("https://openrouter.ai/api/v1/models", timeout=5)
        api_models = response.json().get("data", [])
        
        # Overwrite fallback prices with live prices if we find a match
        for api_model in api_models:
            model_id = api_model["id"]
            if model_id in working_catalog:
                live_input = float(api_model["pricing"]["prompt"]) * 1_000_000
                live_output = float(api_model["pricing"]["completion"]) * 1_000_000
                
                # Only update if the API gives us valid numbers > 0
                if live_input > 0 or live_output > 0:
                    working_catalog[model_id]["input_price"] = live_input
                    working_catalog[model_id]["output_price"] = live_output
                    print(f"Updated {working_catalog[model_id]['name']} with live pricing.")
                    
    except Exception as e:
        print(f"Live pricing fetch failed, using fallback defaults. Error: {e}")

    # Convert the dictionary back into a list for the endpoint to use
    final_list = []
    for key, data in working_catalog.items():
        data["cache_read_price"] = data["input_price"] * data["cache_discount"]
        final_list.append(data)
        
    return final_list

@app.on_event("startup")
async def startup_event():
    global ACTIVE_MODELS_DATA
    ACTIVE_MODELS_DATA = fetch_live_pricing()

@app.post("/estimate", response_model=EstimationResponse)
async def calculate_metrics(request: EstimationRequest):
    results = []
    actual_cached = min(request.cached_tokens, request.input_tokens)
    actual_fresh_input = request.input_tokens - actual_cached

    for model in ACTIVE_MODELS_DATA:
        input_cost = (actual_fresh_input / 1_000_000) * model["input_price"]
        cache_cost = (actual_cached / 1_000_000) * model["cache_read_price"]
        output_cost = (request.output_tokens / 1_000_000) * model["output_price"]
        total_cost = input_cost + cache_cost + output_cost
        
        estimated_time_sec = round(request.output_tokens / model["speed_tps"], 1)

        results.append(ModelEstimation(
            model_name=model["name"],
            input_cost=round(input_cost, 6),
            cache_cost=round(cache_cost, 6),
            output_cost=round(output_cost, 6),
            total_cost=round(total_cost, 6),
            estimated_latency_sec=estimated_time_sec
        ))

    # Sort the results so the cheapest model appears first on your webpage
    results.sort(key=lambda x: x.total_cost)

    return EstimationResponse(estimations=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
