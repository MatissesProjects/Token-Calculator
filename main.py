from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import requests
import math

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
    provider: str
    input_cost: float
    cache_cost: float
    output_cost: float
    total_cost: float
    estimated_latency_sec: float

class EstimationResponse(BaseModel):
    estimations: List[ModelEstimation]

# --- 1. THE FALLBACK CATALOG WITH CONTEXT LIMITS ---
# Added 'min_context' and 'max_context' for tiered models. 
# If not specified, they default to 0 and infinity respectively.
BASE_CATALOG = {
    # ------------------ AMAZON ------------------
    "amazon-nova-micro": {"name": "Amazon Nova Micro", "provider": "Amazon", "input_price": 0.035, "output_price": 0.14, "speed_tps": 120, "cache_discount": 1.0},
    "amazon-nova-lite": {"name": "Amazon Nova Lite", "provider": "Amazon", "input_price": 0.06, "output_price": 0.24, "speed_tps": 120, "cache_discount": 1.0},
    "amazon-nova-pro": {"name": "Amazon Nova Pro", "provider": "Amazon", "input_price": 0.80, "output_price": 3.20, "speed_tps": 40, "cache_discount": 1.0},
    "amazon-nova-premier": {"name": "Amazon Nova Premier", "provider": "Amazon", "input_price": 2.50, "output_price": 12.50, "speed_tps": 40, "cache_discount": 1.0},

    # ------------------ GOOGLE ------------------
    "gemini-1.5-flash-8b": {"name": "Gemini 1.5 Flash-8B ≤128k", "provider": "Google", "input_price": 0.0375, "output_price": 0.15, "speed_tps": 120, "cache_discount": 1.0, "max_context": 128000},
    "gemini-1.5-flash-8b-128k": {"name": "Gemini 1.5 Flash-8B >128k", "provider": "Google", "input_price": 0.075, "output_price": 0.30, "speed_tps": 120, "cache_discount": 1.0, "min_context": 128001},
    "gemini-1.5-flash": {"name": "Gemini 1.5 Flash ≤128k", "provider": "Google", "input_price": 0.075, "output_price": 0.30, "speed_tps": 120, "cache_discount": 1.0, "max_context": 128000},
    "gemini-1.5-flash-128k": {"name": "Gemini 1.5 Flash >128k", "provider": "Google", "input_price": 0.15, "output_price": 0.60, "speed_tps": 120, "cache_discount": 1.0, "min_context": 128001},
    "gemini-1.5-pro": {"name": "Gemini 1.5 Pro ≤128k", "provider": "Google", "input_price": 1.25, "output_price": 5.00, "speed_tps": 40, "cache_discount": 1.0, "max_context": 128000},
    "gemini-1.5-pro-128k": {"name": "Gemini 1.5 Pro >128k", "provider": "Google", "input_price": 2.50, "output_price": 10.00, "speed_tps": 40, "cache_discount": 1.0, "min_context": 128001},
    "gemini-2.0-flash-lite": {"name": "Gemini 2.0 Flash Lite", "provider": "Google", "input_price": 0.075, "output_price": 0.30, "speed_tps": 120, "cache_discount": 1.0},
    "gemini-2.0-flash": {"name": "Gemini 2.0 Flash", "provider": "Google", "input_price": 0.10, "output_price": 0.40, "speed_tps": 120, "cache_discount": 1.0},
    "gemini-2.5-flash-lite": {"name": "Gemini 2.5 Flash-Lite", "provider": "Google", "input_price": 0.10, "output_price": 0.40, "speed_tps": 120, "cache_discount": 0.10},
    "gemini-2.5-flash": {"name": "Gemini 2.5 Flash", "provider": "Google", "input_price": 0.30, "output_price": 2.50, "speed_tps": 120, "cache_discount": 0.10},
    "gemini-2.5-flash-preview-09-2025": {"name": "Gemini 2.5 Flash Preview (09-2025)", "provider": "Google", "input_price": 0.30, "output_price": 2.50, "speed_tps": 120, "cache_discount": 0.10},
    "gemini-2.5-pro-preview-03-25": {"name": "Gemini 2.5 Pro Preview ≤200k", "provider": "Google", "input_price": 1.25, "output_price": 10.00, "speed_tps": 40, "cache_discount": 1.0, "max_context": 200000},
    "gemini-2.5-pro-preview-03-25-200k": {"name": "Gemini 2.5 Pro Preview >200k", "provider": "Google", "input_price": 2.50, "output_price": 15.00, "speed_tps": 40, "cache_discount": 1.0, "min_context": 200001},
    "gemini-2.5-pro": {"name": "Gemini 2.5 Pro ≤200k", "provider": "Google", "input_price": 1.25, "output_price": 10.00, "speed_tps": 40, "cache_discount": 0.10, "max_context": 200000},
    "gemini-2.5-pro-200k": {"name": "Gemini 2.5 Pro >200k", "provider": "Google", "input_price": 2.50, "output_price": 15.00, "speed_tps": 40, "cache_discount": 0.10, "min_context": 200001},
    "gemini-3-flash-preview": {"name": "Gemini 3 Flash Preview", "provider": "Google", "input_price": 0.50, "output_price": 3.00, "speed_tps": 120, "cache_discount": 1.0},
    "gemini-3.1-flash-lite-preview": {"name": "Gemini 3.1 Flash-Lite", "provider": "Google", "input_price": 0.25, "output_price": 1.50, "speed_tps": 120, "cache_discount": 0.10},
    "gemini-3-pro-preview": {"name": "Gemini 3 Pro ≤200k", "provider": "Google", "input_price": 2.00, "output_price": 12.00, "speed_tps": 40, "cache_discount": 1.0, "max_context": 200000},
    "gemini-3-pro-preview-200k": {"name": "Gemini 3 Pro >200k", "provider": "Google", "input_price": 4.00, "output_price": 18.00, "speed_tps": 40, "cache_discount": 1.0, "min_context": 200001},
    "gemini-3-1-pro-preview": {"name": "Gemini 3.1 Pro ≤200k", "provider": "Google", "input_price": 2.00, "output_price": 12.00, "speed_tps": 40, "cache_discount": 1.0, "max_context": 200000},
    "gemini-3-1-pro-preview-200k": {"name": "Gemini 3.1 Pro >200k", "provider": "Google", "input_price": 4.00, "output_price": 18.00, "speed_tps": 40, "cache_discount": 1.0, "min_context": 200001},

    # ------------------ OPENAI ------------------
    "gpt-4o-mini": {"name": "GPT-4o Mini", "provider": "OpenAI", "input_price": 0.15, "output_price": 0.60, "speed_tps": 120, "cache_discount": 0.50},
    "gpt-4o": {"name": "GPT-4o", "provider": "OpenAI", "input_price": 2.50, "output_price": 10.00, "speed_tps": 80, "cache_discount": 0.50},
    "chatgpt-4o-latest": {"name": "ChatGPT 4o Latest", "provider": "OpenAI", "input_price": 5.00, "output_price": 15.00, "speed_tps": 80, "cache_discount": 1.0},
    "gpt-4.1-nano": {"name": "GPT-4.1 Nano", "provider": "OpenAI", "input_price": 0.10, "output_price": 0.40, "speed_tps": 120, "cache_discount": 0.25},
    "gpt-4.1-mini": {"name": "GPT-4.1 Mini", "provider": "OpenAI", "input_price": 0.40, "output_price": 1.60, "speed_tps": 120, "cache_discount": 0.25},
    "gpt-4.1": {"name": "GPT-4.1", "provider": "OpenAI", "input_price": 2.00, "output_price": 8.00, "speed_tps": 80, "cache_discount": 0.25},
    "gpt-4.5": {"name": "GPT-4.5", "provider": "OpenAI", "input_price": 75.00, "output_price": 150.00, "speed_tps": 40, "cache_discount": 0.50},
    "gpt-5-nano": {"name": "GPT-5 Nano", "provider": "OpenAI", "input_price": 0.05, "output_price": 0.40, "speed_tps": 120, "cache_discount": 0.10},
    "gpt-5-mini": {"name": "GPT-5 Mini", "provider": "OpenAI", "input_price": 0.25, "output_price": 2.00, "speed_tps": 120, "cache_discount": 0.10},
    "gpt-5": {"name": "GPT-5", "provider": "OpenAI", "input_price": 1.25, "output_price": 10.00, "speed_tps": 60, "cache_discount": 0.10},
    "gpt-5-pro": {"name": "GPT-5 Pro", "provider": "OpenAI", "input_price": 15.00, "output_price": 120.00, "speed_tps": 40, "cache_discount": 1.0},
    "gpt-5.1-codex-mini": {"name": "GPT-5.1 Codex mini", "provider": "OpenAI", "input_price": 0.25, "output_price": 2.00, "speed_tps": 120, "cache_discount": 0.10},
    "gpt-5.1-codex": {"name": "GPT-5.1 Codex", "provider": "OpenAI", "input_price": 1.25, "output_price": 10.00, "speed_tps": 80, "cache_discount": 0.10},
    "gpt-5.1": {"name": "GPT-5.1", "provider": "OpenAI", "input_price": 1.25, "output_price": 10.00, "speed_tps": 60, "cache_discount": 0.10},
    "gpt-5.2": {"name": "GPT-5.2", "provider": "OpenAI", "input_price": 1.75, "output_price": 14.00, "speed_tps": 60, "cache_discount": 0.10},
    "gpt-5.2-pro": {"name": "GPT-5.2 Pro", "provider": "OpenAI", "input_price": 21.00, "output_price": 168.00, "speed_tps": 40, "cache_discount": 1.0},
    "gpt-5.4-nano": {"name": "GPT-5.4 Nano", "provider": "OpenAI", "input_price": 0.20, "output_price": 1.25, "speed_tps": 120, "cache_discount": 0.10},
    "gpt-5.4-mini": {"name": "GPT-5.4 Mini", "provider": "OpenAI", "input_price": 0.75, "output_price": 4.50, "speed_tps": 120, "cache_discount": 0.10},
    "gpt-5.4": {"name": "GPT-5.4 ≤272k", "provider": "OpenAI", "input_price": 2.50, "output_price": 15.00, "speed_tps": 60, "cache_discount": 0.10, "max_context": 272000},
    "gpt-5.4-272k": {"name": "GPT-5.4 >272k", "provider": "OpenAI", "input_price": 5.00, "output_price": 22.50, "speed_tps": 60, "cache_discount": 0.10, "min_context": 272001},
    "gpt-5.4-pro": {"name": "GPT-5.4 Pro ≤272k", "provider": "OpenAI", "input_price": 30.00, "output_price": 180.00, "speed_tps": 40, "cache_discount": 1.0, "max_context": 272000},
    "gpt-5.4-pro-272k": {"name": "GPT-5.4 Pro >272k", "provider": "OpenAI", "input_price": 60.00, "output_price": 270.00, "speed_tps": 40, "cache_discount": 1.0, "min_context": 272001},
    "gpt-image-1-mini": {"name": "gpt-image-1-mini", "provider": "OpenAI", "input_price": 2.00, "output_price": 8.00, "speed_tps": 120, "cache_discount": 0.10},
    "gpt-image-1": {"name": "gpt-image-1", "provider": "OpenAI", "input_price": 10.00, "output_price": 40.00, "speed_tps": 80, "cache_discount": 0.125},
    "o1-mini": {"name": "o1-mini", "provider": "OpenAI", "input_price": 1.10, "output_price": 4.40, "speed_tps": 40, "cache_discount": 0.50},
    "o1-preview": {"name": "o1 and o1-preview", "provider": "OpenAI", "input_price": 15.00, "output_price": 60.00, "speed_tps": 40, "cache_discount": 0.50},
    "o1-pro": {"name": "o1 Pro", "provider": "OpenAI", "input_price": 150.00, "output_price": 600.00, "speed_tps": 20, "cache_discount": 1.0},
    "o3-mini": {"name": "o3-mini", "provider": "OpenAI", "input_price": 1.10, "output_price": 4.40, "speed_tps": 40, "cache_discount": 0.50},
    "o3": {"name": "o3", "provider": "OpenAI", "input_price": 10.00, "output_price": 40.00, "speed_tps": 40, "cache_discount": 0.05},
    "o3-deep-research": {"name": "o3 Deep Research", "provider": "OpenAI", "input_price": 10.00, "output_price": 40.00, "speed_tps": 40, "cache_discount": 0.25},
    "o3-pro": {"name": "o3 Pro", "provider": "OpenAI", "input_price": 20.00, "output_price": 80.00, "speed_tps": 20, "cache_discount": 1.0},
    "o4-mini": {"name": "o4-mini", "provider": "OpenAI", "input_price": 1.10, "output_price": 4.40, "speed_tps": 40, "cache_discount": 0.25},
    "o4-mini-deep-research": {"name": "o4-mini Deep Research", "provider": "OpenAI", "input_price": 2.00, "output_price": 8.00, "speed_tps": 40, "cache_discount": 0.25},
    "text-davinci-003": {"name": "GPT-3 Text Davinci 003", "provider": "OpenAI", "input_price": 20.00, "output_price": 20.00, "speed_tps": 80, "cache_discount": 1.0},

    # ------------------ ANTHROPIC ------------------
    "claude-3-haiku": {"name": "Claude 3 Haiku", "provider": "Anthropic", "input_price": 0.25, "output_price": 1.25, "speed_tps": 120, "cache_discount": 1.0},
    "claude-3.5-haiku": {"name": "Claude 3.5 Haiku", "provider": "Anthropic", "input_price": 0.80, "output_price": 4.00, "speed_tps": 120, "cache_discount": 1.0},
    "claude-4.5-haiku": {"name": "Claude 4.5 Haiku", "provider": "Anthropic", "input_price": 1.00, "output_price": 5.00, "speed_tps": 120, "cache_discount": 1.0},
    "claude-3.5-sonnet": {"name": "Claude 3.5 Sonnet", "provider": "Anthropic", "input_price": 3.00, "output_price": 15.00, "speed_tps": 80, "cache_discount": 1.0},
    "claude-3.7-sonnet": {"name": "Claude 3.7 Sonnet", "provider": "Anthropic", "input_price": 3.00, "output_price": 15.00, "speed_tps": 80, "cache_discount": 1.0},
    "claude-sonnet-4.5": {"name": "Claude Sonnet 4 and 4.5 ≤200k", "provider": "Anthropic", "input_price": 3.00, "output_price": 15.00, "speed_tps": 80, "cache_discount": 1.0, "max_context": 200000},
    "claude-sonnet-4.5-200k": {"name": "Claude Sonnet 4.5 >200k", "provider": "Anthropic", "input_price": 6.00, "output_price": 22.50, "speed_tps": 80, "cache_discount": 1.0, "min_context": 200001},
    "claude-3-opus": {"name": "Claude 3 Opus", "provider": "Anthropic", "input_price": 15.00, "output_price": 75.00, "speed_tps": 40, "cache_discount": 1.0},
    "claude-opus-4": {"name": "Claude Opus 4", "provider": "Anthropic", "input_price": 15.00, "output_price": 75.00, "speed_tps": 40, "cache_discount": 1.0},
    "claude-opus-4-1": {"name": "Claude Opus 4.1", "provider": "Anthropic", "input_price": 15.00, "output_price": 75.00, "speed_tps": 40, "cache_discount": 1.0},
    "claude-opus-4-5": {"name": "Claude Opus 4.5", "provider": "Anthropic", "input_price": 5.00, "output_price": 25.00, "speed_tps": 40, "cache_discount": 1.0},

    # ------------------ MISTRAL ------------------
    "ministral-3b-latest": {"name": "Ministral 3B", "provider": "Mistral", "input_price": 0.04, "output_price": 0.04, "speed_tps": 120, "cache_discount": 1.0},
    "ministral-8b-latest": {"name": "Ministral 8B", "provider": "Mistral", "input_price": 0.10, "output_price": 0.10, "speed_tps": 120, "cache_discount": 1.0},
    "mistral-small-latest": {"name": "Mistral Small 3.1", "provider": "Mistral", "input_price": 0.10, "output_price": 0.30, "speed_tps": 80, "cache_discount": 1.0},
    "pixtral-12b": {"name": "Pixtral 12B", "provider": "Mistral", "input_price": 0.15, "output_price": 0.15, "speed_tps": 120, "cache_discount": 1.0},
    "mistral-nemo": {"name": "Mistral NeMo", "provider": "Mistral", "input_price": 0.15, "output_price": 0.15, "speed_tps": 80, "cache_discount": 1.0},
    "mistral-saba-latest": {"name": "Mistral Saba", "provider": "Mistral", "input_price": 0.20, "output_price": 0.60, "speed_tps": 80, "cache_discount": 1.0},
    "open-mistral-7b": {"name": "Mistral 7B", "provider": "Mistral", "input_price": 0.25, "output_price": 0.25, "speed_tps": 120, "cache_discount": 1.0},
    "codestral-latest": {"name": "Codestral", "provider": "Mistral", "input_price": 0.30, "output_price": 0.90, "speed_tps": 80, "cache_discount": 1.0},
    "mistral-medium-2505": {"name": "Mistral Medium 3", "provider": "Mistral", "input_price": 0.40, "output_price": 2.00, "speed_tps": 80, "cache_discount": 1.0},
    "open-mixtral-8x7b": {"name": "Mixtral 8x7B", "provider": "Mistral", "input_price": 0.70, "output_price": 0.70, "speed_tps": 120, "cache_discount": 1.0},
    "open-mixtral-8x22b": {"name": "Mixtral 8x22B", "provider": "Mistral", "input_price": 2.00, "output_price": 6.00, "speed_tps": 40, "cache_discount": 1.0},
    "mistral-large-latest": {"name": "Mistral Large 24.11", "provider": "Mistral", "input_price": 2.00, "output_price": 6.00, "speed_tps": 40, "cache_discount": 1.0},
    "pixtral-large-latest": {"name": "Pixtral Large", "provider": "Mistral", "input_price": 2.00, "output_price": 6.00, "speed_tps": 40, "cache_discount": 1.0},
    "magistral-medium-latest": {"name": "Magistral Medium", "provider": "Mistral", "input_price": 2.00, "output_price": 5.00, "speed_tps": 80, "cache_discount": 1.0},

    # ------------------ XAI (GROK) ------------------
    "grok-4-fast": {"name": "Grok 4 Fast ≤128k", "provider": "xAI", "input_price": 0.20, "output_price": 0.50, "speed_tps": 120, "cache_discount": 0.25, "max_context": 128000},
    "grok-4-fast-reasoning": {"name": "Grok 4 Fast Reasoning ≤128k", "provider": "xAI", "input_price": 0.20, "output_price": 0.50, "speed_tps": 40, "cache_discount": 0.25, "max_context": 128000},
    "grok-code-fast-1": {"name": "Grok Code Fast 1", "provider": "xAI", "input_price": 0.20, "output_price": 1.50, "speed_tps": 120, "cache_discount": 0.10},
    "grok-3-mini": {"name": "Grok 3 Mini", "provider": "xAI", "input_price": 0.30, "output_price": 0.50, "speed_tps": 120, "cache_discount": 0.25},
    "grok-4-fast-128k": {"name": "Grok 4 Fast >128k", "provider": "xAI", "input_price": 0.40, "output_price": 1.00, "speed_tps": 120, "cache_discount": 0.125, "min_context": 128001},
    "grok-4-fast-reasoning-128k": {"name": "Grok 4 Fast Reasoning >128k", "provider": "xAI", "input_price": 0.40, "output_price": 1.00, "speed_tps": 40, "cache_discount": 0.125, "min_context": 128001},
    "grok-3": {"name": "Grok 3", "provider": "xAI", "input_price": 3.00, "output_price": 15.00, "speed_tps": 60, "cache_discount": 0.25},
    "grok-4": {"name": "Grok 4 ≤128k", "provider": "xAI", "input_price": 3.00, "output_price": 15.00, "speed_tps": 60, "cache_discount": 0.25, "max_context": 128000},
    "grok-4-128k": {"name": "Grok 4 >128k", "provider": "xAI", "input_price": 6.00, "output_price": 30.00, "speed_tps": 60, "cache_discount": 0.125, "min_context": 128001},

    # ------------------ MOONSHOT (KIMI) ------------------
    "kimi-k2-0905-preview": {"name": "Kimi K2 0905 Preview", "provider": "Moonshot", "input_price": 0.60, "output_price": 2.50, "speed_tps": 80, "cache_discount": 0.25},
    "kimi-k2-0711-preview": {"name": "Kimi K2 0711 Preview", "provider": "Moonshot", "input_price": 0.60, "output_price": 2.50, "speed_tps": 80, "cache_discount": 0.25},
    "kimi-k2-thinking": {"name": "Kimi K2 Thinking", "provider": "Moonshot", "input_price": 0.60, "output_price": 2.50, "speed_tps": 40, "cache_discount": 0.25},
    "kimi-k2-turbo-preview": {"name": "Kimi K2 Turbo Preview", "provider": "Moonshot", "input_price": 1.15, "output_price": 8.00, "speed_tps": 120, "cache_discount": 0.1304},
    "kimi-k2-thinking-turbo": {"name": "Kimi K2 Thinking Turbo", "provider": "Moonshot", "input_price": 1.15, "output_price": 8.00, "speed_tps": 80, "cache_discount": 0.1304},

    # ------------------ OTHER ------------------
    "deepseek-chat": {"name": "DeepSeek Chat", "provider": "DeepSeek", "input_price": 0.27, "output_price": 1.10, "speed_tps": 80, "cache_discount": 1.0},
    "deepseek-reasoner": {"name": "DeepSeek Reasoner", "provider": "DeepSeek", "input_price": 0.55, "output_price": 2.19, "speed_tps": 40, "cache_discount": 1.0},
    "minimax-m2": {"name": "MiniMax M2", "provider": "MiniMax", "input_price": 0.30, "output_price": 1.20, "speed_tps": 80, "cache_discount": 1.0},
}

ACTIVE_MODELS_DATA = []

def fetch_live_pricing():
    print("Fetching live pricing from OpenRouter...")
    working_catalog = dict(BASE_CATALOG)
    
    try:
        response = requests.get("https://openrouter.ai/api/v1/models", timeout=5)
        api_models = response.json().get("data", [])
        
        for api_model in api_models:
            model_id = api_model["id"]
            if model_id in working_catalog:
                live_input = float(api_model["pricing"]["prompt"]) * 1_000_000
                live_output = float(api_model["pricing"]["completion"]) * 1_000_000
                
                if live_input > 0 or live_output > 0:
                    working_catalog[model_id]["input_price"] = live_input
                    working_catalog[model_id]["output_price"] = live_output
                    
    except Exception as e:
        print(f"Live pricing fetch failed, using fallback defaults. Error: {e}")

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
    
    # Calculate total context size used
    total_context_used = request.input_tokens + request.output_tokens

    for model in ACTIVE_MODELS_DATA:
        # Check against context limits. If no limit is set in the dict, it defaults to 0 and infinity.
        min_context = model.get("min_context", 0)
        max_context = model.get("max_context", float('inf'))

        # Skip this model entirely if the total tokens fall outside its allowed tier
        if not (min_context <= total_context_used <= max_context):
            continue

        input_cost = (actual_fresh_input / 1_000_000) * model["input_price"]
        cache_cost = (actual_cached / 1_000_000) * model["cache_read_price"]
        output_cost = (request.output_tokens / 1_000_000) * model["output_price"]
        total_cost = input_cost + cache_cost + output_cost
        
        estimated_time_sec = round(request.output_tokens / model["speed_tps"], 1)

        results.append(ModelEstimation(
            model_name=model["name"],
            provider=model["provider"],
            input_cost=round(input_cost, 6),
            cache_cost=round(cache_cost, 6),
            output_cost=round(output_cost, 6),
            total_cost=round(total_cost, 6),
            estimated_latency_sec=estimated_time_sec
        ))

    results.sort(key=lambda x: x.total_cost)
    return EstimationResponse(estimations=results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
