import json
import requests
import os
from typing import List, Dict, Optional

# Constants
DEFAULT_MODELS_PATH = os.path.join(os.path.dirname(__file__), "models.json")
DEFAULT_SUBS_PATH = os.path.join(os.path.dirname(__file__), "subscriptions.json")
OPENROUTER_URL = "https://openrouter.ai/api/v1/models"

class ModelCalculator:
    def __init__(self, models_path: str = DEFAULT_MODELS_PATH, subs_path: str = DEFAULT_SUBS_PATH):
        self.models_path = models_path
        self.subs_path = subs_path
        self.raw_catalog = self._load_json(self.models_path)
        self.subscriptions = self._load_json(self.subs_path)
        self.active_models = []
        self.refresh_catalog(use_live_pricing=False)

    def _load_json(self, path: str) -> Dict:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def get_subscriptions(self) -> List[Dict]:
        return [{"id": k, **v} for k, v in self.subscriptions.items()]

    def refresh_catalog(self, use_live_pricing: bool = False):
        """Builds the active model list, optionally fetching live prices."""
        working_catalog = dict(self.raw_catalog)
        
        if use_live_pricing:
            try:
                response = requests.get(OPENROUTER_URL, timeout=5)
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
                # Silently fail and use fallbacks
                pass

        final_list = []
        for key, data in working_catalog.items():
            data["id"] = key
            # Advanced caching logic: Use explicit factors if they exist
            # cache_read_factor: 1.0 (no discount), 0.5 (50%), 0.1 (10%)
            # cache_write_factor: 1.0 (no surcharge), 1.25 (25% surcharge)
            data["cache_read_price"] = data["input_price"] * data.get("cache_read_factor", data.get("cache_discount", 1.0))
            data["cache_write_price"] = data["input_price"] * data.get("cache_write_factor", 1.0)
            
            if "release_date" not in data:
                data["release_date"] = "2024-01"
            final_list.append(data)
        
        self.active_models = final_list

    def estimate(self, input_tokens: int, cache_read_tokens: int, output_tokens: int, cache_write_tokens: int = 0) -> List[Dict]:
        """Calculates cost and latency for all supported models."""
        results = []
        
        # total_tokens is the sum of fresh input, read cache, and write cache.
        # But usually 'input_tokens' represents the total prompt size.
        # Let's adjust for consistency: actual_fresh = input - read - write
        actual_read = min(cache_read_tokens, input_tokens)
        actual_write = min(cache_write_tokens, input_tokens - actual_read)
        actual_fresh_input = input_tokens - actual_read - actual_write
        
        total_context = input_tokens + output_tokens

        for model in self.active_models:
            max_ctx = model.get("max_context", float('inf'))
            # Still filter by hard limits if they exist (Gemini context-based pricing)
            min_ctx = model.get("min_context", 0)
            if not (min_ctx <= total_context):
                continue

            input_cost = (actual_fresh_input / 1_000_000) * model["input_price"]
            read_cost = (actual_read / 1_000_000) * model["cache_read_price"]
            write_cost = (actual_write / 1_000_000) * model["cache_write_price"]
            output_cost = (output_tokens / 1_000_000) * model["output_price"]
            total_cost = input_cost + read_cost + write_cost + output_cost
            
            speed = model.get("speed_tps", 40)
            latency = round(output_tokens / speed, 1)

            results.append({
                "model_id": model["id"],
                "model_name": model["name"],
                "provider": model["provider"],
                "input_cost": round(input_cost, 6),
                "cache_cost": round(read_cost + write_cost, 6), # Combined for simplicity in UI
                "output_cost": round(output_cost, 6),
                "total_cost": round(total_cost, 6),
                "estimated_latency_sec": latency,
                "release_date": model["release_date"],
                "max_context": max_ctx,
                "is_too_big": total_context > max_ctx
            })

        # Sort by total cost by default
        return sorted(results, key=lambda x: x["total_cost"])
