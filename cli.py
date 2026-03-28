import argparse
import sys
from calculator import ModelCalculator

def main():
    parser = argparse.ArgumentParser(description="AI Token Cost Estimator CLI")
    parser.add_argument("-i", "--input", type=int, default=100000, help="Input tokens (default: 100k)")
    parser.add_argument("-c", "--cached", type=int, default=0, help="Cached tokens (default: 0)")
    parser.add_argument("-o", "--output", type=int, default=5000, help="Output tokens (default: 5k)")
    parser.add_argument("-l", "--live", action="store_true", help="Fetch live pricing from OpenRouter")
    parser.add_argument("-t", "--top", type=int, default=10, help="Show top N cheapest models (default: 10)")
    parser.add_argument("-p", "--provider", type=str, help="Filter by provider (e.g., OpenAI, Google)")

    args = parser.parse_args()

    calc = ModelCalculator()
    if args.live:
        print("Fetching live prices...", file=sys.stderr)
        calc.refresh_catalog(use_live_pricing=True)

    results = calc.estimate(args.input, args.cached, args.output)

    if args.provider:
        results = [r for r in results if args.provider.lower() in r["provider"].lower()]

    print(f"\nEstimation for {args.input:,} input ({args.cached:,} cached) and {args.output:,} output tokens:")
    print("-" * 100)
    print(f"{'Provider':<15} {'Model Name':<35} {'Total Cost':<12} {'Latency':<10}")
    print("-" * 100)

    for res in results[:args.top]:
        print(f"{res['provider']:<15} {res['model_name']:<35} ${res['total_cost']:<11.4f} {res['estimated_latency_sec']:>6.1f}s")

if __name__ == "__main__":
    main()
