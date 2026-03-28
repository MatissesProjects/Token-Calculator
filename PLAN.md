# AI Token Calculator Plan

- [x] Phase 1: Making it "Callable" (CLI & Library)
    - [x] Extract core logic to `calculator.py`
    - [x] Externalize `models.json`
    - [x] Create `cli.py` for terminal usage
    - [x] Refactor `main.py` backend

- [x] Phase 2: Tool/Service Presets
    - [x] Add Subscription Comparison (Cursor, Copilot, etc.)
    - [x] Batch Estimation Scaling (1x, 100x, 1000x)

- [x] Phase 3: Advanced Pricing Logic
    - [x] Anthropic Cache Write vs Read logic
    - [x] Context Window Warnings
    - [x] Cached tokens additive to fresh input tokens

- [x] Phase 4: Data & Maintenance
    - [x] Improved Live Sync auto-discovery

- [ ] Phase 5: Visual Cost Analysis & Comparison
    - [ ] Add Break-even Chart (Chart.js)
    - [ ] Locked Comparison View (Pinning)