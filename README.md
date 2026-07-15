# Macro LLM Tournament

This repository now contains a runnable 200-household LLM economy. Real SCE
histories supply household priors, matched SCF records supply continuous financial
states, GPT-5.5 supplies household beliefs and conditional dollar policies, and
ordinary code enforces budgets and settlement. A small producer then turns demand
into output, inventory, aggregate employment, wages, and family income before the
same households decide again.

The latest result is an improvement, not a finished forecast. On four historical
monthly origins, the LLM household economy gets all four consumption directions
right and lowers RMSE from `0.701` to `0.508` percentage points. It still
underpredicts every increase, and a simple origin-visible spending drift remains
better at `0.243` points.

## Active Economy

```text
200 anonymized SCE histories + matched SCF financial states
                              +
                 public information known at the origin
                              |
                              v
                one isolated GPT-5.5 call per household
                              |
                              v
       beliefs + employed/not-employed policies in nominal dollars
                              |
                              v
          code-enforced household budgets, credit, and settlement
                              |
                              v
                    population-weighted demand
                              |
                              v
     producer output + inventory -> employment + wages -> family income
                              |
                              v
               fresh period-two household decisions
```

The producer is deliberately mechanical. Prior demand sets expected sales; it
closes 35% of the inventory gap, 25% of the aggregate labor gap, and 10% of the
employment-rate change into wages, capped at 2% per month. There are no LLM firms
or banks.

## Latest Diagnostic

| Origin | Target | LLM household economy | First-release PCE | Origin-visible drift |
| --- | --- | ---: | ---: | ---: |
| Jan 2026 | Feb 2026 | +0.15% | +0.48% | +0.37% |
| Feb 2026 | Mar 2026 | +0.26% | +0.90% | +0.51% |
| Mar 2026 | Apr 2026 | +0.11% | +0.51% | +0.38% |
| Apr 2026 | May 2026 | +0.12% | +0.71% | +0.48% |

- Consumption direction: **4/4**.
- Consumption RMSE: **0.508 pp**.
- Consumption correlation: **0.759**.
- Origin-visible drift RMSE: **0.243 pp**.
- Revolving-credit direction: **1/4**.

The frozen July-to-August forecast is **+0.19%**. August remains unscored. In the
unscored second period, settled aggregate employment rises `0.0013%`, the average
wage rises `0.00013%`, and fresh household spending rises **0.23%** from period
one. This proves the recursive mechanism executes; it is not a September forecast
or a causal estimate of the firm-feedback effect.

## Household Saving

The LLM no longer chooses deposits separately. Deposits are the cash residual
after income, spending, debt service, borrowing, and fixed outflows.

The current state also stops routing every dollar of household saving into liquid
deposits. Each household has a total-saving target from its existing saving-rate
field. Code uses deposits and the household's buffer target to close any liquid
shortfall gradually over twelve months; the rest is recorded with taxes, recurring
obligations, and non-deposit saving. Households whose matched expenditure exceeds
income retain an explicit cash deficit. In the 200-household cohort, 49.5% of
population weight has no baseline liquid-saving target.

## Run It

```bash
make ecology-fixture
make ecology-current-replay
make ecology-retrospective-replay
make ecology-feedback-replay
make ecology-observability
make check
make test
```

Fresh model calls use Codex CLI only:

```bash
make ecology-live-200
make ecology-retrospective-live
make ecology-feedback-live
```

Every model call runs in an empty directory without local-file, shell, web,
browser, app, memory, plugin, or subagent access. Replays bind the prompt, cards,
accepted payloads, consumed parent artifacts, input files, source revision, and
economic result.

## Evidence Boundary

- January-April 2026 is retrospective development evidence and may be in model
  knowledge.
- The July 2026 origin is frozen for August and remains unscored.
- Household budgets, goods inventory, bank stocks, and named counterparty flows
  reconcile. A firm balance sheet and full external-sector stocks are not modeled.
- The historical firm shadow and the recursive producer loop are distinct,
  separately named mechanisms.
- The remaining macro problem is amplitude: households move in the right direction
  but not far enough.

See [CURRENT.md](CURRENT.md) for the current milestone,
[reports/current_ecology_report.md](reports/current_ecology_report.md) for the
sendable result, and [research_history.md](research_history.md) for the full
experiment trail. The previous weighted-demand economy is recoverable at Git tag
`macro-v1-weighted-demand`.
