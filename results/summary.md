| arm | final test acc (mean ± 95% CI) | per seed | stale tokens clipped | dropped non-renderable | CPU-hours |
|---|---|---|---|---|---|
| A discard stale | 1.74 ± 0.33 | 1.82, 1.82, 1.59 | 1.02% | 0 | 23.9 |
| B naive reuse | 1.84 ± 1.15 | 2.35, 1.44, 1.74 | n/a (no IS) | 0 | 20.1 |
| C re-render, no IS | 1.97 ± 1.15 | 2.50, 1.67, 1.74 | n/a (no IS) | 0 | 19.8 |
| D harnesslag | 2.12 ± 1.00 | 2.58, 1.97, 1.82 | 1.37% | 0 | 20.4 |
| E re-render, one cap | 1.77 ± 0.76 | 2.12, 1.59, 1.59 | 1.02% | 0 | 20.4 |
