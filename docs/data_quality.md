# Data quality report

- Snapshot: `32bc5db46626acc8`  
- Sample: 2002-07-01 → 2026-09-30 (6102 sessions)  
- Research window used for statistics: up to 2022-12-31  
- Holdout (from 2023-01-01): **structural checks only**, no return statistics, to keep it sealed.

Total-return index reconstructed from split-adjusted close + ex-date distributions; cross-checked against vendor Adj Close at fetch time (fetch fails above 0.5% daily divergence).

## Research window — per asset

| ticker | class | first_obs | investable_from | n_obs | internal_nans | zero_vol_days | cagr | vol | max_dd | median_adv_musd |
|---|---|---|---|---|---|---|---|---|---|---|
| SPY | equity | 2002-07-01 | 2003-06-30 | 5163 | 0 | 0 | 0.08999 | 0.1953 | -0.552 | 2.037e+04 |
| IWM | equity | 2002-07-01 | 2003-06-30 | 5163 | 0 | 0 | 0.08344 | 0.2428 | -0.5863 | 3717 |
| EFA | equity | 2002-07-01 | 2003-06-30 | 5163 | 0 | 0 | 0.0538 | 0.2169 | -0.6105 | 1082 |
| EWJ | equity | 2002-07-01 | 2003-06-30 | 5163 | 0 | 0 | 0.03723 | 0.2111 | -0.5445 | 301.7 |
| VGK | equity | 2005-03-10 | 2006-03-09 | 4485 | 0 | 0 | 0.0425 | 0.2398 | -0.6366 | 151 |
| EEM | equity | 2003-04-14 | 2004-04-13 | 4965 | 0 | 0 | 0.08354 | 0.2834 | -0.6645 | 2169 |
| SHY | government_bond | 2002-07-30 | 2003-07-29 | 5143 | 0 | 0 | 0.01661 | 0.01427 | -0.05707 | 91.64 |
| IEF | government_bond | 2002-07-30 | 2003-07-29 | 5143 | 0 | 0 | 0.03696 | 0.06749 | -0.2265 | 101.5 |
| TLT | government_bond | 2002-07-30 | 2003-07-29 | 5143 | 0 | 0 | 0.04415 | 0.1427 | -0.4414 | 855.1 |
| AGG | government_bond | 2003-09-29 | 2004-09-28 | 4849 | 0 | 0 | 0.02899 | 0.05068 | -0.1844 | 125.8 |
| TIP | inflation_linked | 2003-12-05 | 2004-12-06 | 4801 | 0 | 0 | 0.03461 | 0.06302 | -0.144 | 104.8 |
| LQD | credit | 2002-07-30 | 2003-07-29 | 5143 | 0 | 0 | 0.04522 | 0.08458 | -0.2496 | 193.3 |
| HYG | credit | 2007-04-11 | 2008-04-09 | 3961 | 0 | 0 | 0.04158 | 0.1172 | -0.3428 | 614.4 |
| EMB | credit | 2007-12-19 | 2008-12-17 | 3785 | 0 | 1 | 0.0366 | 0.1132 | -0.347 | 123.7 |
| VNQ | real_estate | 2004-09-29 | 2005-09-27 | 4597 | 0 | 0 | 0.07391 | 0.3003 | -0.7307 | 215.9 |
| GLD | commodity | 2004-11-18 | 2005-11-16 | 4561 | 0 | 0 | 0.07692 | 0.1786 | -0.4556 | 1093 |
| DBC | commodity | 2006-02-06 | 2007-02-06 | 4256 | 0 | 0 | 0.007055 | 0.1982 | -0.7635 | 39.76 |
| BIL | cash | 2007-05-30 | 2008-05-28 | 3927 | 0 | 0 | 0.006522 | 0.005182 | -0.007799 | 40.81 |

Cash (3M T-bill, FRED DTB3, 1-session lag): CAGR 0.0121 over the research window.

## Validator findings (research window)

- `warning` **zero_volume** EMB: 1 sessions with zero volume (e.g. 2008-02-20)
- `warning` **stale** BIL: 18 unchanged days

## Holdout window — structural only

| ticker | n_rows | missing | non_positive |
|---|---|---|---|
| SPY | 939 | 0 | 0 |
| IWM | 939 | 0 | 0 |
| EFA | 939 | 0 | 0 |
| EWJ | 939 | 0 | 0 |
| VGK | 939 | 0 | 0 |
| EEM | 939 | 0 | 0 |
| SHY | 939 | 0 | 0 |
| IEF | 939 | 0 | 0 |
| TLT | 939 | 0 | 0 |
| AGG | 939 | 0 | 0 |
| TIP | 939 | 0 | 0 |
| LQD | 939 | 0 | 0 |
| HYG | 939 | 0 | 0 |
| EMB | 939 | 0 | 0 |
| VNQ | 939 | 0 | 0 |
| GLD | 939 | 0 | 0 |
| DBC | 939 | 0 | 0 |
| BIL | 939 | 0 | 0 |
