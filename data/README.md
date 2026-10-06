# data/

Git ignores every file in this folder except this README. Do not commit downloaded data.
The offline demo and the tests do not need this folder: `ev-diffusion synth --out data` writes
synthetic files with the same columns.

## Expected files

| File | Required columns | Optional columns | Used by |
|---|---|---|---|
| `registrations.csv` (or `.xlsx` with the `excel` extra) | `year`, `count` | `state`, `make_model` | `coverage`, `fit`, `backtest`, `forecast`, `report` |
| `covariates.csv` | `year` + one column for each positive covariate | — | `fit --covariates`, the GBM |
| `reviews.csv` | `text`, `rating` | `model_name`, `market` | `sentiment`, `aspects`, `report` |

Accepted header aliases (case and punctuation do not matter):

- `year`: `registration_year`, `yr`
- `count`: `vehicle_count`, `registrations`, `vehicles`
- `state`: `region`, `geography`, `state_code`. The codes `US`, `USA`, `NATIONAL`, `UNITED STATES` and `ALL STATES` mark national rows.
- `make_model`: `vehicle_make_&_model`, `vehicle_make_model`, `model`
- `text`: `review`, `review_text`, `body`, `comment`
- `rating`: `stars`, `overall_rating`, `score`. A rating of `0` or an empty cell means "not rated".

## Sources

| Data | Source | URL | License / terms |
|---|---|---|---|
| EV registrations by state and year | Atlas EV Hub, State EV Registration Data | https://www.atlasevhub.com/materials/state-ev-registration-data/ | Read the Atlas EV Hub terms of use before you redistribute the data. Keep the raw download outside git. |
| Public charging ports by year (a covariate) | U.S. Department of Energy, Alternative Fuels Data Center (AFDC) | https://afdc.energy.gov/data | U.S. government data. Cite the AFDC. |
| EV purchase incentives (a covariate) | Make an index from the AFDC "Laws and Incentives" pages | https://afdc.energy.gov/laws | U.S. government data. Record how you built the index. |
| Four-wheeler EV owner reviews | Kaggle dataset with the file `4-wheeler-EV-carwale.csv` (reviews from an Indian car portal) | https://www.kaggle.com/datasets (search "4-wheeler EV carwale") | Kaggle dataset license. The reviews are from the Indian market. Use `--review-market IN`. |

## How to prepare the real data

1. Download the state registration files from Atlas EV Hub. Keep the `state` column.
2. Stack the files into one `registrations.csv` with `year`, `state`, `make_model`, `count`.
3. If the newest year is not complete, give `--last-complete-year` (or `EV_DIFFUSION_LAST_COMPLETE_YEAR`).
4. For the GBM, write `covariates.csv` with one row for each year. Include the year before the first registration year.
   Include the forecast years if you want a GBM forecast.
5. Download the review CSV from Kaggle and save it as `reviews.csv`.
6. Run `ev-diffusion coverage --registrations data/registrations.csv` and read the coverage table first.
