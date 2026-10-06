# The writing standard: ASD-STE100 Simplified Technical English for ev-diffusion

Use these rules for every README and for `docs/ste-style-guide.md` in each repository. Copy this file
into the repository as `docs/ste-style-guide.md` and add a **project vocabulary** section (Section 3)
with the technical names and technical verbs of that project.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **registration** | One newly registered electric vehicle, counted in a registration file. | sale, purchase |
| **registration file** | The input table with `year`, `count` and optional `state` and `make_model` columns. | dataset (alone), sheet |
| **adoption** | The new registrations in one period. | uptake, sales, penetration |
| **series** | The yearly adoptions that the models use, from one consistent source. | curve (for data), trend |
| **source** | How ev-diffusion made the series: `national` or `balanced`. | origin (for data), method |
| **national rows** | Rows with the state code `US`, `USA`, `NATIONAL`, `UNITED STATES` or `ALL STATES`. | total rows, country rows |
| **coverage** | The years in which each state has rows in the registration file. | completeness, availability |
| **balanced panel** | The states that have rows in every year of the window. | complete states, panel (alone) |
| **window** | The first year to the last complete year of the series. | range (for years), span |
| **partial year** | A final year that the source did not finish. | incomplete year, half year |
| **period** | One year of the series. Period 1 is the first year. | step, time point |
| **Bass model** | The diffusion model with the parameters `p`, `q` and `m`. | Bass curve (for the model), BM |
| **GBM** | The Generalized Bass Model of Bass, Krishnan and Jain (1994), with one `beta` for each covariate. | extended Bass, Bass x g |
| **covariate** | A positive yearly value that can change the diffusion speed, for example charging ports. | driver, feature (for the GBM) |
| **log change** | `ln(c(t) / c(t-1))` of a covariate. | growth rate, delta |
| **p** | The innovation coefficient of the Bass model. | external influence |
| **q** | The imitation coefficient of the Bass model. | internal influence, word of mouth |
| **m** | The market size: the number of adoptions at the end of the diffusion. | potential, saturation level |
| **peak period** | The period with the highest adoption rate of a fitted Bass model. | inflection year, top |
| **mode** | The fit target: `period` (adoptions) or `cumulative` (running total). | form, variant |
| **fit** | The result of one least-squares estimate of a model. | estimation (as a noun), training |
| **residual** | Observed value minus fitted value. | error (for a fit), noise |
| **HAC standard error** | The Newey-West standard error that allows for autocorrelated residuals. | robust error, corrected error |
| **backtest** | The rolling-origin test of forecasts on years that the fit did not see. | validation (alone), holdout test |
| **origin** | The last year that a backtest fit can see. | cut-off, split point |
| **horizon** | The number of years after the origin that a forecast covers. | lead time, steps ahead |
| **baseline** | A simple forecaster: `naive`, `drift`, `exponential` or `logistic`. | benchmark model, dummy |
| **bootstrap** | The moving-block residual bootstrap of the Bass fit. | Monte-Carlo, simulation |
| **block** | A run of neighbouring residuals that the bootstrap copies as one unit. | chunk, segment |
| **prediction interval** | The 95 % range of future adoptions, with the residual noise added. | uncertainty band, range |
| **review** | One owner text with a star rating. | comment, post, opinion |
| **rating** | The star value of a review, 1 to 5. The value 0 means "not rated". | score, stars (in prose) |
| **label** | The sentiment class of a review: `negative`, `neutral` or `positive`. | class (in prose), tag |
| **unrated review** | A review with the rating 0, an empty rating or a rating outside 1 to 5. | zero review, negative review |
| **duplicate** | A review with the same text as an earlier review after normalization. | copy, repeat |
| **negation scope** | The words after a negation word, up to the next punctuation mark or "but". | negation window |
| **macro-F1** | The mean of the F1 values of the three labels. | F1 (alone), accuracy |
| **fold** | One test part of one cross-validation repeat. | split, partition |
| **corrected interval** | The Nadeau-Bengio corrected 95 % interval of the fold scores. | confidence band |
| **aspect** | A topic of a review: charging, range, price, service, performance, comfort or design. | theme, feature (for a topic) |
| **mention** | One sentence that holds a keyword of an aspect. | hit, occurrence |
| **concern** | An aspect with enough mentions and a clear negative share. Status `supported` or `not supported`. | barrier, pain point |
| **market** | The country or region of the data, for example `US` or `IN`. | geography, locale |
| **synthetic data** | The generated files of `ev-diffusion synth`. They are not real data. | fake data, mock data, sample data |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **load** | Read an input file and check it against its column contract. |
| **audit** | Make the coverage table of the registration file. |
| **build** | Make the series from the national rows or from the balanced panel. |
| **fit** | Estimate the parameters of a model by least squares. |
| **forecast** | Calculate the adoptions of future periods from a fit. |
| **backtest** | Fit on the years before an origin and compare the forecast with the actual years. |
| **resample** | Draw residual blocks with replacement for the bootstrap. |
| **label** | Give a review a label from its rating. |
| **mark** | Add the prefix `NOT_` to a word in a negation scope. |
| **cross-validate** | Fit and score a sentiment model on repeated stratified folds. |
| **tag** | Find the aspects that a sentence mentions. |
