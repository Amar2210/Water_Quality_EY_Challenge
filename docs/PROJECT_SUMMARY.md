# Project summary — in plain English

## The idea

Can we estimate the chemistry of a river without collecting a fresh water sample at every location?

This project explored that question for the **2026 EY AI & Data Challenge**, using river measurements from South Africa, satellite imagery, and other open environmental datasets. The challenge supplied historical measurements and a set of locations and dates where predictions were required.

The three measurements to predict were:

- **Total Alkalinity:** the water's ability to neutralize acids. Minerals dissolved from soil and rock can influence it.
- **Electrical Conductance:** how readily the water conducts electricity. Dissolved ions increase conductance, although a high value alone does not establish whether water is safe or polluted.
- **Dissolved Reactive Phosphorus:** a readily available form of phosphorus. Excess nutrients can contribute to algal growth.

The project produces **three numerical estimates**, not a single “good water/bad water” verdict.

## What data was available?

The local training file contains **9,319 measurements from 162 stations**, covering **2011–2015**. Each row has coordinates, a sampling date, and the three measured targets. The prediction template contains **200 rows at 24 other locations**.

The EY notebooks provide a starting point for extracting Landsat imagery and TerraClimate climate data. The project expanded that starting point substantially:

| Data | What it tells the model |
|---|---|
| **Landsat imagery** | Reflected light in several wavelengths, water/vegetation indices, and surface temperature where available. |
| **TerraClimate** | Rainfall, temperature, evaporation, soil moisture, drought indicators, runoff, and earlier months' conditions. |
| **Elevation and nearby terrain** | Height, slope, roughness, and local elevation differences. |
| **SoilGrids** | Soil texture, organic carbon, pH, nutrient content, and capacity to retain ions. |
| **Land-cover maps** | How much surrounding land is agricultural, urban, forested, bare, wetland, or water. |
| **Geology extraction** | An attempted description of bedrock, age, and soil-derived proxies; the local rock classifications need fixing. |
| **Other training stations** | A spatial estimate based on measured chemistry at neighboring stations. This is target-derived information, not imagery. |

These features offer possible explanations for water chemistry. For example, rainfall may dilute dissolved substances or transport agricultural nutrients into a river. Slope and soil can affect runoff, while land cover describes potential sources of that runoff. Such explanations are hypotheses; feature importance alone does not prove causation.

## Why was the challenge difficult?

The prediction locations are not the same stations used for training. In the saved feature data, their nearest training stations are approximately **82–282 km away**, with a median distance of **194 km**.

That makes this much harder than filling in a missing measurement at a familiar station. A model may learn patterns in one region that do not carry over to another region with different land use, geology, river systems, or human activity.

The project recognized this and used **spatial cross-validation**: hold out a geographic group, train on the remaining groups, and test on the held-out group. This is more informative for the challenge than randomly mixing measurements from the same stations into training and testing.

## How the approach evolved

### 1. Establish a baseline

Early versions used the supplied Landsat and climate features with XGBoost. They added dates, seasons, and spectral ratios, but performance was poor when entire geographic regions were held out.

### 2. Describe the river's surroundings

The next versions added climate history, soil, elevation, slope, and interactions motivated by water processes. They also compared linear models and stronger regularization rather than assuming a more complicated model would always win.

A recurring observation in the recorded experiments was that **terrain and soil often ranked above the spectral features**. This suggested that understanding the landscape around a river mattered alongside observing its appearance.

### 3. Check the imagery rather than assuming more is better

Sentinel-2 was explored, but the local extracted spectral columns are entirely missing. Historical notes describe a slightly different missing-data rate, so the existing cache should not be treated as a successful Sentinel-2 dataset.

The fuller Landsat extraction was much more useful operationally: six reflected-light bands are present for about **85.7% of training rows**, although surface temperature is available for only **32.0%**. The feature pipeline uses the original EY Landsat values as a fallback for some missing bands.

### 4. Add landscape context

V8 combined land cover, attempted geology features, imagery, climate, soil, and terrain. It produced **271 environmental features**. Electrical Conductance improved in the recorded comparison, but Alkalinity and Phosphorus did not. This did not establish that each added data source worked independently.

### 5. Anchor predictions to other stations

V9 added **88 spatial/neighbor features**, for **359 features in total**. Its central idea was:

> Start with an estimate from other stations, then use environmental information to adjust that estimate.

It tried both a conventional ML model using those neighbor estimates as inputs and a distance-weighted estimate with a smaller ML correction. Predictions from the two approaches were blended, with different choices for the three targets.

V10 simplified this to spatial interpolation alone. V11 tried finding stations that were environmentally similar instead of just geographically close. V12 proposed a smaller raster-focused feature set and using predicted Conductance to help predict Phosphorus; no V12 result artifact is present in this checkout.

## What was achieved — and what remains unproven?

The project delivered a substantial **data-extraction, feature-engineering, modeling, and submission pipeline**, plus an unusually detailed record of experiments and lessons learned.

The author's saved restart baseline is **V9**, in `best_submission/submission_v9.csv`. It contains all 200 required rows, with complete predictions and the correct location/date order.

However, the strongest claims in the older log need qualification:

- V9's neighbor features were calculated using all training targets **before** the regional test split. Excluding a station itself does not exclude other stations in the held-out region. Its positive reported results therefore do not prove clean cross-region generalization.
- V10's stricter regional tests were negative despite positive leave-one-station-out results. The easier test and the harder test measure different tasks.
- V11 has saved results, despite the older log saying they were pending; those regional results remained negative, and the final prediction code needs repairs.
- The local geology categories are all “unknown,” so successful bedrock classification cannot be claimed.
- No independently verified V9 leaderboard score is stored with the saved submission.

A negative R² means the evaluated predictions were worse than predicting the mean of the evaluated observations. It is an important diagnostic, not a reason to hide the work.

## The most valuable lesson

This was not simply a matter of feeding satellite pictures into a model. It was a lesson in **joining different kinds of environmental data, testing whether a model transfers to new places, and checking whether its apparent improvements are real**.

To finish on a strong note, the next modeling step is not another large experiment. It is a clean regional evaluation: build neighbor features and preprocessing only from each fold's training data, fix data-unit and geology issues, and compare a few well-defined baselines on untouched holdouts.

## A concise portfolio description

> Built an end-to-end geospatial machine-learning pipeline to estimate three river-water chemistry measurements in South Africa. Combined Landsat imagery with open climate, soil, terrain, and land-cover data; compared environmental models with spatial interpolation; and investigated the challenges of predicting at unseen river stations. The work emphasized spatial validation, missing-data checks, and reproducible experiment tracking.

For detailed evidence and caveats, see [REVIEW.md](REVIEW.md). For running the pipeline, see [REPRODUCING.md](REPRODUCING.md).
