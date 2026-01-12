# WeatherNext 2
[**Overview**](#overview) | [**Use Cases**](#use-cases) | [**Documentation**](#documentation) | [**Pricing**](#pricing) | [**Quick start**](#quick-start)

## Overview
**Disclaimer:**

*Experimental*\
This product is subject to the "Pre-GA Offerings Terms" in the General Service Terms section of the [Service Specific Terms](https://cloud.google.com/terms/service-terms#1). Pre-GA products are available "as is" and might have limited support. For more information, see the [launch stage descriptions](https://cloud.google.com/products#product-launch-stages). <!-- disableFinding(LINE_OVER_80) -->

Access to the forecasting capabilities requires application and approval. Users must be added to an allowlist to generate forecasts using this service. Review pricing details at [Vertex AI Custom Training pricing,](https://cloud.google.com/vertex-ai/pricing?hl=en&e=48754805#custom-trained-models) [Cloud Storage pricing](https://cloud.google.com/storage/pricing), and use the [Pricing Calculator](https://cloud.google.com/products/calculator/) before running. <!-- disableFinding(LINE_OVER_80) -->

**Overview**

WeatherNext is a family of state-of-the-art AI weather forecasting models from Google DeepMind and Google Research. WeatherNext models are faster and more efficient than traditional physics-based weather models and yield superior forecast reliability. <!-- disableFinding(LINE_OVER_80) -->

WeatherNext 2
Generates ensemble forecasts at a spatial resolution of 0.25 degrees. Forecast init times have 6 hour resolution (00z, 06z, 12z, 18z). Forecast lead times have 1 hour resolution and a lead time of 15 days. Additional information on the model is described in "[Skillful joint probabilistic weather forecasting from marginals](https://arxiv.org/abs/2506.10772)”. <!-- disableFinding(LINE_OVER_80) -->

## Use Cases
**Use Cases**

*What use cases are best for WeatherNext 2?*\
WeatherNext 2 offers a fast, high-resolution global medium-range forecasting
model, serving as a powerful alternative to traditional systems like ECMWF ENS
or NOAA GEFS. Key applications include medium-range forecasts (15 days),
predicting severe events like tropical cyclones, feeding data into downstream
systems such as flood models, and generating large ensembles for comprehensive
uncertainty quantification

## Documentation
**Training data**

ECMWF [ERA5](https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5) (1979-2018)

ECMWF [HRES](https://www.ecmwf.int/en/forecasts/datasets/set-i)

NOAA [IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive)

**Model Evaluation**

Model training and evaluation methodology, and a complete set of model
scorecards evaluating each across variables and lead times, are reported in
the research paper associated with the WeatherNext 2 model:

* [Skillful joint probabilistic weather forecasting from marginals](https://arxiv.org/abs/2506.10772)

**Documentation**

See [the WeatherNext developer’s guide](http://developers.google.com/weathernext) for documentation regarding the models.

**Acknowledgements**

The models communicate with and reference data and products of the European
Centre for Medium-range Weather Forecasts (ECMWF), as modified by Google.

**Contact**

If you have any questions on using these models please contact: [weathernext-cloud@google.com](mailto:weathernext-cloud@google.com). Any information collected via email will be used in accordance with [Google's privacy policy](https://policies.google.com/privacy).

**Links**

* [WeatherNext developer’s guide](http://developers.google.com/weathernext)
* [Research model paper](https://arxiv.org/abs/2506.10772)

## Pricing
Access to forecasting with the WeatherNext models on Vertex AI is currently
restricted.
To utilize these models for generating real-time forecasts via this service:

* You must **apply for access**.
* Your application will be reviewed, and if approved, you will be **added to an allowlist**.
* Only allow listed users can access the forecasting service.
* **Pricing information** for the forecasting service will be shared directly with users upon approval and placement on the allowlist.

Running these models *will incur costs* for the GPUs and other Google Cloud resources used. Learn about [Vertex AI Custom Training pricing](https://cloud.google.com/vertex-ai/pricing?hl=en&e=48754805#custom-trained-models), [Cloud Storage pricing](https://cloud.google.com/storage/pricing), and use the [Pricing Calculator](https://cloud.google.com/products/calculator/) to generate an estimate.

## Quick start
The quickest way to get started with the WeatherNext in Google Cloud Platform is to run [our example notebook](weathernext_2_early_access_program.ipynb) in [Google Colab](https://colab.research.google.com/).
