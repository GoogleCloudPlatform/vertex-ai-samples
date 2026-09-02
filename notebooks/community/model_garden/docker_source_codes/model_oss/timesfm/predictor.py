"""Adapts a pretrained TimesFM to the CPR framework.

Documentation for the model is here:
https://github.com/google-research/timesfm

Model checkpoints can be found here:
https://www.huggingface.co/google/timesfm-1.0-200m
"""

import datetime
import os
from typing import Any, Dict, List, Sequence, Tuple

import fastapi
from google.cloud.aiplatform.utils import prediction_utils
from jax._src import config
import timesfm

HTTPException = fastapi.HTTPException
_BACKEND = os.getenv("TIMESFM_BACKEND", default="cpu")

config.update(
    "jax_platforms", {"cpu": "cpu", "gpu": "cuda", "tpu": ""}[_BACKEND]
)

TsArray = None | float | int | str | List["TsArray"]

_BAD_REQUEST_STATUS = 400
_EXPECTED_FORMAT = """

[NOTICE] TimesFM inference server expects input format:
{
    "instances": [
        {
            "input": [0.0, 0.1, 0.2, ...],
            "freq": 0,                                       # optional, 0/1/2
            "horizon": 12,                                   # optional
            "timestamp": ["2024-01-01", "2024-01-02", ...],  # optional
            "timestamp_format": "%Y-%m-%d",                  # optional
            "dynamic_numerical_covariates": {
                "dncov1": [1.0, 2.0, 1.5, ...],
                "dncov2": [3.0, 1.1, 2.4, ...],
            },                                               # optional
            "dynamic_categorical_covariates": {
                "dccov1": ["a", "b", "a", ...],
                "dccov2": [0, 1, 0, ...],
            },                                               # optional
            "static_numerical_covariates": {
                "sncov1": 1.0,
                "sncov2": 2.0,
            },                                               # optional
            "static_categorical_covariates": {
                "sccov1": "a",
                "sccov2": "b",
            },                                               # optional
            "xreg_kwargs": {...},                            # optional
        },
        {"input": [113.2, 15.0, 65.4, ...], ...},
        {"input": [  0.0, 10.0, 20.0, ...], ...},
        ...
    ]
}
"""


def _raise_bad_request(message: str):
  message = message + "\n" + _EXPECTED_FORMAT
  raise HTTPException(
      status_code=_BAD_REQUEST_STATUS,
      detail=message,
  )


def _datetime_to_freq(dt1: datetime.datetime, dt2: datetime.datetime) -> int:
  delta = dt2 - dt1
  if delta.days <= 1:
    return 0
  elif delta.days <= 31:
    return 1
  else:
    return 2


def _add_cov_to_dict(
    index: int,
    cov_input: dict[str, TsArray],
    cov_dict: dict[str, list[TsArray]],
):
  """Adds covariates to the dictionary of covariates.

  Args:
    index: Index of the instance.
    cov_input: Dictionary of covariates for the current instance.
    cov_dict: Dictionary of covariates for all instances.
  """
  if index == 0:
    cov_dict.update({k: [v] for k, v in cov_input.items()})
  else:
    if set(cov_input.keys()) != set(cov_dict.keys()):
      _raise_bad_request(
          f"Instance {index}:"
          " All instances must have the same set of covariates if any."
      )
    for k, v in cov_input.items():
      cov_dict[k].append(v)


def _linear_interpolate_missing_timepoints(
    timestamp: list[datetime.datetime],
    value: list[float],
) -> tuple[list[datetime.datetime], list[TsArray]]:
  """Linearly interpolates missing timepoints in a timeseries."""

  def _gcd_timelapse(t1, t2):
    if (w := t2 % t1) == datetime.timedelta(0):
      return t1
    if t1 > t2:
      return _gcd_timelapse(t2, t1)
    return _gcd_timelapse(w, t1)

  if len(timestamp) < 3:
    return timestamp, value, False

  no_missing = True
  delta = timestamp[1] - timestamp[0]
  if delta <= datetime.timedelta(0):
    _raise_bad_request(
        f"Timestamps must be in ascending order. Got {timestamp}"
    )
  for i in range(2, len(timestamp)):
    delta_next = timestamp[i] - timestamp[i - 1]
    if delta_next <= datetime.timedelta(0):
      _raise_bad_request(
          f"Timestamps must be in ascending order. Got {timestamp}"
      )
    delta_new = _gcd_timelapse(delta, delta_next)
    if delta_new != delta:
      no_missing = False
      delta = delta_new

  if no_missing:
    return timestamp, value, False

  new_timestamp = []
  new_value = []
  for i in range(len(timestamp) - 1):
    new_timestamp.append(timestamp[i])
    new_value.append(value[i])
    if (num_deltas := int((timestamp[i + 1] - timestamp[i]) / delta + 0.5)) > 1:
      value_delta = (value[i + 1] - value[i]) / num_deltas
      for j in range(1, num_deltas):
        new_timestamp.append(timestamp[i] + j * delta)
        new_value.append(value[i] + j * value_delta)
  new_timestamp.append(timestamp[-1])
  new_value.append(value[-1])

  return new_timestamp, new_value, True


class TimesFMPredictor:
  """Predictor class for time-series foundation model TimesFM."""

  TIMESFM_MODEL_NAME = os.getenv(
      "TIMESFM_MODEL_NAME", default="timesfm-1.0-200m"
  )
  CONTEXT_LEN = 512
  INPUT_PATCH_LEN = 32
  OUTPUT_PATCH_LEN = 128
  NUM_LAYERS = 20
  MODEL_DIMS = 1280
  BACKEND = os.getenv("TIMESFM_BACKEND", default="cpu")
  MAX_HORIZON = int(os.getenv("TIMESFM_HORIZON", default="128"))

  def load(self, artifacts_uri: str = ""):
    """Initializes the model and preprocessing transforms.

    Args:
      artifacts_uri: Directory where state dict is stored. Can be a GCS URI or
        local path.
    """
    if not (os.path.isdir(artifacts_uri) or artifacts_uri.startswith("gs://")):
      raise ValueError(
          f"Provided artifact_uri is not a directory: {artifacts_uri}"
      )

    print(f"Downloading checkpoints from {artifacts_uri}")
    prediction_utils.download_model_artifacts(artifacts_uri)
    artifact_path = os.getcwd()

    print(f"Loading checkpoints from {artifact_path}")
    self._model = timesfm.TimesFm(
        context_len=self.CONTEXT_LEN,
        horizon_len=(
            ((self.MAX_HORIZON - 1) // self.OUTPUT_PATCH_LEN + 1)
            * self.OUTPUT_PATCH_LEN
        ),
        input_patch_len=self.INPUT_PATCH_LEN,
        output_patch_len=self.OUTPUT_PATCH_LEN,
        num_layers=self.NUM_LAYERS,
        model_dims=self.MODEL_DIMS,
        backend=self.BACKEND,
    )
    self._model.load_from_checkpoint(artifact_path)
    print(f"Loaded TimesFM model from {artifact_path}")

  def preprocess(
      self, request_dict: Dict[str, Sequence[Dict[str, TsArray]]]
  ) -> Dict[str, TsArray]:
    """Performs preprocessing.

    By default, the server expects a request body consisting of a valid JSON
    object. This will be parsed by the handler before it's evaluated by the
    preprocess method.

    Args:
      request_dict: Parsed request body. We expect that the input consists of a
        list of time-series forecast contexts. Each context should be in a
        format convertible to JTensor by `jnp.array`.

    Returns:
      Time-series forecast contexts are passed as is from the input as a list.
    """

    if "instances" not in request_dict:
      _raise_bad_request('Request must contain "instances" as a top-level key.')

    input_instances = request_dict["instances"]
    if not input_instances or not isinstance(input_instances, List):
      _raise_bad_request(
          f"Received `instances` not a list. Got {type(input_instances)}"
      )

    inputs, freqs, timestamps, timestamp_formats = [], [], [], []
    horizon_lens = []
    static_numerical_covariates, static_categorical_covariates = {}, {}
    dynamic_numerical_covariates, dynamic_categorical_covariates = {}, {}
    xreg_kwargs = {}

    exists_missing = False
    for index, each_input in enumerate(input_instances):

      # 1. Add input time-series context.
      if (
          (not isinstance(each_input, Dict))
          or ("input" not in each_input)
          or (len(each_input["input"]) < 2)
      ):
        _raise_bad_request(
            f"Instance {index}:"
            " Invalid datatype. Each input example must have `input` key"
            " mapped to a list of time-series forecast context with length > 1."
        )
      new_input = each_input["input"]

      # 2. Process timestamps.
      if "timestamp" not in each_input:
        timestamps.append(None)
      else:
        if len(each_input["timestamp"]) != len(each_input["input"]):
          _raise_bad_request(
              f"Instance {index}:"
              " Invalid datatype. `timestamp` if given must have same length as"
              "`input`."
          )
        new_timestamp = [
            datetime.datetime.fromisoformat(s) for s in each_input["timestamp"]
        ]
        # Linearly interpolate missing timepoints and values.
        new_timestamp, new_input, new_exists_missing = (
            _linear_interpolate_missing_timepoints(new_timestamp, new_input)
        )
        exists_missing = exists_missing or new_exists_missing
        timestamps.append(new_timestamp)
      if "timestamp_format" in each_input:
        timestamp_formats.append(each_input["timestamp_format"])
      else:
        timestamp_formats.append(None)

      inputs.append(new_input)

      # 3. Process frequency.
      if "freq" in each_input:
        freqs.append(each_input["freq"])
      elif timestamps[index]:
        freqs.append(
            _datetime_to_freq(timestamps[index][0], timestamps[index][1])
        )
      else:
        freqs.append(0)

      # 4. Process covariate data.
      for cov_category, cov_dict in [
          ("static_numerical_covariates", static_numerical_covariates),
          ("static_categorical_covariates", static_categorical_covariates),
          ("dynamic_numerical_covariates", dynamic_numerical_covariates),
          ("dynamic_categorical_covariates", dynamic_categorical_covariates),
      ]:
        if cov_category in each_input:
          _add_cov_to_dict(index, each_input[cov_category], cov_dict)

      # 5. Process xreg config. Power user option. If nothing set we apply
      # TimesFM default.
      if "xreg_kwargs" in each_input:
        if not xreg_kwargs:
          xreg_kwargs = each_input["xreg_kwargs"]
        elif xreg_kwargs != each_input["xreg_kwargs"]:
          _raise_bad_request(
              f"Instance {index}:"
              " All instances must have the same xreg_kwargs if any."
          )

      # 6. Process horizon length.
      if "horizon" in each_input:
        if (w := each_input["horizon"]) > self.MAX_HORIZON:
          _raise_bad_request(
              f"Instance {index}: `horizon` must be <= maximum horizon"
              f" {self.MAX_HORIZON}. Got {w}. To increase the maximum horizon,"
              " recreate the endpoint with a higher `TIMESFM_HORIZON` env"
              " value."
          )
        horizon_lens.append(w)
      else:
        horizon_lens.append(self.MAX_HORIZON)

    return {
        "inputs": inputs,
        "freqs": freqs,
        "timestamps": timestamps,
        "timestamp_formats": timestamp_formats,
        "exists_missing": exists_missing,
        "static_numerical_covariates": static_numerical_covariates,
        "static_categorical_covariates": static_categorical_covariates,
        "dynamic_numerical_covariates": dynamic_numerical_covariates,
        "dynamic_categorical_covariates": dynamic_categorical_covariates,
        "xreg_kwargs": xreg_kwargs,
        "horizon_lens": horizon_lens,
    }

  def predict(self, instances: Dict[str, Any]) -> Any:
    """Performs prediction.

    Args:
      instances: A dictionary with two keys - `inputs` and `freq` where `inputs`
        is list of time series forecast contexts. Each context time series
        should be in a format convertible to JTensor by `jnp.array`. `freq` is
        frequencies of each forecast context with values as 0 (high), 1 (medium)
        and 2 (low). If not provided, all contexts are assumed to be high
        frequency.

    Returns:
        A tuple of List:
        - the mean forecast of size (# inputs, # forecast horizon),
        - the full forecast (mean + quantiles) of size
            (# inputs,  # forecast horizon, 1 + # quantiles).
    """
    (
        inputs,
        freqs,
        timestamps,
        timestamp_formats,
        exists_missing,
        static_numerical_covariates,
        static_categorical_covariates,
        dynamic_numerical_covariates,
        dynamic_categorical_covariates,
        xreg_kwargs,
        horizon_lens,
    ) = (
        instances["inputs"],
        instances["freqs"],
        instances["timestamps"],
        instances["timestamp_formats"],
        instances["exists_missing"],
        instances["static_numerical_covariates"],
        instances["static_categorical_covariates"],
        instances["dynamic_numerical_covariates"],
        instances["dynamic_categorical_covariates"],
        instances["xreg_kwargs"],
        instances["horizon_lens"],
    )

    if (
        static_numerical_covariates
        or static_categorical_covariates
        or dynamic_numerical_covariates
        or dynamic_categorical_covariates
    ):
      if (
          dynamic_categorical_covariates or dynamic_numerical_covariates
      ) and exists_missing:
        _raise_bad_request(
            "Dynamic covariates are not supported when input has missing"
            " timestamps."
        )
      print("Detected covariates. Callng model.forecast_with_covariates.")
      try:
        point_forecast, _ = self._model.forecast_with_covariates(
            inputs=inputs,
            dynamic_numerical_covariates=dynamic_numerical_covariates,
            dynamic_categorical_covariates=dynamic_categorical_covariates,
            static_numerical_covariates=static_numerical_covariates,
            static_categorical_covariates=static_categorical_covariates,
            freq=freqs,
            **xreg_kwargs,
        )
        # point_forecast is a list of np.ndarrays.
        point_forecast = [p.tolist() for p in point_forecast]
        quantile_forecast = None
      except ValueError as e:
        _raise_bad_request(f"model.forecast_with_covariates failed from {e}.")
        return
    else:
      print("Calling model.forecast.")
      point_forecast, quantile_forecast = self._model.forecast(
          inputs=inputs, freq=freqs
      )
      # point_forecast and quantile_forecast are JTensors (np.ndarrays).
      point_forecast = point_forecast.tolist()
      quantile_forecast = quantile_forecast.tolist()

    return (
        point_forecast,
        quantile_forecast,
        timestamps,
        timestamp_formats,
        horizon_lens,
    )

  def postprocess(
      self, forecasts: Tuple[TsArray, TsArray, TsArray, TsArray]
  ) -> Dict[str, List[Dict[str, TsArray]]]:
    """Translates the model output.

    Args:
       forecasts: A tuple of List - the mean forecast of size (# inputs, #
         forecast horizon), - the full forecast (mean + quantiles) of size (#
         inputs,  # forecast horizon, 1 + # quantiles).

    Returns:
      Dictionary containing the list of point forecasts and quantile forecasts
      for each of the input time-series context.
    """
    (
        point_forecasts,
        quantile_forecasts,
        timestamps,
        timestamp_formats,
        horizon_lens,
    ) = forecasts
    predictions = []

    quantile_names = ["mean"] + [
        f"p{int(quantile * 100)}" for quantile in self._model.model_p.quantiles
    ]
    for i, point_forecast in enumerate(point_forecasts):
      response = {"point_forecast": point_forecast[: horizon_lens[i]]}

      if quantile_forecasts:
        for j, quantile_name in enumerate(quantile_names):
          response[quantile_name] = [x[j] for x in quantile_forecasts[i]][
              : horizon_lens[i]
          ]

      if timestamps[i]:
        last_timestamp = timestamps[i][-1]
        timestamp_delta = timestamps[i][-1] - timestamps[i][-2]
        response["timestamp"] = []
        for _ in range(len(point_forecast)):
          last_timestamp = last_timestamp + timestamp_delta
          response["timestamp"].append(
              datetime.datetime.strftime(last_timestamp, timestamp_formats[i])
              if timestamp_formats[i]
              else last_timestamp.isoformat()
          )
        response["timestamp"] = response["timestamp"][: horizon_lens[i]]

      predictions.append(response)

    return {"predictions": predictions}
