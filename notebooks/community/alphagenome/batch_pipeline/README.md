# AlphaGenome Batch Inference Pipeline

This directory provides a pre-compiled **Vertex AI Pipeline** (`alphagenome_batch_inference_pipeline.yaml`) for running AlphaGenome predictions in batch on JSONL input files stored in Google Cloud Storage (GCS).

You can run the pipeline directly from the **Google Cloud Console**, run the end-to-end [batch inference notebook](../cloudai_alphagenome_batch_pipeline.ipynb), or submit it with the **Vertex AI Python SDK**—no custom container build is required.

---

## How It Works

The pipeline runs in three steps:

1. **Endpoint setup**:
   - **Existing Shared Endpoint (`existing_endpoint` set)**: Connects to your deployed Vertex AI endpoint over gRPC.
   - **Existing Dedicated Endpoint (`existing_vertex_ai_url` set)**: Connects to your dedicated endpoint URL over HTTPS.
   - **Temporary Endpoint (both left blank)**: Uploads the AlphaGenome serving container, creates a temporary Vertex AI endpoint for the run, and deploys the requested GPU replicas.
2. **Batch prediction**:
   - Reads prediction requests from a `.jsonl` file (or a folder of `.jsonl` files) in GCS and sends them concurrently (`workers`, default `8`).
   - Automatically retries temporary network or rate-limit errors with exponential backoff (`max_retries`, default `4`).
   - Logs any invalid or permanently failing rows to `errors.jsonl` so the remaining valid requests still complete (`fail_on_any_error=False` by default).
   - Periodically saves progress to GCS so that if a worker is interrupted and restarted, already-completed requests are skipped.
3. **Automatic cleanup**:
   - Runs automatically when the pipeline finishes, fails, or is cancelled.
   - Leaves pre-existing customer endpoints untouched; if the pipeline created a temporary endpoint in Step 1, undeploys the model and deletes the temporary endpoint and model resources.

---

## Prerequisites

- **Google Cloud Storage bucket** in the same region as your pipeline run (or in the `US` multi-region) for input files, output files, and pipeline staging.
- **Service account permissions**: The service account running the pipeline needs:
  - `Vertex AI User` (`roles/aiplatform.user`) on the project
  - `Storage Object Admin` (`roles/storage.objectAdmin`) on the GCS bucket

---

## Option 1: Run from the Google Cloud Console

1. Upload [`sample_batch_requests.jsonl`](sample_batch_requests.jsonl) (or your own `.jsonl` file) to your GCS bucket, for example: `gs://YOUR_BUCKET/alphagenome/input/sample_batch_requests.jsonl`.
2. Open **Vertex AI -> Pipelines -> Create run** in the Google Cloud Console.
3. Choose **Upload a file** and select [`alphagenome_batch_inference_pipeline.yaml`](alphagenome_batch_inference_pipeline.yaml).
4. Set **Pipeline root** (for example, `gs://YOUR_BUCKET/pipeline_root/`), select your **Region** and **Service account**, and click **Continue**.
5. Enter the pipeline parameters:
   - **`input_gcs_uri`**: `gs://YOUR_BUCKET/alphagenome/input/sample_batch_requests.jsonl` (or a `gs://.../input/` folder prefix)
   - **`output_gcs_uri`**: `gs://YOUR_BUCKET/alphagenome/output/run_01/`
   - **`region`**: Region of your endpoint or GPU quota (for example, `us-central1`, `us-west1`, or `us-east5`)
   - **Endpoint configuration (choose one)**:
     - **Shared Endpoint**: Set `existing_endpoint` to `projects/<PROJECT_NUMBER>/locations/<REGION>/endpoints/<ENDPOINT_ID>`
     - **Dedicated Endpoint**: Set `existing_vertex_ai_url` to `https://<ENDPOINT_DNS>/v1/projects/<PROJECT_NUMBER>/locations/<REGION>/endpoints/<ENDPOINT_ID>:streamRawPredict`
     - **Temporary GPU Endpoint**: Leave both `existing_endpoint` and `existing_vertex_ai_url` blank, and optionally adjust `serving_accelerator_type` (`NVIDIA_H100_80GB`, `NVIDIA_H100_MEGA_80GB`, or `NVIDIA_A100_80GB`) and `serving_replica_count` (default `8`).
6. Click **Submit**.

---

## Option 2: Run with the Vertex AI Python SDK

```python
from google.cloud import aiplatform

PIPELINE_YAML_URL = (
    "https://github.com/GoogleCloudPlatform/vertex-ai-samples/raw/main/"
    "notebooks/community/alphagenome/batch_pipeline/"
    "alphagenome_batch_inference_pipeline.yaml"
)

aiplatform.init(
    project="YOUR_PROJECT_ID",
    location="us-west1",
    staging_bucket="gs://YOUR_BUCKET",
)

job = aiplatform.PipelineJob(
    display_name="alphagenome-batch-run",
    template_path=PIPELINE_YAML_URL,  # Or local file: "alphagenome_batch_inference_pipeline.yaml"
    pipeline_root="gs://YOUR_BUCKET/pipeline_root",
    parameter_values={
        "input_gcs_uri": "gs://YOUR_BUCKET/alphagenome/input/sample_batch_requests.jsonl",
        "output_gcs_uri": "gs://YOUR_BUCKET/alphagenome/output/run_01/",
        "region": "us-west1",
        "existing_endpoint": "projects/YOUR_PROJECT_NUMBER/locations/us-west1/endpoints/YOUR_ENDPOINT_ID",
        "model_version": "ALL_FOLDS",
        "workers": 8,
    },
)
job.submit(service_account="YOUR_SERVICE_ACCOUNT@YOUR_PROJECT_ID.iam.gserviceaccount.com")
```

---

## Input Format

Each line in the input `.jsonl` file is a JSON object specifying a `request_id`, a `request_type`, and the parameters for that request. Supported `request_type` values are:

- `predict_sequence`
- `predict_interval`
- `predict_variant`
- `score_variant`
- `score_interval`
- `score_ism_variants`
- `output_metadata`

See [`sample_batch_requests.jsonl`](sample_batch_requests.jsonl) for complete examples of all seven request types.

---

## Output Files

All outputs are written to `output_gcs_uri`:

```text
gs://YOUR_BUCKET/alphagenome/output/run_01/
├── predictions.jsonl                  # Per-request status, latency, summary metrics, and output file URIs
├── summary.json                       # Overall success/failure counts, throughput, and p50/p95/p99 latency
├── errors.jsonl                       # Created only if one or more requests fail (includes original input and error message)
├── checkpoints/
│   └── part-*.jsonl                   # Progress files used automatically to resume interrupted runs
├── tracks/
│   ├── <request_id>.npz               # Compressed NumPy track arrays
│   └── <request_id>_metadata.parquet  # Track metadata table
├── scores/
│   ├── <request_id>.h5ad              # AnnData score matrix
│   └── <request_id>_tidy.parquet      # Flattened score table (written when save_tidy_scores=True)
└── metadata/
    └── <request_id>_<modality>.parquet
```
