from kfp.v2 import dsl

@dsl.component(base_image='python:3.8',packages_to_install=['google-cloud-aiplatform==1.36.0'])
def batch_predict(
    model_id: str,
    job_name: str,
    file_format: str,
    file_sources: List[str],
    out_sources: str,
    machine_type: str,
    min_replica_count: int,
    max_replica_count: int,
):
    import json
    from google.cloud import aiplatform

    aiplatform.init(
        project=project_id,
        location=location,
    )
    
    model = aiplatform.Model(model_id)
    batch_job = model.batch_predict(
        job_display_name=job_name,
        instances_format=file_format,
        gcs_source=file_sources,
        gcs_destination_prefix=out_sources,
        machine_type=machine_typem,
        starting_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
    )
    batch_job.wait()


@dsl.pipeline(name='batch-predict')
def pipeline_batch_predict():
    batch_predict('projects/990000000009/locations/us-west1/models/1100000000000000001',
        'batch-predict-job', 'csv',
        ['gs://yourbucket/predict/file.csv'], 'gs://yourbucket/results/',
        'n1-standard-2', 1, 1)


if __name__ == "__main__":
    from kfp.v2 import compiler
    compiler.Compiler().compile(
        pipeline_func=pipeline_batch_predict,
        package_path='batch_predict.json')