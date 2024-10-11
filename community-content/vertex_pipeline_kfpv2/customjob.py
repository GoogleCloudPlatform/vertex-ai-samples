from kfp.v2 import dsl

@dsl.component(base_image='python:3.8',packages_to_install=['google-cloud-aiplatform==1.36.0'])
def customjob(
    project_id: str,
    location: str,
    staging_bucket: str,
    experiment: str,
    job_name: str,
    script_path: str,
    container_uri: str,
    machine_type: str,
):
    import os
    from google.cloud import aiplatform

    aiplatform.init(
        project=project_id,
        location=location,
        staging_bucket=staging_bucket,
        experiment=experiment,
    )
    job = aiplatform.CustomJob.from_local_script(
        display_name=job_name,
        script_path=os.path.join(os.getcwd(), script_path),
        container_uri=container_uri,
        machine_type=machine_type,
    )
    job.run()

@dsl.pipeline(name='run-customjob')
def pipeline_customjob():
    customjob("990000000009", "us-west1", "gs://staging-bucket/customjob",
        "run-experiment", "custom-job", "customjob.py",
        "gcr.io/path/to/model_name:latest", "n1-standard-4")

if __name__ == "__main__":
    from kfp.v2 import compiler
    compiler.Compiler().compile(
        pipeline_func=pipeline_customjob,
        package_path='customjob.json')