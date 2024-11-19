from kfp.v2 import dsl

@dsl.component(base_image='python:3.8',packages_to_install=['google-cloud-aiplatform==1.36.0'])
def pipelineJob(
    project_id: str,
    location: str,
    display_name: str,
    json_file: str,
    pipeline_root: str,
):
    import os
    from google.cloud import aiplatform

    aiplatform.init(
        project=project_id,
        location=location,
    )

    job = aiplatform.PipelineJob(
        display_name=display_name,
        template_path=json_file,
        pipeline_root=pipeline_root,
        enable_caching=False,
    ).run()

    job.delete()


@dsl.pipeline(name='pipelineJobs')
def pipeline_run_jobs():
    # 1. create endpoint
    pipelineJob("990000000009", "us-west1", "Pipeline-create endpoint",
        "create_endpoint.json", "gs://pipeline-root-bucket/pipelines")
        
    # 2. deploy model to endpoint
    pipelineJob("990000000009", "us-west1", "Pipeline-deploy model",
        "deploy_model.json", "gs://pipeline-root-bucket/pipelines")

if __name__ == "__main__":
    from kfp.v2 import compiler
    compiler.Compiler().compile(
        pipeline_func=pipeline_run_jobs,
        package_path='pipelineJobs.json')