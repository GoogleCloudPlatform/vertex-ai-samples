from kfp.v2 import dsl

@dsl.component(base_image='python:3.8',packages_to_install=['google-cloud-aiplatform==1.36.0'])
def create_endpoint(
    endpoint_name: str,
    project_id: str,
    location: str,
):
    import json
    from google.cloud import aiplatform

    aiplatform.init(
        project=project_id,
        location=location,
    )
    endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)


@dsl.pipeline(name='create-endpoint')
def pipeline_create_endpoint():
    create_endpoint("auto_endpoint", "990000000009", "us-west1")


if __name__ == "__main__":
    from kfp.v2 import compiler
    compiler.Compiler().compile(
        pipeline_func=pipeline_create_endpoint,
        package_path='create_endpoint.json')