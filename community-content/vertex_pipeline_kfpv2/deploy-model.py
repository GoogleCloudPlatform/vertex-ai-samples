from kfp.v2 import dsl

@dsl.component(base_image='python:3.8',packages_to_install=['google-cloud-aiplatform==1.36.0'])
def deploy_model(
    model_id: str,
    endpoint_id: str,
    machine_type: str,
    min_replica_count: int,
    max_replica_count: int,
):
    import json
    from google.cloud import aiplatform

    model = aiplatform.Model(model_id)
    endpoint = aiplatform.Endpoint(endpoint_id)

    endpoint = model.deploy(
        endpoint=endpoint,
        machine_type=machine_type,
        min_replica_count=min_replica_count,
        max_replica_count=max_replica_count,
    )


@dsl.pipeline(name='deploy-model')
def pipeline_deploy_model():
    project = "projects/990000000009/locations/us-west1"
    model_id = project + "/models/1100000000000000001"
    endpoint_id = project + "/endpoints/2200000000000000002"
    deploy_model(model_id, endpoint_id, "n1-standard-2", 1, 1)


if __name__ == "__main__":
    from kfp.v2 import compiler
    compiler.Compiler().compile(
        pipeline_func=pipeline_deploy_model,
        package_path='deploy_model.json')