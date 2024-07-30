import numpy as np
from kfp.v2 import dsl

@dsl.component(base_image='python:3.8',packages_to_install=['google-cloud-aiplatform==1.36.0'])
def async_predict(
    endpoint_id: str,
    instances: dict,
) -> np.ndarray:
    import numpy as np
    from google.cloud import aiplatform

    endpoint = aiplatform.Endpoint(endpoint_id)
    response = await endpoint.predict_async(instances)
    predictions = np.asarray(response.predictions)
    print(predictions.tolist())
    return predictions

@dsl.pipeline(name='async-prediction')
def pipeline_prediction():
    project = "projects/990000000009/locations/us-west1"
    endpoint_id = project + "/endpoints/2200000000000000002"
    instances = [{
        "key1": "value1",
        "key2": 2
    }]
    async_predict(endpoint_id, instances)

if __name__ == "__main__":
    from kfp.v2 import compiler
    compiler.Compiler().compile(
        pipeline_func=pipeline_prediction,
        package_path='async_prediction.json')