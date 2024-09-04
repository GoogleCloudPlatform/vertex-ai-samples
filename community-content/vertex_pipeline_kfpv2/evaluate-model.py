from kfp.v2 import dsl

@dsl.component(base_image='python:3.8',packages_to_install=['google-cloud-aiplatform==1.36.0'])
def evaluate_model(
    model_name: str,
    project_id: str,
    location: str,
    data_uris: str,
):
    from google.cloud import aiplatform

    aiplatform.init(
        project=project_id,
        location=location,
    )
    experiment_model = aiplatform.get_experiment_model(model_name)
    lr_model = experiment_model.load_model()
    evaluate_job = lr_model.evaluate(
        prediction_type="regression",
        target_field_name="type",
        data_source_uris=[data_uris],
        staging_bucket="gs://model-bucket/evaluation",
    )
    evaluate_job.wait()

@dsl.pipeline(name='model-evaluation')
def pipeline_evaluation():
    evaluate_model("lr-model", "990000000009", "us-west1", "gs://path/to/evaluation_dataset.csv")

if __name__ == "__main__":
    from kfp.v2 import compiler
    compiler.Compiler().compile(
        pipeline_func=pipeline_evaluation,
        package_path='evaluate_model.json')