from kfp.v2 import dsl

@dsl.component(base_image='python:3.8',packages_to_install=['google-cloud-aiplatform==1.36.0'])
def run_experiment(
    project_id: str,
    location: str,
    experiment_name: str,
    run_name: str,
):
    import json
    from google.cloud import aiplatform

    aiplatform.init(
        project=project_id,
        location=location,
    )
    test_expt = aiplatform.Experiment.create(experiment_name)
    test_run = aiplatform.ExperimentRun.create(run_name, experiment=test_expt)
    metric = test_run.get_classification_metrics()[0]
    print(metric)



@dsl.pipeline(name='run_experiment')
def pipeline_run_experiment():
    run_experiment('990000000009', 'us-west1', 'test-experiment', 'test-run')


if __name__ == "__main__":
    from kfp.v2 import compiler
    compiler.Compiler().compile(
        pipeline_func=pipeline_run_experiment,
        package_path='run_experiment.json')