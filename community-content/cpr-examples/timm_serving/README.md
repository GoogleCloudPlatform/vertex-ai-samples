# CPR Example: PyTorch Image Models (timm)

## About CPR

CPR (custom prediction routines) is a framework designed by Google Cloud developers to make it easier to combine machine learning models with custom preprocessing and postprocessing logic in a real-time serving application. 

## Using this example

This code is a self-contained example of a custom model server project built using CPR.

As is, you can use it to serve the ViT-Small image classification model from Ross Wightman's fantastic [`timm`](https://github.com/rwightman/pytorch-image-models) library of image model implementations in PyTorch. Other Imagenet-based image classification models from timm probably also work. Both CPU and GPU are supported.

### Requirements

In order to use this example, you'll need Docker and Python 3 installed on your system.

To get started, first create a virtual environment in an empty directory:
```sh
mkdir cpr-example
python3 -m venv cpr-example
cd cpr-example && source bin/activate
``` 

Then, clone the [vertex-ai-samples repo](https://github.com/GoogleCloudPlatform/vertex-ai-samples) in that directory:
```sh
git clone https://github.com/GoogleCloudPlatform/vertex-ai-samples.git
cd vertex-ai-samples/community-content/cpr-examples/timm_serving
```

Finally, install the Python modules required to build and run the model server:
```sh
pip install -r requirements.txt
```

### Predictor

The `TimmPredictor` class in `timm_serving/predictor.py` implements most of the important logic for the server.

- `load(artifacts_dir)`: The predictor's `load` method is called when the server starts up in order to set up the predictor, usually by loading model weights and any artifacts needed for preprocessing and postprocessing. In this example, we initialize the saved model from the `state_dict.pth` file located inside the `artifacts_dir` folder and create the preprocessing transform from the model config.

- `preprocess`, `predict`, `postprocess`: These methods are applied in sequence to the deserialized JSON data from each request. 
    - `preprocess` decodes images from base64 and apply cropping, scaling and normalizing transforms. 
    - `predict` runs the ViT-Small model on the preprocessed images and returns class scores.
    - `postprocess` finds the top five classes and packs the class names, probabilities, and indices in a serializable result.

### Building the container

To build the model server locally, run the build command:
```sh
python build.py build
```

You can edit configuration values such as the model server's base image, the name and tag assigned to the image, and the path where model weights are stored locally.

When you run the build command, model weights are downloaded and the model server container is built.

### Running local tests

`test.py` contains a suite of unit tests for the predictor as well as end-to-end tests for the model server. 

To run the tests:
```sh
python test.py
```

All of the test images are public domain.
- [Cat](https://commons.wikimedia.org/wiki/File:Stray_cat_on_wall.jpg)
- [Airplane](https://commons.wikimedia.org/wiki/File:Airplanes_jets.jpg)
- The infamous [mandrill](https://commons.wikimedia.org/wiki/File:Wikipedia-sipi-image-db-mandrill-4.2.03.png)

### Deploying to Vertex AI

Before uploading or deploying the container, you'll need to modify `config.py` to set appropriate values for: 
- `project_id`: Your GCP project id.
- `region`: Region where the model will be uploaded and deployed.
- `repository`: [Artifact Registry repository](https://cloud.google.com/artifact-registry/docs/repositories/create-repos) in your project where the container image will be uploaded. 
- `artifacts_gcs_dir`: Folder in a [Google Cloud Storage bucket](https://cloud.google.com/storage/docs/creating-buckets) where the model weights will be uploaded.

Once this is done, first upload the model:
```sh
python build.py upload
```

Then deploy it:
```sh
python build.py deploy
```

If you run the deploy command again, it will create a new endpoint. If you want to undeploy the model, you can do so using the Vertex AI dashboard on the Google Cloud console, or use `gsutil ai endpoints undeploy` from the command line.
