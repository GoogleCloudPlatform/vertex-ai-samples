## General style examples

### Notebook heading

- Include the collapsed license at the top (this uses Colab's "Form" mode to hide the cells).
- Only include a single H1 title.
- Include the button-bar immediately under the H1.
- Check that the Colab and GitHub links at the top are correct.

### Notebook sections

- Use H2 (##) and H3 (###) titles for notebook section headings.
- Use [sentence case to capitalize titles and headings](https://developers.google.com/style/capitalization#capitalization-in-titles-and-headings). ("Train the model" instead of "Train the Model")
- Include a brief text explanation before any code cells.
- Use short titles/headings: "Download the data", "Build the model", "Train the model".

### Writing style

- Use [present tense](https://developers.google.com/style/tense). ("You receive a response" instead of "You will receive a response")
- Use [active voice](https://developers.google.com/style/voice). ("The service processes the request" instead of "The request is processed by the service")
- Use [second person](https://developers.google.com/style/person) and an imperative style. 
    - Correct examples: "Update the field", "You must update the field"
    - Incorrect examples: "Let's update the field", "We'll update the field", "The user should update the field"
- **Googlers**: Please follow our [branding guidelines](http://goto/cloud-branding).


## Authoring guidelines

### Focus

Notebooks for official are expected to be narrow focused, which highlight a subset of features of a Vertex AI product/service.
The product/feature is to be highlighted in the Overview section. For example:

```
This tutorial demonstrates using Vertex AI Training to train an XGBoost model using a XGBoost pre-built training container.
```

In the above example, the Vertex AI product/service is `Vertex AI Training` and the feature is `XGBoost pre-built training container`.

### Scope

Notebooks for official are expected to be narrow in scope, without extra extraneous steps. For example, if the notebook is about training, we discourage ending the notebook with deploying the model and doing an online/batch prediction. On the later, we recommend a separate notebook about prediction that uses a pretrained model.

#### Training

Notebooks for training should be constructed as follows:

1. If the training script(s) are small, embed them in the notebook and use %writefile to store them locally.
2. If the training script(s) are large, store them in our public bucket: gs://cloud-samples-data/vertex-ai/dataset-management/script, and use !wgets to retrieve and store the script locally.
3. Train the model using the Vertex AI SDK methods for custom training.
4. Preferrable have the service upload the trained model to the Vertex AI Model Registry.
4. Have the script do an evaluation.
5. Retrieve the evaluation metrics and attach them as an artifact to the corresponding entry in the Model Registry.
6. Optionally, download the model artifacts and test locally -- i.e., make a local prediction request.

#### Evaluation

Notebooks for evaluation should be constructed as follows:

1. Use a pretrained model from a public repository.
2. Upload the pretrained model to the Vertex AI Model Registry.
3. Perform a model evaluation.
4. Review the model evaluation.
4. Attach the model evaluation to the corresponding entry in the Model Registry.

#### Prediction

Notebooks for prediction should be constructed as follows:

1. Use a pretrained model from a public repository.
2. If relevant, attach a serving function to the model artifacts.
3. Upload the pretrained model to the Vertex AI Model Registry.
4. For online:<br/>
    A. Deploy the model.<br/>
    B. Perform an online prediction.</br>
    C. Review the result.
5. For batch:<br/>
    A. Perform a batch prediction.</br/>
    B. Review the result.
    
## Code

- Put all your installs and imports in a setup section.
- Save the notebook with the Table of Contents open.
- Write Python 3 compatible code.
- Follow the [Google Python Style guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md) and write readable code.
- Keep cells small (max ~20 lines).

### TensorFlow code style

Use the highest level API that gets the job done (unless the goal is to demonstrate the low level API). For example, when using Tensorflow:

- Use TF.keras.Sequential > keras functional api > keras model subclassing > ...

- Use  model.fit > model.train_on_batch > manual GradientTapes.

- Use eager-style code.

- Use tensorflow_datasets and tf.data where possible.

### Notebook code style examples

 - Notebooks are for people. Write code optimized for clarity.

 - Demonstrate small parts before combining them into something more complex. Like below:
 
 
 ```
 # Build the model
import tensorflow as tf

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(10, activation="relu", input_shape=(None, 5)),
        tf.keras.layers.Dense(3),
    ]
)
```

```
# Run the model on a single batch of data, and inspect the output.
import numpy as np

result = model(tf.constant(np.random.randn(10, 5), dtype=tf.float32)).numpy()

print("min:", result.min())
print("max:", result.max())
print("mean:", result.mean())
print("shape:", result.shape)
```

```
# Compile the model for training
model.compile(
    optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy
)
```

- Keep examples quick. Use small datasets, or small slices of datasets. You don't need to train to convergence, train until it's obvious it's making progress.

- For a large example, don't try to fit all the code in the notebook. Add python files to tensorflow examples, and in the notebook run: 
! pip3 install git+https://github.com/tensorflow/examples