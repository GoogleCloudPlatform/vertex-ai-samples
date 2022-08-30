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

### Code

- Put all your installs and imports in a setup section.
- Save the notebook with the Table of Contents open.
- Write Python 3 compatible code.
- Follow the [Google Python Style guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md) and write readable code.
- Keep cells small (max ~20 lines).

## TensorFlow code style

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