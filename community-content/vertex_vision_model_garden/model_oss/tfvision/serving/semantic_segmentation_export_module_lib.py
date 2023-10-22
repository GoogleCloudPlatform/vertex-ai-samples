"""Semantic segmentation input and model functions for serving/inference."""


import tensorflow as tf

from official.vision.serving import semantic_segmentation


class OssSegmentationModule(semantic_segmentation.SegmentationModule):
  """OSS Segmentation Module."""

  def serve(self, images):
    """Cast image to float and run inference.

    Overrides the method in the super class, and changes the output format.

    Args:
      images: uint8 Tensor of shape [batch_size, None, None, 3]

    Returns:
      Dict containing the following key value pairs:
        category_bytes: Encoded PNG image of grayscale output categories.
        score_bytes: Encoded PNG image of grayscale probability scores mapped to
          [0, 255].
    """
    result = super().serve(images)
    logits = result['logits']

    probabilities = tf.nn.softmax(logits)
    scores = tf.reduce_max(probabilities, 3, keepdims=True)
    scores = tf.cast(tf.minimum(scores * 255.0, 255), dtype=tf.uint8)

    categories = tf.cast(
        tf.expand_dims(tf.argmax(logits, 3), -1), dtype=tf.int32
    )

    score_bytes = tf.map_fn(
        tf.image.encode_png, scores, back_prop=False, dtype=tf.string
    )
    category_bytes = tf.map_fn(
        tf.image.encode_png,
        tf.cast(categories, dtype=tf.uint8),
        back_prop=False,
        dtype=tf.string,
    )

    outputs = {
        'category_bytes': tf.identity(category_bytes, name='category_bytes'),
        'score_bytes': tf.identity(score_bytes, name='score_bytes'),
    }

    return outputs
