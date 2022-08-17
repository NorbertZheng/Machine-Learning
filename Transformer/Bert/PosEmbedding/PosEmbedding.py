#!/usr/bin/env python3
"""
Created on 22:16, Aug. 16th, 2022

@author: Norbert Zheng
"""
import tensorflow as tf
import tensorflow.keras.backend as K

__all__ = [
    "PosEmbedding",
]

class PosEmbedding(tf.keras.layers.Layer):
    """
    Turn integers (position) into dense vectors of fixed size.
    e.g. [[-4], [10]] -> [[0.25, 0.1], [0.6, -0.2]]

    Expand mode: negative integers (relative position) could be used in this mode.
        # Input shape
            2D tensor with shape: `(batch_size, sequence_length)`.

        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, output_dim)`.

    Add mode:
        # Input shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.

        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.

    Concat mode:
        # Input shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.

        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim + output_dim)`.
    """
    modes = ["expand", "add", "concat"]

    def __init__(self, input_dim, output_dim, mode=modes[0], embeddings_initializer="uniform", embeddings_regularizer=None,
        activity_regularizer=None, embeddings_constraint=None, mask_zero=False, **kwargs):
        """
        Initialize `PosEmbedding` class.
        :param input_dim: The size of input.
        :param output_dim: The size of output.
        :param mode: The mode of position embedding, [expand, add, concat].
        :param embeddings_initializer: The type of initializer used to initialize `embeddings` matrix.
        :param embeddings_regularizer: The type of regularizer used to regularize `embeddings` matrix.
        :param activity_regularizer: The type of regularizer used to regularize `activity` vector.
        :param embeddings_constraint: The type of constraint used to restrict `embeddings` matrix.
        :param mask_zero: The index that represents padding. Only works in `append` mode.
        """
        # Initialize super-class (Layer) style class.
        super(PosEmbedding, self).__init__(**kwargs)
        # Initialize parameters.
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mode = mode
        self.embeddings_initializer = tf.keras.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = tf.keras.regularizers.get(embeddings_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.embeddings_constraint = tf.constraints.get(embeddings_constriant)
        self.mask_zero = mask_zero
        # Initialize variables.
        self.embeddings = None

    """
    keras funcs
    """
    # def get_config func
    def get_config(self):
        """
        Get the configuration of `PosEmbedding` class.
        :return config: The configuration of `PosEmbedding` class.
        """
        # Get the config of super class.
        config_base = super(PosEmbedding, self).get_config()
        # Get the additional config of `PosEmbedding` class.
        config_posemb = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "mode": self.mode,
            "embeddings_initializer": tf.keras.initializers.get(self.embeddings_initializer),
            "embeddings_regularizer": tf.keras.regularizers.get(self.embeddings_regularizer),
            "activity_regularizer": tf.keras.regularizers.get(self.activity_regularizer),
            "embeddings_constraint": tf.keras.constraints.get(self.embeddings_constraint),
            "mask_zero": self.mask_zero,
        }
        # Return the final config.
        return dict(list(config_base.items())+list(config_posemb.items()))

    # def build func
    def build(self, input_shape):
        """
        Build the network of `PosEmbedding` class.
        :param input_shape:
        """
        # Build the network of super class.
        super(PosEmbedding, self).build(input_shape)
        # Expand-mode position embeddings.
        if self.mode == "expand":
            self.embeddings = self.add_weight(
                shape = (self.input_dim * 2 + 1, self.output_dim),
                initializer = self.embeddings_initializer,
                name = "embeddings",
                regularizer = self.embeddings_regularizer,
                constraint = self.embeddings_constraint,
            )
        # [Add,Concat]-mode position embeddings.
        else:
            self.embeddings = self.add_weight(
                shape = (self.input_dim, self.output_dim),
                initializer = self.embeddings_initializer,
                name = "embeddings",
                regularizer = self.embeddings_regularizer,
                constraint = self.embeddings_constraint,
            )

    # def call func
    def call(self, inputs, **kwargs):
        """
        Compute the embeddings of inputs.
        :param inputs:
        :return outputs:
        """
        # Expand-mode position embeddings.
        if self.mode == "expand":
            # Cast precision to `int32`.
            if K.dtype(inputs) != "int32": inputs = K.cast(inputs, "int32")
            # Find the corresponding embeddings from the table.
            # Each of embeddings is indexed by inputs (scale).
            pos_embeddings = K.gather(
                self.embeddings,
                K.minimum(K.maximum(inputs, -self.input_dim), self.input_dim) + self.input_dim,
            )
            # Return the final outputs.
            return pos_embeddings
        # Add-mode position embeddings.
        elif self.mode == "add":
            # Get the shape of inputs.
            batch_size, seq_len, output_dim = K.shape(inputs)[0], K.shape(inputs)[1], K.shape(inputs)[2]
            # Get the corresponding embeddings, each of which is indexed by the position of sequence.
            pos_embeddings = K.tile(
                K.expand_dims(self.embeddings[:seq_len,:output_dim], axis=0),
                [batch_size, 1, 1],
            )
            # Return the final outputs.
            return inputs + pos_embeddings
        # Concat-mode position embeddings.
        elif self.mode == "concat":
            # Get the shape of inputs.
            batch_size, seq_len, output_dim = K.shape(inputs)[0], K.shape(inputs)[1], self.output_dim
            # Get the corresponding embeddings, each of which is indexed by the position of sequence.
            pos_embeddings = K.tile(
                K.expand_dims(self.embeddings[:seq_len,:output_dim], axis=0),
                [batch_size, 1, 1],
            )
            # Return the final outputs.
            return K.concatenate([inputs, pos_embeddings], axis=-1)
        # Unknow mode of position embeddings.
        else:
            raise ValueError((
                "ERROR: Unknown mode {} of position embeddings in PosEmbedding.PosEmbedding."
            ).format(self.mode))

    """
    embeddings funcs
    """
    # def compute_mask func
    def compute_mask(self, inputs, mask=None):
        """
        Compute the mask of output.
        :param inputs:
        :param mask:
        :return output_mask:
        """
        # Expand-mode position embeddings.
        if self.mode == "expand":
            # If mask_zero is not False, then mask_zero should be matrix.
            output_mask = K.not_equal(inputs, self.mask_zero) if self.mask_zero else None
        # [Add,Concat]-mode position embeddings.
        else:
            output_mask = mask
        # Return the final output_mask.
        return output_mask

    # def compute_output_shape func
    def compute_output_shape(self, input_shape):
        """
        Compute the shape of output.
        :param input_shape:
        :return output_shape:
        """
        # Expand-mode position embeddings.
        if self.mode == "expand":
            return input_shape + (self.output_dim,)
        # Add-mode position embeddings.
        elif self.mode == "add":
            return input_shape
        # Concat-mode position embeddings.
        elif self.mode == "concat":
            return input_shape[:-1] + (input_shape[-1] + self.output_dim,)
        # Unknown mode of position embeddings.
        else:
            raise ValueError((
                "ERROR: Unknown mode {} of position embeddings in PosEmbedding.PosEmbedding."
            ).format(self.mode))

if __name__ == "__main__":
    print("PosEmbedding: Hello World!")

