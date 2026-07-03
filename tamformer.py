import os
import sys
import yaml
import numpy as np
import tensorflow as tf
import random as rn
import copy
from tensorflow.compat.v1.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Concatenate, BatchNormalization, Softmax, Flatten, Add, Activation
from tensorflow.keras import layers, activations
from tensorflow import keras
from tensorflow.keras.applications import vgg16


class masking_models(layers.Layer):
    def __init__(self, final_out=1, func='sigmoid'):
        super(masking_models, self).__init__()
        self.masking_model = keras.Sequential([Dense(128, activation='relu'),
                                               Dropout(0.1),
                                               Dense(64, activation='relu'),
                                               Dropout(0.1),
                                               Dense(32, activation='relu'),
                                               Dropout(0.1),
                                               Dense(final_out, activation=func)])

    def call(self, inputs):
        return self.masking_model(inputs)


class LearnedCausalMask(layers.Layer):
    def __init__(self, query_len, key_len, rate=0.1):
        super(LearnedCausalMask, self).__init__()
        self.query_len = query_len
        self.key_len = key_len
        self.masking_model = keras.Sequential([
            Dense(128, activation='relu'),
            Dropout(rate),
            Dense(64, activation='relu'),
            Dropout(rate),
            Dense(32, activation='relu'),
            Dropout(rate),
            Dense(key_len, activation='sigmoid')
        ])

    def call(self, inputs):
        learned_mask = self.masking_model(inputs)
        causal_mask = tf.linalg.band_part(tf.ones((self.query_len, self.key_len)), -1, 0)
        causal_mask = tf.expand_dims(causal_mask, axis=0)
        return learned_mask * causal_mask


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, normalization=True, cross_attention=False):
        super(TransformerBlock, self).__init__()
        self.cross_attention = cross_attention
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        if normalization:
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.normalization = normalization

    def call(self, inputs, training=None, attention_mask=None):
        if self.cross_attention:
            attn_output = self.att(inputs[0], inputs[1], attention_mask=attention_mask)
            attn_output = self.dropout1(attn_output, training=training)
            if self.normalization:
                out1 = self.layernorm1(inputs[0] + attn_output)
            else:
                out1 = inputs[0] + attn_output
        else:
            attn_output = self.att(inputs, inputs, attention_mask=attention_mask)
            attn_output = self.dropout1(attn_output, training=training)
            if self.normalization:
                out1 = self.layernorm1(inputs + attn_output)
            else:
                out1 = inputs + attn_output
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        if self.normalization:
            return self.layernorm2(out1 + ffn_output)
        else:
            return out1 + ffn_output


class QueryEmbedding(layers.Layer):
    def __init__(self, num_of_queries, embed_dim):
        super(QueryEmbedding, self).__init__()
        self.query_emb = layers.Embedding(input_dim=num_of_queries, output_dim=embed_dim)
        self.num_of_queries = num_of_queries

    def call(self, x):
        queries = tf.range(start=0, limit=self.num_of_queries, delta=1)
        queries = self.query_emb(queries)
        return queries


class PositionEmbedding(layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        return x + positions


class TAMformer(object):
    def __init__(self, model_opts=None, auxiliary_loss=False):
        self.model_opts = model_opts
        self.auxiliary_loss = auxiliary_loss

    def _active_modalities(self):
        input_types = self.model_opts['obs_input_type']
        feat_sizes = self.model_opts['feat_size']
        pose_enabled = self.model_opts.get('pose_enabled', True)
        active = []
        active_sizes = []
        for input_type, feat_size in zip(input_types, feat_sizes):
            if input_type == 'pose' and not pose_enabled:
                continue
            active.append(input_type)
            active_sizes.append(feat_size)
        return active, active_sizes

    def _target_dim(self):
        target_dim = self.model_opts.get('target_dim', (224, 224))
        return tuple(target_dim)

    def _local_context_encoder(self, seq_len, feat_size):
        target_dim = self._target_dim()
        context_input = Input((seq_len, target_dim[0], target_dim[1], 3), name='local_context')
        backbone_weights = self.model_opts.get('backbone_weights', 'imagenet')
        if backbone_weights in ['none', 'None', None]:
            backbone_weights = None
        backbone = vgg16.VGG16(input_shape=(target_dim[0], target_dim[1], 3),
                              include_top=False,
                              weights=backbone_weights,
                              pooling='max')
        backbone.trainable = not self.model_opts.get('freeze_backbone', False)
        processed = Lambda(lambda x: vgg16.preprocess_input(x), name='vgg16_preprocess')(context_input)
        context_features = layers.TimeDistributed(backbone, name='local_context_vgg16')(processed)
        if context_features.shape[-1] != feat_size:
            context_features = Dense(feat_size, activation='relu', name='local_context_projection')(context_features)
        return context_input, context_features

    def tamformer(self):
        input_types, feat_sizes = self._active_modalities()
        num_modalities = len(input_types)
        seq_len = self.model_opts.get('seq_len', self.model_opts.get('sequence_length', 136))
        obs_length = self.model_opts.get('obs_length', seq_len)
        step = self.model_opts.get('step', 1)
        num_classes = self.model_opts.get('num_classes', 1)
        dropout_rate = self.model_opts.get('dropout', 0.1)
        prediction_mode = self.model_opts.get('prediction_mode', 'temporal')
        trainable_backbone = self.model_opts.get('trainable_backbone', False)

        inputs = []
        modality_features = []
        for input_type, feat_size in zip(input_types, feat_sizes):
            if input_type == 'local_context' and trainable_backbone:
                context_input, context_features = self._local_context_encoder(seq_len, feat_size)
                inputs.append(context_input)
                modality_features.append(context_features)
            else:
                layer_name = input_type if input_type != 'box' else 'bbox'
                modality_input = Input((seq_len, feat_size), name=layer_name)
                inputs.append(modality_input)
                modality_features.append(modality_input)

        embeddings = [PositionEmbedding(seq_len, feat_sizes[i])(modality_features[i]) for i in range(num_modalities)]
        concatenated_inputs = Concatenate(axis=-1, name='available_modalities')(modality_features)

        if num_classes > 1 or prediction_mode == 'final':
            query_inputs = concatenated_inputs
            query_len = seq_len
        else:
            query_inputs = Lambda(lambda s: s[:, obs_length::step], name='prediction_queries')(concatenated_inputs)
            query_len = int((self.model_opts['seq_len'] - self.model_opts['obs_length']) / self.model_opts['step'])

        masks_encoder = LearnedCausalMask(seq_len, seq_len, rate=dropout_rate)(concatenated_inputs)
        masks_decoder = LearnedCausalMask(query_len, seq_len, rate=dropout_rate)(query_inputs)

        transformer_blocks = [TransformerBlock(feat_sizes[i], 6, 1024, rate=dropout_rate,
                                               normalization=True, cross_attention=False)
                              (embeddings[i], attention_mask=masks_encoder) for i in range(num_modalities)]

        concatenated_encodings = Concatenate(axis=-1, name='encoded_modalities')(transformer_blocks)
        query_transformer = TransformerBlock(sum(feat_sizes), 6, 1024, rate=dropout_rate,
                                             normalization=True, cross_attention=True)\
                                            ([query_inputs, concatenated_inputs], attention_mask=masks_decoder)

        cross_transformer_block = TransformerBlock(sum(feat_sizes), 6, 1024, rate=dropout_rate,
                                                   normalization=True, cross_attention=True)\
                                                  ([query_transformer, concatenated_encodings], attention_mask=masks_decoder)

        if num_classes > 1:
            if prediction_mode == 'sequence':
                x = layers.TimeDistributed(Dense(64, activation='relu'))(cross_transformer_block)
                x = Dropout(dropout_rate)(x)
                x = layers.TimeDistributed(Dense(32, activation='relu'))(x)
                x = Dropout(dropout_rate)(x)
                outputs = layers.TimeDistributed(Dense(num_classes), name='motion_logits')(x)
            else:
                x1 = Lambda(lambda s: s[:, -1], name='final_decoded_token')(cross_transformer_block)
                x2 = Dropout(dropout_rate)(x1)
                x3 = Dense(64, activation='relu')(x2)
                x4 = Dropout(dropout_rate)(x3)
                x5 = Dense(32, activation='relu')(x4)
                x6 = Dropout(dropout_rate)(x5)
                outputs = Dense(num_classes, name='motion_logits')(x6)
        else:
            outputs = []
            for i in range(query_len):
                x1 = Lambda(lambda s, i=i: s[:,i])(cross_transformer_block)
                x2 = Dropout(dropout_rate)(x1)
                x3 = Dense(64, activation='relu')(x2)
                x4 = Dropout(dropout_rate)(x3)
                x5 = Dense(32, activation='relu')(x4)
                x6 = Dropout(dropout_rate)(x5)
                x7 = Dense(1, activation='sigmoid', name='output_'+str(i))(x6)
                outputs.append(x7)

        model = Model(inputs, outputs, name='tamformer')

        if self.auxiliary_loss and num_classes == 1:
            to_add_losses = []
            for i in range(query_len):
                #ce = K.binary_crossentropy(tf.round(o), tf.round(outputs[-1])) #L_r could be the ce with the closest prediction
                mse = K.square(Lambda(lambda s, i=i: s[:,i])(cross_transformer_block) - Lambda(lambda s, i=i: s[:,-1])(cross_transformer_block))
                loss = K.mean(mse)
                to_add_losses.append(loss)
            model.add_loss(to_add_losses)

        return model
