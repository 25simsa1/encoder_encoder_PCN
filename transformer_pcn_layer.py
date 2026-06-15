import tensorflow as tf
from dense_pcn_layer import DensePCNLayer
import numpy as np

class AttentionPCNLayer:
    is_clamped : tf.Variable # bool
    fix_wts_b : tf.Variable # bool
    prev_layer : DensePCNLayer
    next_layers: list
    output_shape : tuple
    d_model: int
    num_heads : int
    mask : tf.Variable
    def __init__(self, d_model, num_heads, prev_layer:object, next_layers:list=None, mask:tf.Tensor=None):
        self.is_clamped = True
        self.fix_wts_b = True
        self.prev_layer = prev_layer
        self.next_layers = [] if next_layers is None else next_layers
        self.output_shape = None
        self.d_model = d_model
        self.num_heads = num_heads
        if mask is not None:
            self.mask = (mask[:, :, None]+mask[:, :, None])[:, None, :, :]
        else:
            self.mask=None
        
    
    def __call__(self, x:tf.Tensor, mask:tf.Tensor=None):
        if mask is not None:
            self.mask = (mask[:, :, None]+mask[:, :, None])[:, None, :, :]
        q, k, v = tf.split(tf.transpose(tf.reshape(x, (*x.shape[:2], self.num_heads, 3*(self.d_model//self.num_heads))), perm=[0, 2, 1, 3]), 3, -1)
        attention = ( q @ tf.linalg.matrix_transpose(k) ) / (self.d_model//self.num_heads)
        if self.mask is not None:
            attention += self.mask
        self.output_shape=(*x.shape[:2], self.d_model)
        return tf.reshape(tf.transpose(tf.nn.softmax(attention, axis=-1) @ v, perm = [0, 2, 1, 3]), self.output_shape)
        
    def predict_next(self):
        return self(self.prev_layer.predict_next())

class AddNormalizePCNLayer:
    is_clamped : tf.Variable # bool
    fix_wts_b : tf.Variable # bool
    gamma : tf.Variable # float
    beta : tf.Variable # float
    prev_layers : list
    next_layers : list
    learning_rate : float
    def __init__(self, learning_rate:float, prev_layers:list, next_layers:list=None):
        self.is_clamped = True
        self.fix_wts_b = False
        self.learning_rate = learning_rate
        self.gamma = tf.Variable(tf.cast(1., tf.float32), trainable=False)
        self.beta = tf.Variable(tf.cast(0., tf.float32), trainable=False)
        self.prev_layers = [] if prev_layers is None else prev_layers
        self.next_layers = [] if next_layers is None else next_layers
        self.output_shape = None

    def __call__(self, x1:tf.Tensor, x2:tf.Tensor):
        input = x1+x2
        mean, variance = tf.nn.moments(input, axes=[-1], keepdims=True)
        norm = (input - mean)/tf.sqrt(variance + 1e-7)
        self.output_shape = norm.shape
        return self.gamma*norm+self.beta
    
    def predict_next(self):
        return self(self.prev_layers[0].predict_next(), self.prev_layers[1].predict_next())
    
    def update_state(self):
        pass

    def init_state(self):
        pass

    def update_wts(self):
        if not self.fix_wts_b:
            # 0.5*(self.predict_next() - layer.predict_prev())^2
            # (self.predict_next() - layer.predict_prev())*norm

            d_pred = 0.
            num_layers = 0
            for layer in self.next_layers:
                if layer.is_clamped:
                    continue
                num_layers+=1
                x = (self.predict_next() - layer.predict_prev())*(self.predict_next()-self.beta)/self.gamma
                d_pred += tf.reduce_mean(x, axis=tf.range(0, tf.rank(x)))
            if num_layers!=0:
                self.gamma.assign_sub(self.learning_rate*d_pred)

    def update_b(self):
        if  not self.fix_wts_b:
            # 0.5*(self.predict_next() - layer.predict_prev())^2
            # (self.predict_next() - layer.predict_prev())

            d_pred = 0.
            num_layers = 0
            for layer in self.next_layers:
                if layer.is_clamped:
                    continue
                num_layers+=1
                x = (self.predict_next() - layer.predict_prev())
                d_pred += tf.reduce_mean(x, axis=tf.range(0, tf.rank(x)))
            if num_layers!=0:
                self.beta.assign_sub(self.learning_rate*d_pred)
            



class TransformerPCNLayer:
    is_clamped : tf.Variable # bool
    fix_wts_b : tf.Variable # bool
    kqv_layer : DensePCNLayer
    attention_layer : AttentionPCNLayer
    attention_dense_layer : DensePCNLayer
    attention_addnorm_layer : AddNormalizePCNLayer
    feed_forward_layers : list[DensePCNLayer]
    feed_forward_addnorm_layer : AddNormalizePCNLayer
    num_layers : int
    num_heads : int
    input_dim :int
    learning_rate:float
    prev_layer : object
    next_layers : list
    mask : tf.Tensor
    def __init__(self, num_layers:int, input_dim:int, num_heads:int, learning_rate:float, prev_layer:object, next_layers:list=None, mask:tf.Tensor=None):
        self.is_clamped = False
        self.fix_wts_b = False
        self.prev_layer = prev_layer
        self.next_layers = next_layers
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.kqv_layer = DensePCNLayer(3*input_dim, learning_rate, 'linear', prev_layer)
        self.mask = mask
        self.attention_layer = AttentionPCNLayer(input_dim, num_heads, self.kqv_layer)
        self.attention_dense_layer = DensePCNLayer(input_dim, learning_rate, 'linear', self.attention_layer)
        self.attention_addnorm_layer = AddNormalizePCNLayer(learning_rate, [self.prev_layer, self.attention_dense_layer])
        self.feed_forward_layers = []
        prev = self.attention_addnorm_layer
        for _ in range(num_layers-1):
            self.feed_forward_layers.append(DensePCNLayer(input_dim, learning_rate, 'relu', prev))
            prev = self.feed_forward_layers[-1]
        self.feed_forward_layers.append(DensePCNLayer(input_dim, learning_rate, 'linear', prev))
        self.feed_forward_addnorm_layer = AddNormalizePCNLayer(learning_rate, [self.attention_addnorm_layer, self.feed_forward_layers[-1]])

        self.kqv_layer.next_layers = [self.attention_layer]
        self.attention_layer.next_layers = [self.attention_dense_layer]
        self.attention_dense_layer.next_layers = [self.attention_addnorm_layer]
        self.attention_addnorm_layer.next_layers = [self.feed_forward_layers[0]]
        for i in range(num_layers-1):
            self.feed_forward_layers[i].next_layers = [self.feed_forward_layers[i+1]]
        self.feed_forward_layers[-1].next_layers = [self.feed_forward_addnorm_layer]
        self.feed_forward_addnorm_layer.next_layers = [] if next_layers is None else next_layers

    def get_layers(self):
        return [self.kqv_layer, self.attention_dense_layer, self.attention_addnorm_layer, *self.feed_forward_layers, self.feed_forward_addnorm_layer]
    
    def set_next_layers(self, next_layers:list):
        self.next_layers = next_layers
        self.feed_forward_addnorm_layer.next_layers = next_layers
    
    def predict_next(self):
        return self.feed_forward_addnorm_layer.predict_next()
    
    def predict_prev(self):
        return self.kqv_layer.predict_prev()
    
    def __call__(self, x:tf.Tensor, mask:tf.Tensor = None):
        if mask is not None:
            self.mask = mask
        kqv = self.kqv_layer(x)
        attention = self.attention_layer(kqv, mask=self.mask)
        attention = self.attention_dense_layer(attention)
        attention_norm = self.attention_addnorm_layer(x, attention)
        output = attention_norm
        for i in range(self.num_layers):
            output = self.feed_forward_layers[i](output)
        return self.feed_forward_addnorm_layer(attention_norm, output)

class PositionalEncodingLayer:
    is_clamped : tf.Variable # bool
    fix_wts_b : tf.Variable # bool
    prev_layer : DensePCNLayer
    next_layers: list
    output_shape : tuple
    d_model: int
    activation: str
    def __init__(self, d_model:int, prev_layer:object, next_layers:list=None):
        self.is_clamped = False
        self.fix_wts_b = True
        self.prev_layer = prev_layer
        self.next_layers = [] if next_layers is None else next_layers
        self.output_shape = None
        self.d_model = d_model
        self.activation = 'linear'

    def __call__(self, x:tf.Tensor):
        angle_rads = np.arange(x.shape[1])[:, None] / np.power(10000, (2 * (np.arange(self.d_model)[None, :] // 2)) / np.float32(self.d_model))
    
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[None]
    
        return tf.cast(pos_encoding, dtype=tf.float32) + x
    
    def predict_next(self):
        return self(self.prev_layer.predict_next())
    
    # assume 1 next layer
    def predict_prev(self):
        angle_rads = np.arange(self.next_layers[0].predict_prev().shape[1])[:, None] / np.power(10000, (2 * (np.arange(self.d_model)[None, :] // 2)) / np.float32(self.d_model))
    
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[None]

        return self.next_layers[0].predict_prev() - tf.cast(pos_encoding, dtype=tf.float32)
    
    def pred_loss_d_input(self, x:tf.Tensor):
        return 1.