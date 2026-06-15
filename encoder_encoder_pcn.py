import tensorflow as tf
from conv_pcn_layer import Conv2DPCNLayer, MaxPool2DPCNLayer
from dense_pcn_layer import DensePCNLayer
from transformer_pcn_layer import TransformerPCNLayer, PositionalEncodingLayer, AttentionPCNLayer, AddNormalizePCNLayer
from typing import Literal
import gc
class InputPCNLayer:
    is_clamped : tf.Variable # bool
    fix_wts_b : tf.Variable # bool
    next_layers: list
    output_shape : tuple
    state : tf.Variable # tf.Tensor
    learning_rate:float
    def __init__(self, learning_rate: float, next_layers:list=None):
        self.is_clamped = True
        self.fix_wts_b = True
        self.next_layers = [] if next_layers is None else next_layers
        self.output_shape = None
        self.state = None
        self.learning_rate = learning_rate
        self.state_lr = learning_rate   # inference/relaxation rate (decoupled from weight lr)
        self.bias_lr = learning_rate    # kept for uniform driver setup (input has no bias)

    def update_state(self):
        if not self.is_clamped:
            average_d_pred = tf.zeros_like(self.state)
            average_d_state = tf.zeros_like(self.state)
            num_next_layers = 0
            for layer in self.next_layers:
                if layer.is_clamped:
                    continue
                num_next_layers += 1
                # print(layer)
                state = self.predict_next()
                pred_state = layer.predict_prev()
                if layer.activation == 'relu':
                    state = tf.nn.relu(state)
                    pred_state = tf.nn.relu(pred_state)
                average_d_pred += layer.pred_loss_d_input(self.predict_next())
                average_d_state += (state - pred_state)
            if num_next_layers!=0:
                self.state.assign_sub(self.state_lr * ((average_d_pred+average_d_state)/(2.*float(num_next_layers))))

    def update_wts(self):
        pass # there is no wts

    def update_b(self):
        pass # there is no bias
    
    def predict_next(self):
        return self.state
        
    def init_state(self):
        self.state = None

    def set_state(self, x:tf.Tensor):
        if self.state is None:
            self.state = tf.Variable(x, trainable=False)
        else:
            self.state.assign(x)
        self.output_shape = x.shape

class TransposePCNLayer:
    is_clamped : tf.Variable # bool
    fix_wts_b : tf.Variable # bool
    prev_layer : DensePCNLayer
    next_layers: list
    output_shape : tuple
    activation : str
    def __init__(self, prev_layer:object, next_layers:list=None):
        self.is_clamped = False
        self.fix_wts_b = True
        self.prev_layer = prev_layer
        self.next_layers = [] if next_layers is None else next_layers
        self.output_shape = None
        self.activation = 'linear'

    def __call__(self, x:tf.Tensor):
        self.output_shape = (*x.shape[:-2], x.shape[-1], x.shape[-2])
        # tf.rank(x) is a symbolic tensor in graph mode — avoid Python iteration.
        r = tf.rank(x)
        prefix = tf.range(0, r - 2, dtype=tf.int32)
        last = tf.stack([r - 1, r - 2])
        perm = tf.concat([prefix, last], axis=0)
        return tf.transpose(x, perm=perm)
    
    def predict_next(self):
        return self(self.prev_layer.predict_next())
    
    # assume 1 next layer
    def predict_prev(self):
        return self(self.next_layers[0].predict_prev())
    
    def pred_loss_d_input(self, x:tf.Tensor):
        return 1.

class FlattenPCNLayer:
    is_clamped : tf.Variable # bool
    fix_wts_b : tf.Variable # bool
    prev_layer : object
    next_layers: list
    output_shape : tuple
    input_shape : tuple
    activation : str
    def __init__(self, prev_layer:object, next_layers:list=None):
        self.is_clamped = False
        self.fix_wts_b = True
        self.prev_layer = prev_layer
        self.next_layers = [] if next_layers is None else next_layers
        self.output_shape = None
        self.input_shape = None
        self.activation = 'linear'

    def __call__(self, x:tf.Tensor):
        self.output_shape = (x.shape[0], -1)
        self.input_shape = x.shape
        return tf.reshape(x, (x.shape[0], -1))
    
    def predict_next(self):
        return self(self.prev_layer.predict_next())
    
    # assume 1 next layer
    def predict_prev(self):
        return tf.reshape(self.next_layers[0].predict_prev(), self.input_shape)
    
    def pred_loss_d_input(self, x:tf.Tensor):
        return 1.

class EncoderEncoderPCN:
    trainable_layers : list
    learning_rate : float
    img_input : object
    txt_input : object
    def __init__(self, learning_rate : float):
        self.trainable_layers = []
        self.learning_rate = learning_rate
        self.img_input = InputPCNLayer(learning_rate)
        self.trainable_layers.append(self.img_input)
        conv1 = Conv2DPCNLayer(64, (3, 3), learning_rate, 'relu', self.img_input)
        self.trainable_layers.append(conv1)
        self.img_input.next_layers = [conv1]
        conv2 = Conv2DPCNLayer(64, (3, 3), learning_rate, 'relu', conv1)
        self.trainable_layers.append(conv2)
        conv1.next_layers = [conv2]
        mp1 = MaxPool2DPCNLayer((2, 2), conv2)
        conv2.next_layers = [mp1]
        conv3 = Conv2DPCNLayer(128, (3, 3), learning_rate, 'relu', mp1)
        self.trainable_layers.append(conv3)
        mp1.next_layers = [conv3]
        conv4 = Conv2DPCNLayer(128, (3, 3), learning_rate, 'relu', conv3)
        self.trainable_layers.append(conv4)
        conv3.next_layers = [conv4]
        mp2 = MaxPool2DPCNLayer((2, 2), conv4)
        conv4.next_layers = [mp2]
        conv5 = Conv2DPCNLayer(256, (3, 3), learning_rate, 'relu', mp2)
        self.trainable_layers.append(conv5)
        mp2.next_layers = [conv5]
        conv6 = Conv2DPCNLayer(256, (3, 3), learning_rate, 'relu', conv5)
        self.trainable_layers.append(conv6)
        conv5.next_layers = [conv6]
        mp3 = MaxPool2DPCNLayer((2, 2), conv6)
        conv6.next_layers = [mp3]
        conv7 = Conv2DPCNLayer(512, (3, 3), learning_rate, 'relu', mp3)
        self.trainable_layers.append(conv7)
        mp3.next_layers = [conv7]
        conv8 = Conv2DPCNLayer(512, (3, 3), learning_rate, 'relu', conv7)
        self.trainable_layers.append(conv8)
        conv7.next_layers = [conv8]
        mp4 = MaxPool2DPCNLayer((2, 2), conv8)
        conv8.next_layers = [mp4]
        conv9 = Conv2DPCNLayer(1024, (3, 3), learning_rate, 'relu', mp4)
        self.trainable_layers.append(conv9)
        mp4.next_layers = [conv9]

        flatten1 = FlattenPCNLayer(conv9)
        conv9.next_layers = [flatten1]
        inter1 = DensePCNLayer(100, learning_rate, 'linear', flatten1)
        flatten1.next_layers = [inter1]
        self.trainable_layers.append(inter1)
        dense1 = DensePCNLayer(307200, learning_rate, 'relu', inter1)
        inter1.next_layers = [dense1]
        self.trainable_layers.append(dense1)
        inter2 = DensePCNLayer(100, learning_rate, 'linear', dense1)
        dense1.next_layers = [inter2]
        self.trainable_layers.append(inter2)
        dense2 = DensePCNLayer(102400, learning_rate, 'linear', inter2)
        inter2.next_layers = [dense2]
        self.trainable_layers.append(dense2)

        flatten3 = FlattenPCNLayer(conv8)
        conv8.next_layers.append(flatten3)
        inter3 = DensePCNLayer(100, learning_rate, 'linear', flatten3)
        flatten3.next_layers = [inter3]
        self.trainable_layers.append(inter3)
        dense5 = DensePCNLayer(582542, learning_rate, 'relu', inter3)
        inter3.next_layers = [dense5]
        self.trainable_layers.append(dense5)
        inter4 = DensePCNLayer(100, learning_rate, 'linear', dense5)
        dense5.next_layers = [inter4]
        self.trainable_layers.append(inter4)
        dense6 = DensePCNLayer(161817, learning_rate, 'linear', inter4)
        inter4.next_layers = [dense6]
        self.trainable_layers.append(dense6)

        flatten5 = FlattenPCNLayer(conv6)
        conv6.next_layers.append(flatten5)
        inter5 = DensePCNLayer(100, learning_rate, 'linear', flatten5)
        flatten5.next_layers = [inter5]
        self.trainable_layers.append(inter5)
        dense9 = DensePCNLayer(1279723, learning_rate, 'relu', inter5)
        inter5.next_layers = [dense9]
        self.trainable_layers.append(dense9)
        inter6 = DensePCNLayer(100, learning_rate, 'linear', dense9)
        dense9.next_layers = [inter6]
        self.trainable_layers.append(inter6)
        dense10 = DensePCNLayer(345871, learning_rate, 'linear', inter6)
        inter6.next_layers = [dense10]
        self.trainable_layers.append(dense10)

        flatten7 = FlattenPCNLayer(conv4)
        conv4.next_layers.append(flatten7)
        inter7 = DensePCNLayer(100, learning_rate, 'linear', flatten7)
        flatten7.next_layers = [inter7]
        self.trainable_layers.append(inter7)
        dense13 = DensePCNLayer(2654815, learning_rate, 'relu', inter7)
        inter7.next_layers = [dense13]
        self.trainable_layers.append(dense13)
        inter8 = DensePCNLayer(100, learning_rate, 'linear', dense13)
        dense13.next_layers = [inter8]
        self.trainable_layers.append(inter8)
        dense14 = DensePCNLayer(702332, learning_rate, 'linear', inter8)
        inter8.next_layers = [dense14]
        self.trainable_layers.append(dense14)

        flatten9 = FlattenPCNLayer(conv2)
        conv2.next_layers.append(flatten9)
        inter9 = DensePCNLayer(100, learning_rate, 'linear', flatten9)
        flatten9.next_layers = [inter9]
        self.trainable_layers.append(inter9)
        dense17 = DensePCNLayer(5433667, learning_rate, 'relu', inter9)
        inter9.next_layers = [dense17]
        self.trainable_layers.append(dense17)
        inter10 = DensePCNLayer(100, learning_rate, 'linear', dense17)
        dense17.next_layers = [inter10]
        self.trainable_layers.append(inter10)
        dense18 = DensePCNLayer(1429912, learning_rate, 'linear', inter10)
        inter10.next_layers = [dense18]
        self.trainable_layers.append(dense18)



        self.txt_input = InputPCNLayer(learning_rate)
        self.trainable_layers.append(self.txt_input)
        txt_embedding = DensePCNLayer(512, learning_rate, 'linear', self.txt_input)
        self.trainable_layers.append(txt_embedding)
        self.txt_input.next_layers = [txt_embedding]
        pos_encoding = PositionalEncodingLayer(512, txt_embedding)
        txt_embedding.next_layers = [pos_encoding]
        transformer1 = TransformerPCNLayer(3, 512, 8, learning_rate, pos_encoding)
        transformer1_layers = transformer1.get_layers()
        pos_encoding.next_layers=[transformer1_layers[0]]
        self.trainable_layers += transformer1_layers
        transformer2 = TransformerPCNLayer(3, 512, 8, learning_rate, self.trainable_layers[-1])
        transformer2_layers = transformer2.get_layers()
        self.trainable_layers[-1].next_layers = [transformer2_layers[0]]
        self.trainable_layers += transformer2_layers
        transformer3 = TransformerPCNLayer(3, 512, 8, learning_rate, self.trainable_layers[-1])
        transformer3_layers = transformer3.get_layers()
        self.trainable_layers[-1].next_layers = [transformer3_layers[0]]
        self.trainable_layers += transformer3_layers
        linear_1 = DensePCNLayer(1024, learning_rate, 'linear', self.trainable_layers[-1])
        self.trainable_layers[-1].next_layers = [linear_1]
        self.trainable_layers.append(linear_1)
        tp1 = TransposePCNLayer(linear_1)
        linear_1.next_layers = [tp1]
        linear_2 = DensePCNLayer(48, learning_rate, 'linear', tp1)
        tp1.next_layers = [linear_2]
        self.trainable_layers.append(linear_2)
        tp2 = TransposePCNLayer(linear_2)
        linear_2.next_layers = [tp2]
        # mask_bool = (mask==0)
        # new_mask_bool_48 = (mask_bool @ tf.abs(linear_2.wts)) > 0
        # new_mask_48 = tf.where(new_mask_bool_48, -1e9, 0.0) 
        transformer4 = TransformerPCNLayer(3, 1024, 8, learning_rate, tp2)
        transformer4_layers = transformer4.get_layers()
        tp2.next_layers = [transformer4_layers[0]]
        self.trainable_layers += transformer4_layers
        transformer5 = TransformerPCNLayer(3, 1024, 8, learning_rate, self.trainable_layers[-1])
        transformer5_layers = transformer5.get_layers()
        self.trainable_layers[-1].next_layers = [transformer5_layers[0]]
        self.trainable_layers += transformer5_layers
        transformer6 = TransformerPCNLayer(3, 1024, 8, learning_rate, self.trainable_layers[-1])
        transformer6_layers = transformer6.get_layers()
        self.trainable_layers[-1].next_layers = [transformer6_layers[0]]
        self.trainable_layers += transformer6_layers
        linear_3 = DensePCNLayer(2048, learning_rate, 'linear', self.trainable_layers[-1])
        self.trainable_layers[-1].next_layers = [linear_3]
        self.trainable_layers.append(linear_3)
        tp3 = TransposePCNLayer(linear_3)
        linear_3.next_layers = [tp3]
        linear_4 = DensePCNLayer(12, learning_rate, 'linear', tp3)
        tp3.next_layers = [linear_4]
        self.trainable_layers.append(linear_4)
        tp4 = TransposePCNLayer(linear_4)
        linear_4.next_layers = [tp4]
        # new_mask_bool_12 = (new_mask_bool_48 @ tf.abs(linear_4.wts)) > 0
        # new_mask_12 = tf.where(new_mask_bool_12, -1e9, 0.0) 
        transformer7 = TransformerPCNLayer(3, 2048, 8, learning_rate, tp4)
        transformer7_layers = transformer7.get_layers()
        tp4.next_layers = [transformer7_layers[0]]
        self.trainable_layers += transformer7_layers
        transformer8 = TransformerPCNLayer(3, 2048, 8, learning_rate, self.trainable_layers[-1])
        transformer8_layers = transformer8.get_layers()
        self.trainable_layers[-1].next_layers = [transformer8_layers[0]]
        self.trainable_layers += transformer8_layers
        transformer9 = TransformerPCNLayer(3, 2048, 8, learning_rate, self.trainable_layers[-1])
        transformer9_layers = transformer9.get_layers()
        self.trainable_layers[-1].next_layers = [transformer9_layers[0]]
        self.trainable_layers += transformer9_layers
        linear_5 = DensePCNLayer(4096, learning_rate, 'linear', self.trainable_layers[-1])
        self.trainable_layers[-1].next_layers = [linear_5]
        self.trainable_layers.append(linear_5)
        tp5 = TransposePCNLayer(linear_5)
        linear_5.next_layers = [tp5]
        linear_6 = DensePCNLayer(3, learning_rate, 'linear', tp5)
        tp5.next_layers = [linear_6]
        self.trainable_layers.append(linear_6)
        tp6 = TransposePCNLayer(linear_6)
        linear_6.next_layers = [tp6]
        # new_mask_bool_3 = (new_mask_bool_12 @ tf.abs(linear_6.wts)) > 0
        # new_mask_3 = tf.where(new_mask_bool_3, -1e9, 0.0)
        transformer10 = TransformerPCNLayer(3, 4096, 8, learning_rate, tp6)
        transformer10_layers = transformer10.get_layers()
        tp6.next_layers = [transformer10_layers[0]]
        self.trainable_layers+=transformer10_layers
        transformer11 = TransformerPCNLayer(3, 4096, 8, learning_rate, self.trainable_layers[-1])
        transformer11_layers = transformer11.get_layers()
        self.trainable_layers[-1].next_layers = [transformer11_layers[0]]
        self.trainable_layers += transformer11_layers
        transformer12 = TransformerPCNLayer(3, 4096, 8, learning_rate, self.trainable_layers[-1])
        transformer12_layers = transformer12.get_layers()
        self.trainable_layers[-1].next_layers = [transformer12_layers[0]]
        self.trainable_layers += transformer12_layers
        transformer13 = TransformerPCNLayer(3, 4096, 8, learning_rate, self.trainable_layers[-1])
        transformer13_layers = transformer13.get_layers()
        self.trainable_layers[-1].next_layers = [transformer13_layers[0]]
        self.trainable_layers += transformer13_layers
        transformer14 = TransformerPCNLayer(3, 4096, 8, learning_rate, self.trainable_layers[-1])
        transformer14_layers = transformer14.get_layers()
        self.trainable_layers[-1].next_layers = [transformer14_layers[0]]
        self.trainable_layers += transformer14_layers
        transformer15 = TransformerPCNLayer(3, 4096, 8, learning_rate, self.trainable_layers[-1])
        transformer15_layers = transformer15.get_layers()
        self.trainable_layers[-1].next_layers = [transformer15_layers[0]]
        self.trainable_layers += transformer15_layers
        transformer16 = TransformerPCNLayer(3, 4096, 8, learning_rate, self.trainable_layers[-1])
        transformer16_layers = transformer16.get_layers()
        self.trainable_layers[-1].next_layers = [transformer16_layers[0]]
        self.trainable_layers += transformer16_layers
        transformer17 = TransformerPCNLayer(3, 4096, 8, learning_rate, self.trainable_layers[-1])
        transformer17_layers = transformer17.get_layers()
        self.trainable_layers[-1].next_layers = [transformer17_layers[0]]
        self.trainable_layers += transformer17_layers

        flatten2 = FlattenPCNLayer(self.trainable_layers[-1])
        self.trainable_layers[-1].next_layers = [flatten2]
        inter11 = DensePCNLayer(100, learning_rate, 'linear', flatten2)
        flatten2.next_layers = [inter11]
        self.trainable_layers.append(inter11)
        dense3 = DensePCNLayer(36864, learning_rate, 'relu', inter11)
        inter11.next_layers = [dense3]
        self.trainable_layers.append(dense3)
        inter12 = DensePCNLayer(100, learning_rate, 'linear', dense3)
        dense3.next_layers = [inter12]
        self.trainable_layers.append(inter12)
        dense4 = DensePCNLayer(102400, learning_rate, 'relu', inter12, share_state_layer=dense2)
        inter12.next_layers = [dense4]
        self.trainable_layers.append(dense4)

        flatten4 = FlattenPCNLayer(transformer13_layers[-1])
        transformer13_layers[-1].next_layers.append(flatten4)
        inter13 = DensePCNLayer(100, learning_rate, 'linear', flatten4)
        flatten4.next_layers = [inter13]
        self.trainable_layers.append(inter13)
        dense7 = DensePCNLayer(44237, learning_rate, 'relu', inter13)
        inter13.next_layers = [dense7]
        self.trainable_layers.append(dense7)
        inter14 = DensePCNLayer(100, learning_rate, 'linear', dense7)
        dense7.next_layers = [inter14]
        self.trainable_layers.append(inter14)
        dense8 = DensePCNLayer(161817, learning_rate, 'linear', inter14, share_state_layer=dense6)
        inter14.next_layers = [dense8]
        self.trainable_layers.append(dense8)

        flatten6 = FlattenPCNLayer(transformer9_layers[-1])
        transformer9_layers[-1].next_layers.append(flatten6)
        inter15 = DensePCNLayer(100, learning_rate, 'linear', flatten6)
        flatten6.next_layers = [inter15]
        self.trainable_layers.append(inter15)
        dense11 = DensePCNLayer(90931, learning_rate, 'relu', inter15)
        inter15.next_layers = [dense11]
        self.trainable_layers.append(dense11)
        inter16 = DensePCNLayer(100, learning_rate, 'linear', dense11)
        dense11.next_layers = [inter16]
        self.trainable_layers.append(inter16)
        dense12 = DensePCNLayer(345871, learning_rate, 'linear', inter16, share_state_layer=dense10)
        inter16.next_layers = [dense12]
        self.trainable_layers.append(dense12)

        flatten8 = FlattenPCNLayer(transformer6_layers[-1])
        transformer6_layers[-1].next_layers.append(flatten8)
        inter17 = DensePCNLayer(100, learning_rate, 'linear', flatten8)
        flatten8.next_layers = [inter17]
        self.trainable_layers.append(inter17)
        dense15 = DensePCNLayer(185795, learning_rate, 'relu', inter17)
        inter17.next_layers = [dense15]
        self.trainable_layers.append(dense15)
        inter18 = DensePCNLayer(100, learning_rate, 'linear', dense15)
        dense15.next_layers = [inter18]
        self.trainable_layers.append(inter18)
        dense16 = DensePCNLayer(702332, learning_rate, 'linear', inter18, share_state_layer=dense14)
        inter18.next_layers = [dense16]
        self.trainable_layers.append(dense16)

        flatten10 = FlattenPCNLayer(transformer3_layers[-1])
        transformer3_layers[-1].next_layers.append(flatten10)
        inter19 = DensePCNLayer(100, learning_rate, 'linear', flatten10)
        flatten10.next_layers = [inter19]
        self.trainable_layers.append(inter19)
        dense19 = DensePCNLayer(373555, learning_rate, 'relu', inter19)
        inter19.next_layers = [dense19]
        self.trainable_layers.append(dense19)
        inter20 = DensePCNLayer(100, learning_rate, 'linear', dense19)
        dense19.next_layers = [inter20]
        self.trainable_layers.append(inter20)
        dense20 = DensePCNLayer(1429912, learning_rate, 'linear', inter20, share_state_layer=dense18)
        inter20.next_layers = [dense20]
        self.trainable_layers.append(dense20)

   
    def pass_next(self, prev_layer, layer, mask=None):
        if hasattr(layer, 'prev_layers') and len(layer.prev_layers)==2:
            new_output = layer(layer.prev_layers[0].predict_next(), layer.prev_layers[1].predict_next())
        elif isinstance(layer, AttentionPCNLayer):
            new_output = layer(prev_layer.predict_next(), mask=mask)
        else:
            if hasattr(layer, 'state'):
                new_output = layer(prev_layer.predict_next(), set_state=True)
                if mask is not None and isinstance(layer, DensePCNLayer) and (layer.num_units == 48 or layer.num_units == 12 or layer.num_units == 3):
                    mask = tf.where(
                        (tf.cast(mask == 0, layer.wts.dtype) @ tf.abs(layer.wts)) == 0,
                        tf.constant(-1e9, dtype=layer.wts.dtype),
                        tf.constant(0.0, dtype=layer.wts.dtype)
                    )
            else:
                new_output = layer(prev_layer.predict_next())
        if layer.next_layers != []:
            for next_layer in layer.next_layers:
                self.pass_next(layer, next_layer, mask)
        else:
            print(new_output.shape)


    def pass_through(self, img_tensor:tf.Tensor, txt_tensor:tf.Tensor, mask:tf.Tensor=None):
        if mask is None:
            mask = tf.zeros((txt_tensor.shape[0], txt_tensor.shape[1]), dtype=tf.float32)
        self.img_input.set_state(img_tensor)
        self.pass_next(self.img_input, self.img_input.next_layers[0])
        self.txt_input.set_state(txt_tensor)
        self.pass_next(self.txt_input, self.txt_input.next_layers[0], mask)

    
    def update_states_wts_b(self, num_steps:int):
        for step in tf.range(num_steps):
            for layer in self.trainable_layers:
                layer.update_state()
                layer.update_wts()
                layer.update_b()
                gc.collect()

    def update_states_wts_b_relaxed(self, num_weight_steps:int, num_relax_steps:int):
        # Predictive-coding training schedule. With inputs clamped, first relax the
        # latent STATES toward equilibrium under fixed weights, THEN take a single
        # weight/bias step from those relaxed states. This only re-sequences the
        # existing per-layer update methods; it does not change any per-layer
        # update_state / update_wts / update_b math.
        for weight_step in range(num_weight_steps):
            # RELAX: iterate ONLY state updates, weights fixed
            for _ in range(num_relax_steps):
                for layer in self.trainable_layers:
                    layer.update_state()
            # LEARN: one weight + bias step using the relaxed states
            for layer in self.trainable_layers:
                layer.update_wts()
                layer.update_b()
            gc.collect()


    def train_step(self, num_steps:int, img_tensor:tf.Tensor, txt_tensor:tf.Tensor, mask:tf.Tensor=None):
        self.img_input.is_clamped = True
        self.txt_input.is_clamped = True
        self.pass_through(img_tensor, txt_tensor, mask)
        self.update_states_wts_b(num_steps)
        
    
    def update_states_img(self, num_steps: int):
        for step in range(num_steps):
            for layer in self.trainable_layers:
                layer.update_state()
            self.img_input.update_state()

    
    def update_states_txt(self, num_steps: int):
        for step in range(num_steps):
            for layer in self.trainable_layers:
                layer.update_state()
            self.txt_input.update_state()

    def test_step(self, num_steps:int, img_tensor:tf.Tensor, txt_tensor:tf.Tensor, predict:Literal['img', 'txt'], mask:tf.Tensor=None):
        self.pass_through(img_tensor, txt_tensor, mask)
        if predict == 'img':
            self.img_input.is_clamped = False
            self.txt_input.is_clamped = True
            self.update_states_img(num_steps)
            return self.img_input.predict_next()
        elif predict == 'txt':
            self.img_input.is_clamped = True
            self.txt_input.is_clamped = False
            self.update_states_txt(num_steps)
            return self.txt_input.predict_next()



        



        



