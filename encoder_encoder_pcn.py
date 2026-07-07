import tensorflow as tf
from conv_pcn_layer import Conv2DPCNLayer, MaxPool2DPCNLayer
from dense_pcn_layer import DensePCNLayer
from transformer_pcn_layer import TransformerPCNLayer, PositionalEncodingLayer, AttentionPCNLayer, AddNormalizePCNLayer
from typing import Literal
from pcn_config import PCNConfig, NATIVE_7B
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
    config : PCNConfig
    img_input : object
    txt_input : object
    def __init__(self, learning_rate : float, config : PCNConfig = NATIVE_7B):
        self.trainable_layers = []
        self.learning_rate = learning_rate
        self.config = config
        # Lazily-built graph-compiled PC sweep (see update_states_wts_b) plus its
        # trace counter. The sweep is compiled once per clamp configuration; the
        # train path (update_states_wts_b) clamps BOTH inputs, so exactly one
        # trace is expected across the whole run.
        self._compiled_sweep = None
        self._sweep_trace_count = 0
        # Lazily-built graph-compiled sweeps for the relaxed (relax-then-step)
        # schedule (see update_states_wts_b_relaxed). Two separate no-arg
        # closures: one that only updates STATES (relax) and one that takes a
        # single weight+bias step (learn). Each is compiled once per clamp
        # configuration; the train path clamps BOTH inputs, so exactly one
        # trace per sweep (2 total) is expected across the whole run.
        self._compiled_relax_sweep = None
        self._compiled_learn_sweep = None
        self._relax_sweep_trace_count = 0
        self._learn_sweep_trace_count = 0
        self.img_input = InputPCNLayer(learning_rate)
        self.trainable_layers.append(self.img_input)
        conv1 = Conv2DPCNLayer(config.conv_channels[0], (3, 3), learning_rate, 'relu', self.img_input, padding=config.conv_padding)
        self.trainable_layers.append(conv1)
        self.img_input.next_layers = [conv1]
        conv2 = Conv2DPCNLayer(config.conv_channels[1], (3, 3), learning_rate, 'relu', conv1, padding=config.conv_padding)
        self.trainable_layers.append(conv2)
        conv1.next_layers = [conv2]
        mp1 = MaxPool2DPCNLayer((2, 2), conv2)
        conv2.next_layers = [mp1]
        conv3 = Conv2DPCNLayer(config.conv_channels[2], (3, 3), learning_rate, 'relu', mp1, padding=config.conv_padding)
        self.trainable_layers.append(conv3)
        mp1.next_layers = [conv3]
        conv4 = Conv2DPCNLayer(config.conv_channels[3], (3, 3), learning_rate, 'relu', conv3, padding=config.conv_padding)
        self.trainable_layers.append(conv4)
        conv3.next_layers = [conv4]
        mp2 = MaxPool2DPCNLayer((2, 2), conv4)
        conv4.next_layers = [mp2]
        conv5 = Conv2DPCNLayer(config.conv_channels[4], (3, 3), learning_rate, 'relu', mp2, padding=config.conv_padding)
        self.trainable_layers.append(conv5)
        mp2.next_layers = [conv5]
        conv6 = Conv2DPCNLayer(config.conv_channels[5], (3, 3), learning_rate, 'relu', conv5, padding=config.conv_padding)
        self.trainable_layers.append(conv6)
        conv5.next_layers = [conv6]
        mp3 = MaxPool2DPCNLayer((2, 2), conv6)
        conv6.next_layers = [mp3]
        conv7 = Conv2DPCNLayer(config.conv_channels[6], (3, 3), learning_rate, 'relu', mp3, padding=config.conv_padding)
        self.trainable_layers.append(conv7)
        mp3.next_layers = [conv7]
        conv8 = Conv2DPCNLayer(config.conv_channels[7], (3, 3), learning_rate, 'relu', conv7, padding=config.conv_padding)
        self.trainable_layers.append(conv8)
        conv7.next_layers = [conv8]
        mp4 = MaxPool2DPCNLayer((2, 2), conv8)
        conv8.next_layers = [mp4]
        conv9 = Conv2DPCNLayer(config.conv_channels[8], (3, 3), learning_rate, 'relu', mp4, padding=config.conv_padding)
        self.trainable_layers.append(conv9)
        mp4.next_layers = [conv9]

        flatten1 = FlattenPCNLayer(conv9)
        conv9.next_layers = [flatten1]
        inter1 = DensePCNLayer(config.inter_dim, learning_rate, 'linear', flatten1)
        flatten1.next_layers = [inter1]
        self.trainable_layers.append(inter1)
        dense1 = DensePCNLayer(config.img_dense_relu_widths[0], learning_rate, 'relu', inter1)
        inter1.next_layers = [dense1]
        self.trainable_layers.append(dense1)
        inter2 = DensePCNLayer(config.inter_dim, learning_rate, 'linear', dense1)
        dense1.next_layers = [inter2]
        self.trainable_layers.append(inter2)
        dense2 = DensePCNLayer(config.shared_latent_dims[0], learning_rate, 'linear', inter2)
        inter2.next_layers = [dense2]
        self.trainable_layers.append(dense2)

        flatten3 = FlattenPCNLayer(conv8)
        conv8.next_layers.append(flatten3)
        inter3 = DensePCNLayer(config.inter_dim, learning_rate, 'linear', flatten3)
        flatten3.next_layers = [inter3]
        self.trainable_layers.append(inter3)
        dense5 = DensePCNLayer(config.img_dense_relu_widths[1], learning_rate, 'relu', inter3)
        inter3.next_layers = [dense5]
        self.trainable_layers.append(dense5)
        inter4 = DensePCNLayer(config.inter_dim, learning_rate, 'linear', dense5)
        dense5.next_layers = [inter4]
        self.trainable_layers.append(inter4)
        dense6 = DensePCNLayer(config.shared_latent_dims[1], learning_rate, 'linear', inter4)
        inter4.next_layers = [dense6]
        self.trainable_layers.append(dense6)

        flatten5 = FlattenPCNLayer(conv6)
        conv6.next_layers.append(flatten5)
        inter5 = DensePCNLayer(config.inter_dim, learning_rate, 'linear', flatten5)
        flatten5.next_layers = [inter5]
        self.trainable_layers.append(inter5)
        dense9 = DensePCNLayer(config.img_dense_relu_widths[2], learning_rate, 'relu', inter5)
        inter5.next_layers = [dense9]
        self.trainable_layers.append(dense9)
        inter6 = DensePCNLayer(config.inter_dim, learning_rate, 'linear', dense9)
        dense9.next_layers = [inter6]
        self.trainable_layers.append(inter6)
        dense10 = DensePCNLayer(config.shared_latent_dims[2], learning_rate, 'linear', inter6)
        inter6.next_layers = [dense10]
        self.trainable_layers.append(dense10)

        flatten7 = FlattenPCNLayer(conv4)
        conv4.next_layers.append(flatten7)
        inter7 = DensePCNLayer(config.inter_dim, learning_rate, 'linear', flatten7)
        flatten7.next_layers = [inter7]
        self.trainable_layers.append(inter7)
        dense13 = DensePCNLayer(config.img_dense_relu_widths[3], learning_rate, 'relu', inter7)
        inter7.next_layers = [dense13]
        self.trainable_layers.append(dense13)
        inter8 = DensePCNLayer(config.inter_dim, learning_rate, 'linear', dense13)
        dense13.next_layers = [inter8]
        self.trainable_layers.append(inter8)
        dense14 = DensePCNLayer(config.shared_latent_dims[3], learning_rate, 'linear', inter8)
        inter8.next_layers = [dense14]
        self.trainable_layers.append(dense14)

        flatten9 = FlattenPCNLayer(conv2)
        conv2.next_layers.append(flatten9)
        inter9 = DensePCNLayer(config.inter_dim, learning_rate, 'linear', flatten9)
        flatten9.next_layers = [inter9]
        self.trainable_layers.append(inter9)
        dense17 = DensePCNLayer(config.img_dense_relu_widths[4], learning_rate, 'relu', inter9)
        inter9.next_layers = [dense17]
        self.trainable_layers.append(dense17)
        inter10 = DensePCNLayer(config.inter_dim, learning_rate, 'linear', dense17)
        dense17.next_layers = [inter10]
        self.trainable_layers.append(inter10)
        dense18 = DensePCNLayer(config.shared_latent_dims[4], learning_rate, 'linear', inter10)
        inter10.next_layers = [dense18]
        self.trainable_layers.append(dense18)



        self.txt_input = InputPCNLayer(learning_rate)
        self.trainable_layers.append(self.txt_input)
        txt_embedding = DensePCNLayer(config.txt_group_widths[0], learning_rate, 'linear', self.txt_input)
        self.trainable_layers.append(txt_embedding)
        self.txt_input.next_layers = [txt_embedding]
        pos_encoding = PositionalEncodingLayer(config.txt_group_widths[0], txt_embedding)
        txt_embedding.next_layers = [pos_encoding]

        # Transformer trunk as four groups. Group g has config.txt_group_blocks[g]
        # TransformerPCNLayer blocks of width config.txt_group_widths[g], each with
        # config.txt_heads heads and config.txt_sublayers feed-forward sublayers.
        # Between consecutive groups a bridge widens to the next group's width
        # (a linear + transpose) then reduces the sequence length to
        # config.txt_bridge_seq_lens[g] (a linear + transpose). This reproduces the
        # hand-built 3+3+3+8 trunk exactly: same block sequence, same append order
        # into trainable_layers, same next_layers wiring (assignment; the group's
        # last block gets its .next_layers set by the following bridge, or stays
        # empty for the final group so the flatten2 tap assigns it below).
        txt_transformers = []          # every block's get_layers(), in build order
        group_input = pos_encoding     # feeds the first block of the current group
        num_groups = len(config.txt_group_widths)
        for g in range(num_groups):
            width = config.txt_group_widths[g]
            for b in range(config.txt_group_blocks[g]):
                prev = group_input if b == 0 else self.trainable_layers[-1]
                block_layers = TransformerPCNLayer(
                    config.txt_sublayers, width, config.txt_heads, learning_rate, prev
                ).get_layers()
                prev.next_layers = [block_layers[0]]
                self.trainable_layers += block_layers
                txt_transformers.append(block_layers)
            if g < num_groups - 1:
                linear_up = DensePCNLayer(config.txt_group_widths[g + 1], learning_rate, 'linear', self.trainable_layers[-1])
                self.trainable_layers[-1].next_layers = [linear_up]
                self.trainable_layers.append(linear_up)
                tp_up = TransposePCNLayer(linear_up)
                linear_up.next_layers = [tp_up]
                linear_down = DensePCNLayer(config.txt_bridge_seq_lens[g], learning_rate, 'linear', tp_up)
                tp_up.next_layers = [linear_down]
                self.trainable_layers.append(linear_down)
                tp_down = TransposePCNLayer(linear_down)
                linear_down.next_layers = [tp_down]
                group_input = tp_down

        # Text-tap attachment blocks, in tap order (dense3, dense7, dense11,
        # dense15, dense19). The source block index for each tap is config-driven
        # (config.txt_tap_indices) so trunks of different depth can both tap
        # validly; only the tap SOURCE index and WIDTHS are config-driven. For
        # NATIVE_7B these resolve to the hand-built (-1, 12, 8, 5, 2) attachments:
        # the final block, then transformer #13 (4th block of the final group,
        # NOT a group boundary), then the group-2/1/0 last blocks.
        tap_dense3  = txt_transformers[config.txt_tap_indices[0]]
        tap_dense7  = txt_transformers[config.txt_tap_indices[1]]
        tap_dense11 = txt_transformers[config.txt_tap_indices[2]]
        tap_dense15 = txt_transformers[config.txt_tap_indices[3]]
        tap_dense19 = txt_transformers[config.txt_tap_indices[4]]

        flatten2 = FlattenPCNLayer(tap_dense3[-1])
        tap_dense3[-1].next_layers = [flatten2]
        inter11 = DensePCNLayer(config.inter_dim, learning_rate, 'linear', flatten2)
        flatten2.next_layers = [inter11]
        self.trainable_layers.append(inter11)
        dense3 = DensePCNLayer(config.txt_dense_relu_widths[0], learning_rate, 'relu', inter11)
        inter11.next_layers = [dense3]
        self.trainable_layers.append(dense3)
        inter12 = DensePCNLayer(config.inter_dim, learning_rate, 'linear', dense3)
        dense3.next_layers = [inter12]
        self.trainable_layers.append(inter12)
        dense4 = DensePCNLayer(config.shared_latent_dims[0], learning_rate, 'relu', inter12, share_state_layer=dense2)
        inter12.next_layers = [dense4]
        self.trainable_layers.append(dense4)

        self._infonce_codes = (inter2, inter12)   # deepest-scale image / text branch codes (for optional InfoNCE coupling)

        flatten4 = FlattenPCNLayer(tap_dense7[-1])
        tap_dense7[-1].next_layers.append(flatten4)
        inter13 = DensePCNLayer(config.inter_dim, learning_rate, 'linear', flatten4)
        flatten4.next_layers = [inter13]
        self.trainable_layers.append(inter13)
        dense7 = DensePCNLayer(config.txt_dense_relu_widths[1], learning_rate, 'relu', inter13)
        inter13.next_layers = [dense7]
        self.trainable_layers.append(dense7)
        inter14 = DensePCNLayer(config.inter_dim, learning_rate, 'linear', dense7)
        dense7.next_layers = [inter14]
        self.trainable_layers.append(inter14)
        dense8 = DensePCNLayer(config.shared_latent_dims[1], learning_rate, 'linear', inter14, share_state_layer=dense6)
        inter14.next_layers = [dense8]
        self.trainable_layers.append(dense8)

        flatten6 = FlattenPCNLayer(tap_dense11[-1])
        tap_dense11[-1].next_layers.append(flatten6)
        inter15 = DensePCNLayer(config.inter_dim, learning_rate, 'linear', flatten6)
        flatten6.next_layers = [inter15]
        self.trainable_layers.append(inter15)
        dense11 = DensePCNLayer(config.txt_dense_relu_widths[2], learning_rate, 'relu', inter15)
        inter15.next_layers = [dense11]
        self.trainable_layers.append(dense11)
        inter16 = DensePCNLayer(config.inter_dim, learning_rate, 'linear', dense11)
        dense11.next_layers = [inter16]
        self.trainable_layers.append(inter16)
        dense12 = DensePCNLayer(config.shared_latent_dims[2], learning_rate, 'linear', inter16, share_state_layer=dense10)
        inter16.next_layers = [dense12]
        self.trainable_layers.append(dense12)

        flatten8 = FlattenPCNLayer(tap_dense15[-1])
        tap_dense15[-1].next_layers.append(flatten8)
        inter17 = DensePCNLayer(config.inter_dim, learning_rate, 'linear', flatten8)
        flatten8.next_layers = [inter17]
        self.trainable_layers.append(inter17)
        dense15 = DensePCNLayer(config.txt_dense_relu_widths[3], learning_rate, 'relu', inter17)
        inter17.next_layers = [dense15]
        self.trainable_layers.append(dense15)
        inter18 = DensePCNLayer(config.inter_dim, learning_rate, 'linear', dense15)
        dense15.next_layers = [inter18]
        self.trainable_layers.append(inter18)
        dense16 = DensePCNLayer(config.shared_latent_dims[3], learning_rate, 'linear', inter18, share_state_layer=dense14)
        inter18.next_layers = [dense16]
        self.trainable_layers.append(dense16)

        flatten10 = FlattenPCNLayer(tap_dense19[-1])
        tap_dense19[-1].next_layers.append(flatten10)
        inter19 = DensePCNLayer(config.inter_dim, learning_rate, 'linear', flatten10)
        flatten10.next_layers = [inter19]
        self.trainable_layers.append(inter19)
        dense19 = DensePCNLayer(config.txt_dense_relu_widths[4], learning_rate, 'relu', inter19)
        inter19.next_layers = [dense19]
        self.trainable_layers.append(dense19)
        inter20 = DensePCNLayer(config.inter_dim, learning_rate, 'linear', dense19)
        dense19.next_layers = [inter20]
        self.trainable_layers.append(inter20)
        dense20 = DensePCNLayer(config.shared_latent_dims[4], learning_rate, 'linear', inter20, share_state_layer=dense18)
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
                if mask is not None and isinstance(layer, DensePCNLayer) and (layer.num_units in self.config.txt_bridge_seq_lens):
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
        # Graph-compile the single interleaved PC sweep ONCE, then drive it from a
        # plain Python loop. Previously this eager-dispatched ~143 layer objects
        # (each doing several assign_sub ops) every step. Wrapping ONE full sweep
        # in tf.function unrolls the `for layer in self.trainable_layers` loop into
        # a single graph at first trace (slow once), after which every step is one
        # fast graph execution. The per-layer math is untouched — the same
        # update_state();update_wts();update_b() calls in the same interleaved
        # order — so the relaxed states match the eager golden baseline.
        #
        # No variables are created inside the graph: lazy tf.Variable init (wts/b,
        # state, and share_state_layer aliasing) all happens in pass_through, which
        # runs before this. The clamp flags are Python bools, so per-layer branches
        # (is_clamped / activation / prev_layer) bake at TRACE time; this method is
        # only used with both inputs clamped, hence a single trace.
        if self._compiled_sweep is None:
            @tf.function(reduce_retracing=True)
            def _sweep():
                # Python side effect: runs only while TRACING, so it counts traces.
                self._sweep_trace_count += 1
                print(f"[tf.function] tracing compiled PC sweep "
                      f"(trace #{self._sweep_trace_count})", flush=True)
                for layer in self.trainable_layers:
                    layer.update_state()
                    layer.update_wts()
                    layer.update_b()
            self._compiled_sweep = _sweep
            # Guard state recorded at trace time: the graph above baked in the
            # per-layer is_clamped branches and captured these exact state
            # tf.Variable objects by reference. Record both so a later call
            # with a different clamp config, or with any layer's .state
            # rebuilt (reset to None then re-created in pass_through), is
            # caught below instead of silently mutating stale Variables.
            self._sweep_clamp_sig = tuple(bool(L.is_clamped) for L in self.trainable_layers)
            self._sweep_state_ids = tuple(id(getattr(L, "state", None)) for L in self.trainable_layers)
        # Cheap eager-Python check, outside the tf.function (does not enter
        # the graph, negligible cost, no numerical effect). Catches reuse of
        # this same model instance under a different clamp configuration or
        # after any layer's state Variable was re-created.
        cur_clamp_sig = tuple(bool(L.is_clamped) for L in self.trainable_layers)
        cur_state_ids = tuple(id(getattr(L, "state", None)) for L in self.trainable_layers)
        if cur_clamp_sig != self._sweep_clamp_sig or cur_state_ids != self._sweep_state_ids:
            raise RuntimeError(
                "update_states_wts_b: the compiled sweep was traced under a "
                "different clamp configuration or state Variables than the "
                "current call (either is_clamped changed on a layer, or a "
                "layer's .state was reset/rebuilt since the first trace). "
                "The cached tf.function would silently keep mutating the "
                "stale Variables it captured at trace time. Rebuild the "
                "model or reset self._compiled_sweep (and the recorded "
                "_sweep_clamp_sig / _sweep_state_ids) before calling this "
                "again."
            )
        for _ in range(int(num_steps)):
            self._compiled_sweep()

    def update_states_wts_b_relaxed(self, num_weight_steps:int, num_relax_steps:int):
        # Predictive-coding training schedule. With inputs clamped, first relax the
        # latent STATES toward equilibrium under fixed weights, THEN take a single
        # weight/bias step from those relaxed states. This only re-sequences the
        # existing per-layer update methods; it does not change any per-layer
        # update_state / update_wts / update_b math.
        #
        # Same graph-compile strategy as update_states_wts_b, but the relaxed
        # schedule needs TWO distinct sweeps, so we compile each once and drive
        # them from plain Python loops (num_weight_steps / num_relax_steps stay
        # Python ints, NOT inside the graph):
        #   - _compiled_relax_sweep: state-only pass, weights frozen.
        #   - _compiled_learn_sweep: one weight + bias pass from relaxed states.
        # Each tf.function unrolls its `for layer in self.trainable_layers` loop
        # into one graph at first trace (slow once), after which every sweep is a
        # single fast graph execution. The per-layer math and the relax-then-step
        # order are untouched, so the relaxed states match the eager golden.
        #
        # No variables are created inside either graph: lazy tf.Variable init
        # (wts/b, state, and share_state_layer aliasing) all happens in
        # pass_through, which runs before this. The clamp flags are Python bools,
        # so per-layer branches (is_clamped / activation / prev_layer) bake at
        # TRACE time; this method is only used with both inputs clamped, hence a
        # single trace per sweep.
        if self._compiled_relax_sweep is None or self._compiled_learn_sweep is None:
            @tf.function(reduce_retracing=True)
            def _relax_sweep():
                # Python side effect: runs only while TRACING, so it counts traces.
                self._relax_sweep_trace_count += 1
                print(f"[tf.function] tracing compiled relax sweep "
                      f"(trace #{self._relax_sweep_trace_count})", flush=True)
                for layer in self.trainable_layers:
                    layer.update_state()

            @tf.function(reduce_retracing=True)
            def _learn_sweep():
                # Python side effect: runs only while TRACING, so it counts traces.
                self._learn_sweep_trace_count += 1
                print(f"[tf.function] tracing compiled learn sweep "
                      f"(trace #{self._learn_sweep_trace_count})", flush=True)
                for layer in self.trainable_layers:
                    layer.update_wts()
                    layer.update_b()
            self._compiled_relax_sweep = _relax_sweep
            self._compiled_learn_sweep = _learn_sweep
            # Guard state recorded once at build time: both graphs are traced
            # under the same clamp config and capture these exact state
            # tf.Variable objects by reference. Record both so a later call with
            # a different clamp config, or with any layer's .state rebuilt (reset
            # to None then re-created in pass_through), is caught below instead of
            # silently mutating stale Variables. One record covers both sweeps.
            self._relaxed_clamp_sig = tuple(bool(L.is_clamped) for L in self.trainable_layers)
            self._relaxed_state_ids = tuple(id(getattr(L, "state", None)) for L in self.trainable_layers)
        # Cheap eager-Python check, outside the tf.functions (does not enter the
        # graphs, negligible cost, no numerical effect). Catches reuse of this
        # same model instance under a different clamp configuration or after any
        # layer's state Variable was re-created.
        cur_clamp_sig = tuple(bool(L.is_clamped) for L in self.trainable_layers)
        cur_state_ids = tuple(id(getattr(L, "state", None)) for L in self.trainable_layers)
        if cur_clamp_sig != self._relaxed_clamp_sig or cur_state_ids != self._relaxed_state_ids:
            raise RuntimeError(
                "update_states_wts_b_relaxed: the compiled relax/learn sweeps "
                "were traced under a different clamp configuration or state "
                "Variables than the current call (either is_clamped changed on "
                "a layer, or a layer's .state was reset/rebuilt since the first "
                "trace). The cached tf.functions would silently keep mutating "
                "the stale Variables they captured at trace time. Rebuild the "
                "model or reset self._compiled_relax_sweep / "
                "self._compiled_learn_sweep (and the recorded _relaxed_clamp_sig "
                "/ _relaxed_state_ids) before calling this again."
            )
        for weight_step in range(int(num_weight_steps)):
            # RELAX: iterate ONLY state updates, weights fixed
            for _ in range(int(num_relax_steps)):
                self._compiled_relax_sweep()
            # LEARN: one weight + bias step using the relaxed states
            self._compiled_learn_sweep()


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



        



        



