import tensorflow as tf
from typing import Literal
class Conv2DPCNLayer:
    is_clamped : tf.Variable # bool
    fix_wts_b : tf.Variable # bool
    num_units : int
    prev_layer: object
    next_layers: list
    wts : tf.Variable # tf.Tensor
    output_shape : tuple
    kernel_size: tuple[int, int]
    activation : str
    state : tf.Variable # tf.Tensor
    learning_rate:float
    padding : str

    def __init__(self, num_units:int, kernel_size:tuple[int, int], learning_rate:float, activation:Literal['linear', 'relu']='linear', prev_layer:object=None, next_layers:list=None, padding:str='VALID', stride:int=1):
        self.is_clamped = False
        self.fix_wts_b = False
        self.num_units = num_units
        self.prev_layer = prev_layer
        self.next_layers = [] if next_layers is None else next_layers
        self.wts = None
        self.output_shape = None
        self.state = None
        self.activation = activation
        self.learning_rate = learning_rate
        self.state_lr = learning_rate   # inference/relaxation rate (decoupled from weight lr)
        self.bias_lr = learning_rate    # kept for uniform driver setup (conv has no bias)
        self.state_clip = float('inf')  # max |state| element magnitude after relaxation; inf = off
        self.weight_decay = 0.0            # LARS beta term; 0 = current beta-less behavior
        self.trust_cap = float("inf")      # cap the trust ratio; inf = off
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.input_shape = None   # pre-downsample spatial shape, recorded at forward time (for the strided transpose)
        self.weight_norm = False          # plain Python bool: tf.function branches resolve at trace time
        self.g_mag = None                 # per-output-filter magnitude, created by enable_weight_norm
        self.hf_gamma = 0.0               # high-frequency boost on the bottom pixel error; 0 = off
        self.noise_temp = 0.0             # Langevin noise temperature for sampling; 0 = off (deterministic)

    def init_params(self, input_shape:tuple):
        # print(self.get_kaiming_gain()/tf.sqrt(float(input_shape[-1])))
        self.wts = tf.Variable(tf.random.normal((*self.kernel_size, input_shape[-1], self.num_units), dtype=tf.float32,
                            stddev=tf.cast(self.get_kaiming_gain()/tf.sqrt(float(self.kernel_size[0]*self.kernel_size[1]*input_shape[-1])), tf.float32)), trainable=False)
    
    def weight(self):
        # The effective weight used identically in predict_next (encode) and predict_prev
        # (decode). Off = self.wts (byte-identical). On = per-output-filter magnitude g_mag
        # times the unit direction wts/||wts||, normalized over the (kh, kw, in) axes.
        if not self.weight_norm:
            return self.wts
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.wts), axis=[0, 1, 2], keepdims=True)) + 1e-8
        return tf.reshape(self.g_mag, (1, 1, 1, -1)) * self.wts / norm

    def enable_weight_norm(self):
        # Seamless enable: g_mag = per-filter ||wts||, so weight() == wts at enable time.
        if self.wts is None:
            raise RuntimeError("realize weights (run a forward pass) before enabling weight_norm")
        norm = tf.sqrt(tf.reduce_sum(tf.square(self.wts), axis=[0, 1, 2]))   # (O,)
        self.g_mag = tf.Variable(norm, trainable=False)
        self.weight_norm = True

    def predict_prev(self):
        # The decode transpose must reconstruct net_in's INPUT shape. Under VALID
        # the forward conv shrank the map by (kernel-1), so the transpose expands
        # it back by (kernel-1). Under SAME the forward conv preserved spatial
        # size, so the SAME transpose must also preserve it (no +kernel-1), or the
        # PC relaxation shapes mismatch.
        if self.stride == 1:
            if self.padding == 'SAME':
                output_shape = (self.output_shape[0], self.output_shape[1], self.output_shape[2], self.wts.shape[-2])
            else:
                output_shape = (self.output_shape[0], self.output_shape[1]+self.kernel_size[0]-1, self.output_shape[2]+self.kernel_size[1]-1, self.wts.shape[-2])
            return tf.nn.conv2d_transpose(self.state, self.weight(), padding=self.padding, strides=1, output_shape=output_shape)
        else:
            output_shape = (self.output_shape[0], self.input_shape[1], self.input_shape[2], self.wts.shape[-2])
            return tf.nn.conv2d_transpose(self.state, self.weight(), padding=self.padding, strides=self.stride, output_shape=output_shape)
    
    def predict_next(self):
        return self.state
    
    def pred_loss_d_input(self, x:tf.Tensor):
        if self.activation == 'relu':
            return tf.nn.conv2d_transpose(-(self.predict_next()-self(x))*self.d_gelu(self.net_in(x)), self.weight(), strides=self.stride, padding=self.padding, output_shape=x.shape)
        elif self.activation == 'gelu':
            return tf.nn.conv2d_transpose(-(self.predict_next()-self(x))*self.d_gelu(self.net_in(x)), self.weight(), strides=self.stride, padding=self.padding, output_shape=x.shape)
        elif self.activation == 'silu':
            return tf.nn.conv2d_transpose(-(self.predict_next()-self(x))*self.d_silu(self.net_in(x)), self.weight(), strides=self.stride, padding=self.padding, output_shape=x.shape)
        else:
            return tf.nn.conv2d_transpose(-(self.predict_next()-self(x)), self.weight(), strides=self.stride, padding=self.padding, output_shape=x.shape)

    def d_gelu(self, x:tf.Tensor):
        return 0.5*(1+tf.math.erf(x/tf.sqrt(2.))) + x/tf.sqrt(2*tf.acos(-1.))*tf.exp(-tf.square(x)/2)

    def d_silu(self, x:tf.Tensor):
        s = tf.sigmoid(x)
        return s * (1.0 + x * (1.0 - s))     # derivative of silu(x)=x*sigmoid(x)

    # 1/2*(next-pred)^2
    # => 
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
                elif layer.activation == 'gelu':
                    state = tf.nn.gelu(state)
                    pred_state = tf.nn.gelu(pred_state)
                elif layer.activation == 'silu':
                    state = tf.nn.silu(state)
                    pred_state = tf.nn.silu(pred_state)
                average_d_pred += tf.cast(layer.pred_loss_d_input(self.predict_next()), tf.float32)
                average_d_state += (state - pred_state)
            if num_next_layers!=0:
                self.state.assign_sub(self.state_lr * ((average_d_pred+average_d_state)/(2.*float(num_next_layers))))
            # pred prev layer & pred from prev layer
            if self.prev_layer is not None:
                d_pred = tf.zeros_like(self.state)
                d_state = tf.zeros_like(self.state)
                layer = self.prev_layer
                if self.activation == 'relu':
                    multiplier = 1.0 + tf.cast(layer.is_clamped, tf.float32)
                    d_pred += tf.nn.conv2d(
                        -multiplier*(tf.nn.relu(layer.predict_next()) - tf.nn.relu(self.predict_prev())),
                        self.weight(), strides=self.stride, padding=self.padding)
                elif self.activation == 'gelu':
                    multiplier = 1.0 + tf.cast(layer.is_clamped, tf.float32)
                    d_pred += tf.nn.conv2d(-multiplier*(tf.nn.gelu(layer.predict_next()) - tf.nn.gelu(self.predict_prev())), self.weight(), strides=self.stride, padding=self.padding)
                elif self.activation == 'silu':
                    multiplier = 1.0 + tf.cast(layer.is_clamped, tf.float32)
                    d_pred += tf.nn.conv2d(-multiplier*(tf.nn.silu(layer.predict_next()) - tf.nn.silu(self.predict_prev())), self.weight(), strides=self.stride, padding=self.padding)
                else:
                    multiplier = 1.0 + tf.cast(layer.is_clamped, tf.float32)
                    d_pred += tf.nn.conv2d(
                        -multiplier*(layer.predict_next() - self.predict_prev()),
                        self.weight(), strides=self.stride, padding=self.padding)
                if not layer.is_clamped:
                    d_state += (self.predict_next() - self(layer.predict_next()))
                self.state.assign_sub(self.state_lr * ((d_pred+d_state)/2.))
            if self.noise_temp > 0.0:
                # Langevin noise: turns the deterministic relaxation into sampling of the energy.
                self.state.assign_add(tf.sqrt(2.0*self.state_lr*self.noise_temp) * tf.random.normal(self.state.shape))
            if self.state_clip != float('inf'):
                self.state.assign(tf.clip_by_value(self.state, -self.state_clip, self.state_clip))

    def _hf(self, e):
        # High-frequency boost of a bottom prediction error: e + hf_gamma * Laplacian(e), a
        # fixed depthwise high-pass so this stays the layer's own local error (no backprop).
        # hf_gamma == 0 returns e unchanged (byte-identical). Built fresh each call (a tiny
        # graph constant), so it is safe inside the compiled relaxation sweep.
        if self.hf_gamma == 0.0:
            return e
        c = int(e.shape[-1])
        lap = tf.constant([[0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]], tf.float32)
        kernel = tf.reshape(lap, (3, 3, 1, 1)) * tf.ones((1, 1, c, 1), tf.float32)
        # reflect-pad then VALID, so a constant field maps to exactly 0 (a true high-pass with
        # no zero-padding border ring at the literal image edge), preserving the spatial shape.
        ep = tf.pad(e, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="REFLECT")
        hp = tf.nn.depthwise_conv2d(ep, kernel, strides=[1, 1, 1, 1], padding="VALID")
        return e + self.hf_gamma * hp

    # 1/2*(gelu(conv(prev.state, self.wts))-self.state)^2
    # => (gelu(conv(prev.state, self.wts))-self.state) * d_gelu(conv(prev.state, self.wts)) * conv2dbackprop
    #          (B, H2, W2, C2)                             (B, H2, W2, C2)                    (Fx, Fy, C1, C2)
    # 1/2*(self.predict_prev - prev.state)^2
    # => (self.predict_prev - prev.state) *
    def update_wts(self):
        d_state = tf.zeros_like(self.wts)
        d_pred = tf.zeros_like(self.wts)
        if not self.fix_wts_b and self.prev_layer is not None:
            if not self.prev_layer.is_clamped:
                pred = self(self.prev_layer.predict_next())
                eps = pred - self.predict_next()
                if self.activation == 'relu':
                    d_state += tf.raw_ops.Conv2DBackpropFilter(input=self.prev_layer.predict_next(), filter_sizes=self.wts.shape, out_backprop=eps*self.d_gelu(pred), strides=[1, self.stride, self.stride, 1], padding=self.padding)
                elif self.activation == 'gelu':
                    d_state += tf.raw_ops.Conv2DBackpropFilter(input=self.prev_layer.predict_next(), filter_sizes=self.wts.shape, out_backprop=eps*self.d_gelu(pred), strides=[1, self.stride, self.stride, 1], padding=self.padding)
                elif self.activation == 'silu':
                    d_state += tf.raw_ops.Conv2DBackpropFilter(input=self.prev_layer.predict_next(), filter_sizes=self.wts.shape, out_backprop=eps*self.d_silu(pred), strides=[1, self.stride, self.stride, 1], padding=self.padding)
                else:
                    d_state += tf.raw_ops.Conv2DBackpropFilter(input=self.prev_layer.predict_next(), filter_sizes=self.wts.shape, out_backprop=eps, strides=[1, self.stride, self.stride, 1], padding=self.padding)
            if not self.is_clamped:
                pred = self.predict_prev()
                if self.activation == 'relu':
                    d_pred += tf.raw_ops.Conv2DBackpropFilter(input=self._hf(tf.nn.relu(pred)-tf.nn.relu(self.prev_layer.predict_next())), filter_sizes=self.wts.shape, out_backprop=self.predict_next(), strides=[1, self.stride, self.stride, 1], padding=self.padding)
                elif self.activation == 'gelu':
                    d_pred += tf.raw_ops.Conv2DBackpropFilter(input=self._hf(tf.nn.gelu(pred)-tf.nn.gelu(self.prev_layer.predict_next())), filter_sizes=self.wts.shape, out_backprop=self.predict_next(), strides=[1, self.stride, self.stride, 1], padding=self.padding)
                elif self.activation == 'silu':
                    d_pred += tf.raw_ops.Conv2DBackpropFilter(input=self._hf(tf.nn.silu(pred)-tf.nn.silu(self.prev_layer.predict_next())), filter_sizes=self.wts.shape, out_backprop=self.predict_next(), strides=[1, self.stride, self.stride, 1], padding=self.padding)
                else:
                    d_pred += tf.raw_ops.Conv2DBackpropFilter(input=self._hf(pred-self.prev_layer.predict_next()), filter_sizes=self.wts.shape, out_backprop=self.predict_next(), strides=[1, self.stride, self.stride, 1], padding=self.padding)
            if not self.is_clamped or not self.prev_layer.is_clamped:
                denom = (tf.cast(tf.logical_not(self.is_clamped), tf.float32) + tf.cast(tf.logical_not(self.prev_layer.is_clamped), tf.float32))
                g = (d_state + d_pred) / denom
                wd = self.weight_decay
                if self.weight_norm:
                    # Split the local gradient g (w.r.t. the effective weight) into a radial
                    # magnitude update and a tangential direction update, per output filter.
                    # ||w|| = |g_mag| stays bounded (damped by wd); ||wts|| ~preserved (dv ⊥ vhat).
                    norm = tf.sqrt(tf.reduce_sum(tf.square(self.wts), axis=[0, 1, 2], keepdims=True)) + 1e-8
                    vhat = self.wts / norm
                    dg = tf.reduce_sum(g * vhat, axis=[0, 1, 2])                 # (O,)
                    dv = (tf.reshape(self.g_mag, (1, 1, 1, -1)) / norm) * (g - tf.reshape(dg, (1, 1, 1, -1)) * vhat)
                    # Trust-normalize the step (LARS-style) so a single lr works across this
                    # model's very wide fan-in. Trust uses ||wts||=||v||, which the tangential
                    # split PRESERVES, so there is no norm-inflation feedback (the runaway ||w||
                    # growth is already killed by diverting the radial part into the damped g_mag).
                    wn = tf.norm(self.wts)
                    trust = wn / (tf.norm(g) + wd * wn + 1e-6)
                    trust = tf.minimum(trust, self.trust_cap)
                    self.g_mag.assign_sub(self.learning_rate * trust * (dg + wd * self.g_mag))
                    self.wts.assign_sub(self.learning_rate * trust * dv)
                else:
                    wn = tf.norm(self.wts)
                    trust = wn / (tf.norm(g) + wd * wn + 1e-6)
                    trust = tf.minimum(trust, self.trust_cap)
                    self.last_trust = trust  # exposed for logging only
                    self.wts.assign_sub(self.learning_rate * trust * (g + wd * self.wts))

    def update_b(self):
        pass # there is no bias

        
    def init_state(self):
        self.state = None

    def init_wts_b(self):
        self.wts = None
    
    def net_in(self, x:tf.Tensor):
        if self.wts is None:
            self.init_params(x.shape)
        self.input_shape = x.shape
        return tf.nn.conv2d(x, self.weight(), padding=self.padding, strides=self.stride)

    def __call__(self, x : tf.Tensor, set_state:bool = False):
        net_in = self.net_in(x)

        if self.activation == 'relu':
            net_act = tf.nn.relu(net_in)
        elif self.activation == 'gelu':
            net_act = tf.nn.gelu(net_in)
        elif self.activation == 'silu':
            net_act = tf.nn.silu(net_in)
        else:
            net_act = net_in

        if set_state:
            if self.state is None:
                self.state = tf.Variable(net_act, trainable=False)
            else:
                self.state.assign(net_act)
            self.output_shape = net_act.shape

        return net_act

        

    def get_kaiming_gain(self):
        if self.activation == 'relu':
            return tf.sqrt(2.)
        else:
            return 1


class MaxPool2DPCNLayer:
    is_clamped : tf.Variable # bool
    fix_wts_b : tf.Variable # bool
    prev_layer : object
    next_layers : list
    kernel_size : tuple[int, int]
    output_shape : tuple
    def __init__(self, kernel_size: tuple[int, int],  prev_layer:object, next_layers:list=None):
        self.is_clamped = True
        self.fix_wts_b = True
        self.prev_layer = prev_layer
        self.next_layers = [] if next_layers is None else next_layers
        self.kernel_size = kernel_size
        self.output_shape = None

    def __call__(self, x:tf.Tensor):
        out = tf.nn.max_pool2d(x, self.kernel_size, self.kernel_size[0], 'VALID')
        self.output_shape = out.shape
        return out
    
    def predict_next(self):
        return self(self.prev_layer.predict_next())