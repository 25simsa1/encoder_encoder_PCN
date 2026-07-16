import tensorflow as tf
from typing import Literal
class DensePCNLayer:
    is_clamped : tf.Variable # bool
    fix_wts_b : tf.Variable # bool
    num_units : int
    prev_layer: object
    next_layers: list
    wts : tf.Variable # tf.Tensor
    b : tf.Variable # tf.Tensor
    output_shape : tuple
    activation : str
    state : tf.Variable # tf.Tensor
    learning_rate:float
    share_wts_layer : object
    share_state_layer : object
    def __init__(self, num_units:int, learning_rate:float, activation:Literal['linear', 'relu']='linear', prev_layer:object=None, next_layers:list=None, share_state_layer:object=None):
        self.is_clamped = False
        self.fix_wts_b = False
        self.num_units = num_units
        self.prev_layer = prev_layer
        self.next_layers = [] if next_layers is None else next_layers
        self.wts = None
        self.b = None
        self.output_shape = None
        self.state = None
        self.activation = activation
        self.learning_rate = learning_rate
        self.state_lr = learning_rate   # inference/relaxation rate (decoupled from weight lr)
        self.bias_lr = learning_rate    # bias update rate (decoupled from weight lr)
        self.state_clip = float('inf')  # max |state| element magnitude after relaxation; inf = off
        self.weight_decay = 0.0            # LARS beta term; 0 = current beta-less behavior
        self.trust_cap = float("inf")      # cap the trust ratio; inf = off
        self.share_state_layer = share_state_layer
        self.weight_norm = False          # plain Python bool
        self.g_mag = None                 # per-output-unit magnitude, created by enable_weight_norm
        self.noise_temp = 0.0             # Langevin noise temperature for sampling; 0 = off (deterministic)
        self.pi_td = 1.0                  # precision on the next-layers (top-down consistency) drive; 1 = default
        self.pi_bu = 1.0                  # precision on the prev-layer (bottom-up consistency) drive; 1 = default
        self.iso_eta = 0.0                # soft orthogonalization rate (isometry constraint); 0 = off
        self.untied = False               # untied top-down prediction weights; False = tied (byte-identical)
        self.wts_td = None                # the top-down weights, created by enable_untied
        self.td_only = False              # distillation mode: freeze wts, train only wts_td

    def init_params(self, input_shape:tuple):
        # print(self.get_kaiming_gain()/tf.sqrt(float(input_shape[-1])))
        self.wts = tf.Variable(tf.random.normal((input_shape[-1], self.num_units), dtype = tf.float32, 
                            stddev=tf.cast(self.get_kaiming_gain()/tf.sqrt(float(input_shape[-1])), tf.float32)), trainable=False)
        self.b = tf.Variable(tf.zeros(self.num_units, dtype = tf.float32), trainable=False)


    def weight(self):
        # Effective weight, used identically in predict_next (encode) and predict_prev (decode).
        # Off = self.wts. On = per-output-column magnitude g_mag times wts/||wts||, normalized
        # over axis 0 (the input dim); wts is (in, out), g_mag is (out,).
        if not self.weight_norm:
            return self.wts
        norm = tf.norm(self.wts, axis=0, keepdims=True) + 1e-8      # (1, out)
        return self.g_mag[None, :] * self.wts / norm

    def enable_weight_norm(self):
        if self.wts is None:
            raise RuntimeError("realize weights (run a forward pass) before enabling weight_norm")
        norm = tf.norm(self.wts, axis=0)     # (out,)
        self.g_mag = tf.Variable(norm, trainable=False)
        self.weight_norm = True

    def weight_td(self):
        # The top-down prediction weight. Tied (default) = the same weight as bottom-up,
        # byte-identical. Untied = this edge's own wts_td, trained by the d_pred local error.
        if not self.untied:
            return self.weight()
        return self.wts_td

    def enable_untied(self):
        # Seamless untie: wts_td starts as a copy of wts (and c_td at zero), so predict_prev
        # is unchanged at enable time; they then learn under their separate local errors.
        if self.wts is None:
            raise RuntimeError("realize weights (run a forward pass) before enabling untied")
        self.wts_td = tf.Variable(tf.identity(self.wts), trainable=False)
        self.c_td = tf.Variable(tf.zeros(int(self.wts.shape[0]), dtype=tf.float32), trainable=False)
        self.untied = True

    def predict_prev(self):
        p = (self.state - self.b) @ tf.linalg.matrix_transpose(self.weight_td())
        if self.untied:
            p = p + self.c_td   # affine inverse: absorbs the constant per-edge offset (gelu DC, vignette)
        return p
    
    def predict_next(self):
        return self.state
    
    def d_gelu(self, x:tf.Tensor):
        return 0.5*(1+tf.math.erf(x/tf.sqrt(2.))) + x/tf.sqrt(2*tf.acos(-1.))*tf.exp(-tf.square(x)/2)
    
    def pred_loss_d_input(self, x:tf.Tensor):
        if self.activation == 'relu':
            return -(self.predict_next()-self(x))*self.d_gelu(self.net_in(x)) @ tf.linalg.matrix_transpose(self.weight())
        else:
            return -(self.predict_next()-self(x)) @ tf.linalg.matrix_transpose(self.weight())

    # pred_err = state - pred
    # 1/2*(state - pred)^2
    def update_state(self):
        if not self.is_clamped:
            # pred next layer & pred from next layer
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
                self.state.assign_sub(self.state_lr * self.pi_td * ((average_d_pred+average_d_state)/(2.*float(num_next_layers))))

            # pred prev layer & pred from prev layer
            if self.prev_layer is not None:
                d_pred = tf.zeros_like(self.state)
                d_state = tf.zeros_like(self.state)
                layer = self.prev_layer
                if self.activation == 'relu':
                    d_pred += -(1+int(layer.is_clamped))*(tf.nn.relu(layer.predict_next()) - tf.nn.relu(self.predict_prev())) @ self.weight_td()
                else:
                    d_pred += -(1+int(layer.is_clamped))*(layer.predict_next() - self.predict_prev()) @ self.weight_td()
                if not layer.is_clamped:
                    d_state += (self.predict_next() - self(layer.predict_next()))
                self.state.assign_sub(self.state_lr * self.pi_bu * ((d_pred+d_state)/2.))
            if self.noise_temp > 0.0:
                # Langevin noise: turns the deterministic relaxation into sampling of the energy.
                self.state.assign_add(tf.sqrt(2.0*self.state_lr*self.noise_temp) * tf.random.normal(self.state.shape))
            if self.state_clip != float('inf'):
                self.state.assign(tf.clip_by_value(self.state, -self.state_clip, self.state_clip))
    # pred_err = state - pred
    # 1/2*(state - pred)^2 = 1/2*(state - act(x@wts+b))^2
    # x.t @ ((state - act(x@wts+b))*act'(x@wts+b))
    # (B, N) (B, M)
    # 1/2*(state - pred)^2 = 1/2*( self.prev_layer.predict_next - self.predict_prev)^2
    # self.predict_prev  = (self.state-self.b) @ tf.transpose(self.wts)
    # ( self.prev_layer.predict_next - self.predict_prev) @ (self.state-self.b)
    # (B, N) (B, M)
    def update_wts(self):
        d_state = tf.zeros_like(self.wts)
        d_pred = tf.zeros_like(self.wts)
        if not self.fix_wts_b and self.prev_layer is not None:
            if not self.prev_layer.is_clamped:
                if self.activation == 'relu':
                    x = tf.linalg.matrix_transpose(self.prev_layer.predict_next()) @ (
                        -(self.predict_next() - self(self.prev_layer.predict_next()))*(self.d_gelu(self.net_in(self.prev_layer.predict_next()))))
                    d_state += tf.reduce_mean(x, axis=tf.range(0, tf.rank(x) - 2))
                else:
                    x = tf.linalg.matrix_transpose(self.prev_layer.predict_next()) @ -(self.predict_next() - self(self.prev_layer.predict_next()))
                    d_state += tf.reduce_mean(x, axis=tf.range(0, tf.rank(x) - 2))
            if not self.is_clamped:
                if self.activation == 'relu':
                    x = tf.linalg.matrix_transpose(tf.nn.relu(self.prev_layer.predict_next()) - tf.nn.relu(self.predict_prev())) @ -(self.predict_next()-self.b)
                    d_pred += tf.reduce_mean(x, axis=tf.range(0, tf.rank(x) - 2))
                else:
                    x = tf.linalg.matrix_transpose(self.prev_layer.predict_next() - self.predict_prev()) @ -(self.predict_next()-self.b)
                    d_pred += tf.reduce_mean(x, axis=tf.range(0, tf.rank(x) - 2))
            if self.untied:
                # untied: each matrix gets ONE duty and ONE local error, nothing competes.
                wd = self.weight_decay
                if not self.td_only and not self.prev_layer.is_clamped:
                    wn = tf.norm(self.wts)
                    trust = tf.minimum(wn / (tf.norm(d_state) + wd * wn + 1e-6), self.trust_cap)
                    self.wts.assign_sub(self.learning_rate * trust * (d_state + wd * self.wts))
                if not self.is_clamped:
                    wn2 = tf.norm(self.wts_td)
                    trust2 = tf.minimum(wn2 / (tf.norm(d_pred) + wd * wn2 + 1e-6), self.trust_cap)
                    self.wts_td.assign_sub(self.learning_rate * trust2 * (d_pred + wd * self.wts_td))
                    # the affine offset's local gradient: the mean top-down prediction error below
                    e_td = self.predict_prev() - self.prev_layer.predict_next()
                    self.c_td.assign_sub(self.learning_rate * tf.reduce_mean(tf.reshape(e_td, (-1, int(e_td.shape[-1]))), axis=0))
                if self.iso_eta != 0.0:
                    # the unopposed td step norm-inflates without this (cliff at ~ep2); in
                    # distillation (td_only) wts is frozen, so the flow touches only wts_td,
                    # where it also normalizes the copy-init gain of the expanding edges
                    if not self.td_only:
                        self._iso_flow(self.wts)
                    self._iso_flow(self.wts_td)
            elif not self.is_clamped or not self.prev_layer.is_clamped:
                denom = tf.cast(int(not self.is_clamped)+int(not self.prev_layer.is_clamped), tf.float32)
                g = (d_state + d_pred) / denom
                wd = self.weight_decay
                if self.weight_norm:
                    norm = tf.norm(self.wts, axis=0, keepdims=True) + 1e-8       # (1, out)
                    vhat = self.wts / norm
                    dg = tf.reduce_sum(g * vhat, axis=0)                         # (out,)
                    dv = (self.g_mag[None, :] / norm) * (g - dg[None, :] * vhat)
                    # Trust-normalize the step (LARS-style) on ||wts||=||v||, which the tangential
                    # split preserves, so a single lr works across fan-in with no inflation feedback.
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
                if self.iso_eta != 0.0:
                    self._iso_flow(self.wts)

    def _iso_flow(self, v):
        # local soft SCALED semi-orthogonalization (isometry constraint): drive the small-side
        # Gram toward c*I with c the matrix's OWN current scale (trace/n), killing anisotropy
        # while leaving the scale to the recon recipe. Relative-deviation step bounded by eta.
        # Local (this matrix only), no error signal, no backprop.
        din, dout = int(v.shape[0]), int(v.shape[-1])
        n = min(din, dout)
        if dout <= din:
            gram = tf.linalg.matrix_transpose(v) @ v
        else:
            gram = v @ tf.linalg.matrix_transpose(v)
        c = tf.linalg.trace(gram) / float(n)
        dev = (c * tf.eye(n) - gram) / (c + 1e-8)
        if dout <= din:
            v.assign_add(self.iso_eta * (v @ dev))
        else:
            v.assign_add(self.iso_eta * (dev @ v))

    # pred_err = state - pred
    # 1/2*(state - pred)^2 = 1/2*(state - act(x@wts+b))^2
    # ((state - act(x@wts+b))*act'(x@wts+b))
    # (B, N) (B, M)
    # 1/2*(state - pred)^2 = 1/2*( self.prev_layer.predict_next - self.predict_prev)^2
    # self.predict_prev  = (self.state-self.b) @ tf.linalg.matrix_transpose(self.wts)
    # ( self.prev_layer.predict_next - self.predict_prev) @ (self.state-self.b)
    # (B, N) (B, M)
    def update_b(self):
        d_state = tf.zeros_like(self.b)
        d_pred = tf.zeros_like(self.b)
        if not self.fix_wts_b and self.prev_layer is not None:
            if not self.prev_layer.is_clamped:
                if self.activation == 'relu':
                    x = (-(self.predict_next() - self(self.prev_layer.predict_next()))*(self.d_gelu(self.net_in(self.prev_layer.predict_next()))))
                    d_state += tf.reduce_mean(x, axis=tf.range(0, tf.rank(x) - 1))
                else:
                    x = -(self.predict_next() - self(self.prev_layer.predict_next()))
                    d_state += tf.reduce_mean(x, axis=tf.range(0, tf.rank(x) - 1))
            if not self.is_clamped:
                if self.activation == 'relu':
                    x = tf.reduce_mean((tf.nn.relu(self.prev_layer.predict_next()) - tf.nn.relu(self.predict_prev())) @ self.weight_td(), axis=0)
                    d_pred += tf.reduce_mean(x, axis=tf.range(0, tf.rank(x) - 1))
                else:
                    x = tf.reduce_mean((self.prev_layer.predict_next() - self.predict_prev()) @ self.weight_td(), axis=0)
                    d_pred += tf.reduce_mean(x, axis=tf.range(0, tf.rank(x) - 1))
            if not self.is_clamped or not self.prev_layer.is_clamped:
                self.b.assign_sub(self.bias_lr*(d_state+d_pred)/tf.cast(int(not self.is_clamped)+int(not self.prev_layer.is_clamped), tf.float32))


    def init_state(self):
        self.state = None

    def init_wts_b(self):
        self.wts = None
        self.b = None

    def net_in(self, x:tf.Tensor):
        if self.wts is None:
            self.init_params(x.shape)
        return x @ self.weight() + self.b

    def __call__(self, x : tf.Tensor, set_state:bool = False):
        net_in = self.net_in(x)
        if self.activation == 'relu':
            net_act = tf.nn.relu(net_in)
        else:
            net_act = net_in

        if set_state:
            if self.share_state_layer is not None:
                self.state = self.share_state_layer.state
            else:
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


