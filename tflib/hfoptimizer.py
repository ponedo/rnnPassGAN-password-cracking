""" Hessian Free Optimizer """
""" Author: MoonLight, 2018 """


import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops


class clr:
  """ Used for color debug output to console. """
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'


class HFOptimizer(object):
  """ Tensorflow based Hessian-Free (Truncated Newton) optimizer.
  More details: (Martens, ICML 2010) and (Martens & Sutskever, ICML 2011).

  Methods to use:
  __init__:
      Creates Tensorflow graph and variables.
  minimize:
      Perfoms HF optimization. """

  DUMPING_NUMERICAL_ERROR_STOP_FLOAT32 = 1e-4
  CG_NUMERICAL_ERROR_STOP_FLOAT32 = 1e-20
  DUMPING_NUMERICAL_ERROR_STOP_FLOAT64 = 1e-8
  CG_NUMERICAL_ERROR_STOP_FLOAT64 = 1e-80

  def __init__(self, sess, loss, output,
        learning_rate=1,
        cg_decay=0.95, 
        damping=0.5,
        adjust_damping=True,
        batch_size=None,
        use_gauss_newton_matrix=False,
        preconditioner=False,
        prec_loss=None,
        gap=10,
        cg_max_iters=50,
        dtype=tf.float32):
    """ Creates Tensorflow graph and variables.

    sess: Tensorflow session object
        Used for conjugate gradient computation.
    loss: Tensorflow tensor object
        Loss function of the neural network.
    output: Tensorflow tensor object
        Variable with respect to which the Hessian of the objective is
        positive-definite, implicitly defining the Gauss-Newton matrix.
        Typically, it is the activation of the output layer.
    learning_rate: float number
        Learning rate parameter for training neural network.
    cg_decay: float number
        Decay for previous result of computing delta with conjugate gradient
        method for the initialization of next iteration conjugate gradient.
    damping: float number
        Initial value of the Tikhonov damping coefficient.
    adjust_damping: bool
        Whether adjust damping parameter dynamically using
        Levenberg-Marquardt heuristic or not.
    batch_size: int number or None
        Used for Jacobian vector product (Rop) computation, necessary if
        used dynamic input batch size.
    use_gauss_newton_matrix: bool
        Whether use Gauss Newton Matrix (True) or Hessian matrix (False) in
        conjugate gradient computation.
    preconditioner: bool
        Martens preconditioner uses The Empirical Fisher Diagonal for its
        computation, don't use it with dynamically adjusting damping
        parameter, because it can cause numerical errors.
        Can be used only with Gauss Newton Matrix.
    prec_loss: Tensorflow tensor object
        Used for computing preconditioner, if use preconditioner it's
        better to set it explicitly. For this parameter use loss before reduce
        sum function over the batch inputs is applied.
    gap: int
        Size of window gap for which delta loss difference is computed,
        used for early stoping in conjugate gradient computation.
    cg_max_iters: int
        Number of maximum iterations of conjugate gradient computations.
    dtype: Tensorflow type
        Type of Tensorflow variables. """

    self.sess = sess
    self.loss = loss
    self.output = output
    self.cg_decay = cg_decay
    self.prec_loss = prec_loss
    self.batch_size = batch_size
    self.use_prec = preconditioner
    self.learning_rate = learning_rate
    self.use_gnm = use_gauss_newton_matrix
    self.damping = damping
    self.gap = gap
    self.cg_max_iters = cg_max_iters
    self.adjust_damping = adjust_damping
    self.damp_pl = tf.placeholder(dtype, shape=())

    self.cg_num_err = HFOptimizer.CG_NUMERICAL_ERROR_STOP_FLOAT32
    self.damp_num_err = HFOptimizer.DUMPING_NUMERICAL_ERROR_STOP_FLOAT32
    if dtype == tf.float64:
      self.cg_num_err = HFOptimizer.CG_NUMERICAL_ERROR_STOP_FLOAT64
      self.damp_num_err = HFOptimizer.DUMPING_NUMERICAL_ERROR_STOP_FLOAT64
    if not self.use_gnm:
      self.damp_num_err = 1e-1

    if not self.use_gnm and self.use_prec:
      self.use_prec = False
      print(clr.WARNING + 'WARNING: You set preconditioner to True but ' +
        'use_gauss_newton_matrix to False, and it\'s prohibited, so we set ' +
        'preconditioner back to False, if you ask why see more information ' +
        'on (Martens & Sutskever, ICML 2011).' + clr.ENDC)
    elif self.use_prec and self.use_gnm and self.prec_loss is None:
      print(clr.WARNING + 'WARNING: If you use preconditioner it is ' +
        'better to set prec_loss explicitly, because it can cause graph ' +
        'making problem. (What\'s prec_loss see in description)' + clr.ENDC)

    """ Network weights. """
    self.W = tf.trainable_variables()

    with tf.name_scope('cg_vars'):
      self.cg_step = tf.Variable(0, trainable=False, dtype=tf.int32)
      self.cg_delta = []
      self.directions = []
      self.residuals = []
      for w in self.W:
        zeros = tf.zeros(w.get_shape(), dtype=dtype)
        delta = tf.Variable(zeros, dtype=dtype, name='delta')
        self.cg_delta.append(delta)
        d = tf.Variable(zeros, dtype=dtype, name='direction')
        self.directions.append(d)
        r = tf.Variable(zeros, dtype=dtype, name='residual')
        self.residuals.append(r)

    gradients = tf.gradients(self.loss, self.W)
    self.grads = gradients

    cg_op, res_norm, dl = self.__conjugate_gradient(gradients)
    self.ops = {
      'cg_update': cg_op,
      'res_norm': res_norm,
      'dl': dl,
      'set_delta_0': self.__update_delta_0(),
      'train': self.__train_op()
    }


  def info(self):
    """ Prints initial settings of HF optimizer. """

    print(clr.BOLD + clr.OKGREEN + 'Hessian-Free Optimizer initial settings:' +\
      clr.ENDC)
    print('    CG delta decay: {}'.format(self.cg_decay))
    print('    Learning Rate: {}'.format(self.learning_rate))
    print('    Initial Tikhonov damping: {}'.format(self.damping))
    if self.adjust_damping:
      print('    Optimizer adjusts damping dynamically using ' +\
       'Levenberg-Marquardt heuristic.')
    else:
      print('    Tikhonov damping is static.')
    if self.use_gnm:
      print('    Optimizer uses Gauss-Newton matrix for cg computation.')
    else:
      print('    Optimizer uses Hessian matrix for cg computation.')
    if self.use_prec:
      print('    Optimizer uses preconditioner.')
    print('    Gap of delta loss tracking: {}'.format(self.gap))
    print('    Max cg iterations: {}'.format(self.cg_max_iters))
    print(clr.OKGREEN + 'Optimizer is ready for using.' +clr.ENDC)


  def minimize(self, feed_dict, debug_print=False):
    """ Performs main training operations.
    feed_dict: dictionary
        Input training batch.
    debug_print: bool
        If True prints CG iteration number. """

    self.sess.run(tf.assign(self.cg_step, 0))
    feed_dict.update({self.damp_pl:self.damping})

    if self.adjust_damping:
      loss_before_cg = self.sess.run(self.loss, feed_dict)

    dl_track = [self.sess.run(self.ops['dl'], feed_dict)]
    self.sess.run(self.ops['set_delta_0'])

    for i in range(self.cg_max_iters):
      if debug_print:
        d_info = clr.OKGREEN + '\r[CG iteration: {}]'.format(i) + clr.ENDC
        sys.stdout.write(d_info)
        sys.stdout.flush()

      k = max(self.gap, i // self.gap)

      rn = self.sess.run(self.ops['res_norm'], feed_dict)
      if rn < self.cg_num_err:
        break

      self.sess.run(self.ops['cg_update'], feed_dict)
      dl_track.append(self.sess.run(self.ops['dl'], feed_dict))

      if i > k:
        stop = (dl_track[i+1] - dl_track[i+1-k]) / dl_track[i+1]
        if not np.isnan(stop) and stop < 1e-4:
          break

    if debug_print:
      sys.stdout.write('\n')
      sys.stdout.flush()

    if self.adjust_damping:
      feed_dict.update({self.damp_pl:0})
      dl = self.sess.run(self.ops['dl'], feed_dict)
      feed_dict.update({self.damp_pl:self.damping})

    self.sess.run(self.ops['train'], feed_dict)

    if self.adjust_damping:
      loss_after_cg = self.sess.run(self.loss, feed_dict)
      reduction_ratio = (loss_after_cg - loss_before_cg) / dl

      if reduction_ratio < 0.25 and self.damping > self.damp_num_err:
        self.damping *= 1.5
      elif reduction_ratio > 0.75 and self.damping > self.damp_num_err:
        self.damping /= 1.5



  def __conjugate_gradient(self, gradients):
    """ Performs conjugate gradient method to minimze quadratic equation 
    and find best delta of network parameters.

    gradients: list of Tensorflow tensor objects
        Network gradients.

    return: Tensorflow tensor object
        Update operation for delta.
    return: Tensorflow tensor object
        Residual norm, used to prevent numerical errors. 
    return: Tensorflow tensor object
        Delta loss. """

    with tf.name_scope('conjugate_gradient'):
      cg_update_ops = []

      prec = None
      if self.use_prec:
        if self.prec_loss is None:
          graph = tf.get_default_graph()
          lop = self.loss.op.node_def
          self.prec_loss = graph.get_tensor_by_name(lop.input[0] + ':0')
        batch_size = None
        if self.batch_size is None:
          self.prec_loss = tf.unstack(self.prec_loss)
          batch_size = self.prec_loss.get_shape()[0]
        else:
          self.prec_loss = [tf.gather(self.prec_loss, i)
            for i in range(self.batch_size)]
          batch_size = len(self.prec_loss)
        prec = [[g**2 for g in tf.gradients(tf.gather(self.prec_loss, i),
          self.W)] for i in range(batch_size)]
        prec = [(sum(tensor) + self.damping)**(-0.75)
          for tensor in np.transpose(np.array(prec))]

      Ax = None
      if self.use_gnm:
        Ax = self.__Gv(self.cg_delta)
      else:
        Ax = self.__Hv(gradients, self.cg_delta)

      b = [-grad for grad in gradients]
      bAx = [b - Ax for b, Ax  in zip(b, Ax)]

      condition = tf.equal(self.cg_step, 0)
      r = [tf.cond(condition, lambda: tf.assign(r,  bax),
        lambda: r) for r, bax  in zip(self.residuals, bAx)]

      d = None
      if self.use_prec:
        d = [tf.cond(condition, lambda: tf.assign(d, p * r), 
          lambda: d) for  p, d, r in zip(prec, self.directions, r)]
      else:
        d = [tf.cond(condition, lambda: tf.assign(d, r), 
          lambda: d) for  d, r in zip(self.directions, r)]

      Ad = None
      if self.use_gnm:
        Ad = self.__Gv(d)
      else:
        Ad = self.__Hv(gradients, d)

      residual_norm = tf.reduce_sum([tf.reduce_sum(r**2) for r in r])

      alpha = tf.reduce_sum([tf.reduce_sum(d * ad) for d, ad in zip(d, Ad)])
      alpha = residual_norm / alpha

      if self.use_prec:
        beta = tf.reduce_sum([tf.reduce_sum(p * (r - alpha * ad)**2) 
          for r, ad, p in zip(r, Ad, prec)])
      else:
        beta = tf.reduce_sum([tf.reduce_sum((r - alpha * ad)**2) for r, ad 
          in zip(r, Ad)])

      self.beta = beta
      beta = beta / residual_norm

      for i, delta  in reversed(list(enumerate(self.cg_delta))):
        update_delta = tf.assign(delta, delta + alpha * d[i],
          name='update_delta')
        update_residual = tf.assign(self.residuals[i], r[i] - alpha * Ad[i],
          name='update_residual')
        p = 1.0
        if self.use_prec:
          p = prec[i]
        update_direction = tf.assign(self.directions[i],
          p * (r[i] - alpha * Ad[i]) + beta * d[i], name='update_direction')
        cg_update_ops.append(update_delta)
        cg_update_ops.append(update_residual)
        cg_update_ops.append(update_direction)

      with tf.control_dependencies(cg_update_ops):
        cg_update_ops.append(tf.assign_add(self.cg_step, 1))
      cg_op = tf.group(*cg_update_ops)

    dl = tf.reduce_sum([tf.reduce_sum(0.5*(delta*ax) + grad*delta)
      for delta, grad, ax in zip(self.cg_delta, gradients, Ax)])

    return cg_op, residual_norm, dl


  def __Hv(self, grads, vec):
    """ Computes Hessian vector product.

    grads: list of Tensorflow tensor objects
        Network gradients.
    vec: list of Tensorflow tensor objects
        Vector that is multiplied by the Hessian.

    return: list of Tensorflow tensor objects
        Result of multiplying Hessian by vec. """

    grad_v = [tf.reduce_sum(g * v) for g, v in zip(grads, vec)]
    Hv = tf.gradients(grad_v, self.W, stop_gradients=vec)
    Hv = [hv + self.damp_pl * v for hv, v in zip(Hv, vec)]

    return Hv


  def __Gv(self, vec):
    """ Computes the product G by vec = JHJv (G is the Gauss-Newton matrix).

    vec: list of Tensorflow tensor objects
        Vector that is multiplied by the Gauss-Newton matrix.

    return: list of Tensorflow tensor objects
        Result of multiplying Gauss-Newton matrix by vec. """

    Jv = self.__Rop(self.output, self.W, vec)
    Jv = tf.reshape(tf.stack(Jv), [-1, 1])
    HJv = tf.gradients(tf.matmul(tf.transpose(tf.gradients(self.loss,
      self.output)[0]), Jv), self.output, stop_gradients=Jv)[0]
    JHJv = tf.gradients(tf.matmul(tf.transpose(HJv), self.output), self.W,
      stop_gradients=HJv)
    JHJv = [gv + self.damp_pl * v for gv, v in zip(JHJv, vec)]

    return JHJv


  def __Rop(self, f, x, vec):
    """ Computes Jacobian vector product.

    f: Tensorflow tensor object
        Objective function.
    x: list of Tensorflow tensor objects
        Parameters with respect to which computes Jacobian matrix.
    vec: list of Tensorflow tensor objects
        Vector that is multiplied by the Jacobian.

    return: list of Tensorflow tensor objects
        Result of multiplying Jacobian (df/dx) by vec. """

    r = None
    if self.batch_size is None:
      try:
        r = [tf.reduce_sum([tf.reduce_sum(v * tf.gradients(f, x)[i])
             for i, v in enumerate(vec)])
             for f in tf.unstack(f)]
      except ValueError:
        assert False, clr.FAIL + clr.BOLD + 'Batch size is None, but used '\
          'dynamic shape for network input, set proper batch_size in '\
          'HFOptimizer initialization' + clr.ENDC
    else:
      r = [tf.reduce_sum([tf.reduce_sum(v * tf.gradients(tf.gather(f, i), x)[j]) 
           for j, v in enumerate(vec)])
           for i in range(self.batch_size)]

    assert r is not None, clr.FAIL + clr.BOLD +\
      'Something went wrong in Rop computation' + clr.ENDC

    return r


  def __update_delta_0(self):
    """ Update initial delta for conjugate gradient method,
    old delta multiplied by cg_decay. 

    return: list of Tensorflow tensor objects
        Update initial delta operation. """

    update_delta_0_ops = []
    for delta in self.cg_delta:
      update_delta = tf.assign(delta, self.cg_decay * delta)
      update_delta_0_ops.append(update_delta)
    update_delta_0_op = tf.group(*update_delta_0_ops)

    return update_delta_0_op


  def __train_op(self):
    """ Performs main training operation, i.e. updates weights

    return: list of Tensorflow tensor objects
        Main training operations """

    update_ops = []
    delta_and_vars = list(zip(self.cg_delta, self.W))
    for delta, w in reversed(delta_and_vars):
      with tf.control_dependencies(update_ops):
        update_ops.append(tf.assign(w, w + self.learning_rate * delta))
    training_op = tf.group(*update_ops)

    return training_op