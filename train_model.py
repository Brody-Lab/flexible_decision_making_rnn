#!/usr/bin/env python
# coding: utf-8

# Copyright 2019 Google LLC
#
# This is the original "decision" model written by David Sussillo.
# 100 hidden neurons, all weights are initialized randomly, 120 iterations. 


# # Imports

from __future__ import print_function, division, absolute_import

import datetime
from functools import partial
#import h5py
import os
import sys
import time

from jax import grad, jacrev, jit, lax, ops, random, vmap
# # # from jax.experimental import optimizers
from jax.example_libraries import optimizers
import jax.numpy as np

#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as onp  # original CPU-backed NumPy
from scipy.spatial.distance import pdist, squareform


num_batchs = 120000         # Total number of batches to train on.
print_every = 1000          # Print training informatino every so often



N_UNITS = int(sys.argv[1])
RANDOM_SEED = int(sys.argv[2])

print("number of units is " + str(N_UNITS))
print("random seed is " + str(RANDOM_SEED))



MAX_SEED_INT = 10000000
ONP_RNG = onp.random.RandomState(seed=0) # For CPU-based numpy randomness


# # Code

# ## My encoding of Pagan's click stimulus

print("note the 0.5 random + 0.25.  That's just to make learning a bit cleaner.")


def generate_pulses(input_params, key):
  """Build an approximation of Pagan's pulse stimulus."""
  rate, T, ntime, do_decision = input_params
  dt = T / ntime

  lmr_input = []
  hml_input = []
  total_isi = 0.0
  max_events_per_bin = 5 # If this is low, poisson rate will be off.
  all_nevents = []
  keys = random.split(key, 6)
  lr_ratio = 0.5*random.uniform(keys[0]) + 0.25 # input lr_ratio isn't used right now
  hl_ratio = 0.5*random.uniform(keys[1]) + 0.25 # input hl_ratio isn't used right now
  isis_lh = random.exponential(keys[2], (max_events_per_bin * ntime,)) / (rate * lr_ratio * hl_ratio)
  isis_ll = random.exponential(keys[3], (max_events_per_bin * ntime,)) / (rate * lr_ratio * (1-hl_ratio))
  isis_rh = random.exponential(keys[4], (max_events_per_bin * ntime,)) / (rate * (1-lr_ratio) * hl_ratio)
  isis_rl = random.exponential(keys[5], (max_events_per_bin * ntime,)) / (rate * (1-lr_ratio) * (1-hl_ratio))
  events_lh = np.cumsum(isis_lh)
  events_ll = np.cumsum(isis_ll)
  events_rh = np.cumsum(isis_rh)
  events_rl = np.cumsum(isis_rl)
  nevents = []
  for tidx, t in enumerate(np.linspace(0, T, ntime)):
    nevents_lh = np.sum(np.logical_and(events_lh > t-dt, events_lh <= t)).astype(np.int32)
    nevents_ll = np.sum(np.logical_and(events_ll > t-dt, events_ll <= t)).astype(np.int32)
    nevents_rh = np.sum(np.logical_and(events_rh > t-dt, events_rh <= t)).astype(np.int32)
    nevents_rl = np.sum(np.logical_and(events_rl > t-dt, events_rl <= t)).astype(np.int32)
    left_minus_right = nevents_lh + nevents_ll - nevents_rh - nevents_rl
    high_minus_low =   nevents_lh - nevents_ll + nevents_rh - nevents_rl
    lmr_input.append(left_minus_right)
    hml_input.append(high_minus_low)
    nevents.append(nevents_lh + nevents_ll + nevents_rh + nevents_rl)

  #print(np.sum(np.array(nevents)))
  lmr_input_tx1 = np.expand_dims(np.array(lmr_input), axis=1)
  hml_input_tx1 = np.expand_dims(np.array(hml_input), axis=1)

  # The context signal a hot one for the duration of the trial.
  con1_tx2 = np.concatenate((np.ones((ntime,1)), np.zeros((ntime,1))), axis=1)
  con2_tx2 = np.concatenate((np.zeros((ntime,1)), np.ones((ntime,1))), axis=1)
  #con1_tx2 = np.concatenate((np.zeros((ntime,1)), np.zeros((ntime,1))), axis=1)
  #con2_tx2 = np.concatenate((np.zeros((ntime,1)), np.zeros((ntime,1))), axis=1)

  context = random.bernoulli(keys[2])
  context_tx2 = np.where(context, con1_tx2, con2_tx2)

  input_tx4 = np.concatenate([lmr_input_tx1, hml_input_tx1, context_tx2], axis=1)

  # The target is a contextual decision
  integral_lmr = np.sum(lmr_input_tx1[:,0])
  integral_hml = np.sum(hml_input_tx1[:,0])
  output1_t = np.zeros((ntime,))
  #output1_t = ops.index_update(output1_t, ntime-1, 1.0)
  output1_t = output1_t.at[ntime-1].set(1.0) # output 1 for left
  output2_t = np.zeros((ntime,))
  #output2_t = ops.index_update(output1_t, ntime-1, -1.0)
  output2_t = output2_t.at[ntime-1].set(-1.0)  # output -1 for left
  output3_t = np.zeros((ntime,)) # 'null' output
  target_decision_t = np.where(context,
                               np.where(integral_lmr > 0.0, output1_t, output2_t), # context == 1 (direction), evaluate this condition
                               np.where(integral_hml > 0.0, output1_t, output2_t)) # context == 0 (freq.), evaluate this condition
  # Handle 0 case, which will definitely arise for small rates.
  target_decision_t = np.where(context, 
                               np.where(integral_lmr == 0.0, output3_t, target_decision_t), # context == 1 (direction), evaluate this condition
                               np.where(integral_hml == 0.0, output3_t, target_decision_t)) # context == 0 (freq.), evaluate this condition
  

  target_integral_t = np.where(context,
                               np.cumsum(lmr_input_tx1[:,0]),
                               np.cumsum(hml_input_tx1[:,0]))

  target_t = np.where(do_decision, target_decision_t, target_integral_t)
  target_tx1 = np.expand_dims(target_t, axis=1)

  # TODO(sussillo): Incorporate mask.
  # Target only defined for the final time step.
  mask_decision_tx1 = np.zeros((ntime,1))
  #mask_decision_tx1 = ops.index_update(mask_decision_tx1, ntime - 1, 1.0) 
  mask_decision_tx1 = mask_decision_tx1.at[ntime-1].set(1.0)
  mask_integral_tx1 = np.ones((ntime,1))
  mask_tx1 = np.where(do_decision, mask_decision_tx1, mask_integral_tx1)
  return input_tx4, target_tx1, mask_tx1


def build_inputs_and_targets(input_params, keys):
  f = partial(generate_pulses, input_params)
  f_vmap = vmap(f, (0,))
  return f_vmap(keys)

build_inputs_and_targets_jit = jit(build_inputs_and_targets, 
                                   static_argnums=(0,))


# In[6]:


def plot_input_output(input_params, input_bxtxu, target_bxtxo=None,
                      output_bxtxo=None, errors_bxtxo=None):
  rate, T, ntimesteps, do_decision = input_params

  """Plot some white noise / integrated white noise examples."""
  B = input_bxtxu.shape[0]
  idx = onp.random.randint(B)
  plt.figure(figsize=(24,4))
  plt.subplot(141)
  cinp0_val = input_bxtxu[idx,0,2] 
  lw = (2*cinp0_val+1)
  plt.stem(input_bxtxu[idx,:,0].T,use_line_collection=True)
  inp0_mean = onp.mean(input_bxtxu[idx,:,0])
  plt.plot([0, ntimesteps-1], [inp0_mean, inp0_mean], 'r', lw=lw)
  plt.xlim([0, ntimesteps-1])
  plt.ylabel('Noise')
  plt.xlabel('Timesteps')

  plt.subplot(142)
  cinp1_val = input_bxtxu[idx,0,3] 
  lw = (2*cinp1_val+1)
  plt.stem(input_bxtxu[idx,:,1].T,use_line_collection=True)
  inp1_mean = onp.mean(input_bxtxu[idx,:,1])
  plt.plot([0, ntimesteps-1], [inp1_mean, inp1_mean], 'r', lw=lw)
  plt.xlim([0, ntimesteps-1])
  plt.ylabel('Noise')
  plt.xlabel('Timesteps')

  inp_mean = (inp0_mean * input_bxtxu[idx,0,2] + 
              inp1_mean * input_bxtxu[idx,1,3])

  plt.subplot(143)
  if output_bxtxo is not None:
    plt.plot(output_bxtxo[idx,:,0].T);
    plt.xlim([0, ntimesteps-1]);
    plt.xlabel('Timesteps')
  if target_bxtxo is not None:
    plt.plot(target_bxtxo[idx,:,0].T, '--');
    plt.xlim([0, ntimesteps-1]);
    plt.xlabel('Timesteps')
    plt.ylabel("Integration")
  if errors_bxtxo is not None:
    plt.subplot(144)
    plt.plot(errors_bxtxo[idx,:,0].T, '--');
    plt.xlim([0, ntimesteps-1]);
    plt.ylabel("|Errors|")
    plt.xlabel('Timesteps')


# ## RNN definition

# In[7]:


"""Vanilla RNN functions for init, definition and running."""

def random_vrnn_params(key, u, n, o, g=1.0, h=0.0, input_scale=1.0):
  """Generate random RNN parameters"""
  keys = random.split(key, 4)
  hscale = 0.1
  ifactor = input_scale / np.sqrt(u)
  rfactor = g / np.sqrt(n)
  pfactor = 1.0 / np.sqrt(n)
  return {'h0' : random.normal(keys[0], (n,)) * hscale,
          'wI' : random.normal(keys[1], (n,u)) * ifactor,
          'wR' : random.normal(keys[2], (n,n)) *  rfactor + h * np.eye(n),
          'wO' : random.normal(keys[3], (o,n)) * pfactor,
          'bR' : np.zeros([n]),
          'bO' : np.zeros([o])}


def affine(params, x):
  """Implement y = w x + b"""
  return np.dot(params['wO'], x) + params['bO']


# Affine expects n_W_m m_x_1, but passing in t_x_m (has txm dims) So
# map over first dimension to hand t_x_m.  I.e. if affine yields 
# n_y_1 = dot(n_W_m, m_x_1), then batch_affine yields t_y_n.  
batch_affine = vmap(affine, in_axes=(None, 0))


def vrnn(params, h, x):
  """Run the Vanilla RNN one step"""
  a = np.dot(params['wI'], x) + params['bR'] + np.dot(params['wR'], h)
  return np.tanh(a)


def vrnn_scan(params, h, x):
  """Run the Vanilla RNN one step, returning (h ,h)."""  
  h = vrnn(params, h, x)
  return h, h


def vrnn_run_with_h0(params, x_t, h0):
  """Run the Vanilla RNN T steps, where T is shape[0] of input."""
  h = h0
  f = partial(vrnn_scan, params)
  _, h_t = lax.scan(f, h, x_t)
  o_t = batch_affine(params, h_t)
  return h_t, o_t


def vrnn_run(params, x_t):
  """Run the Vanilla RNN T steps, where T is shape[0] of input."""
  return vrnn_run_with_h0(params, x_t, params['h0'])

  
# Let's upgrade it to handle batches using `vmap`
# Make a batched version of the `predict` function
batched_rnn_run = vmap(vrnn_run, in_axes=(None, 0))
batched_rnn_run_w_h0 = vmap(vrnn_run_with_h0, in_axes=(None, 0, 0))
  
  
def loss(params, inputs_bxtxu, targets_bxtxo, masks_bxtxo, l2reg):
  """Compute the least squares loss of the output, plus L2 regularization."""
  _, outs_bxtxo = batched_rnn_run(params, inputs_bxtxu)
  l2_loss = l2reg * optimizers.l2_norm(params)**2
  lms_loss = np.mean((masks_bxtxo * (outs_bxtxo - targets_bxtxo)) ** 2)
  total_loss = lms_loss + l2_loss
  return {'total' : total_loss, 'lms' : lms_loss, 'l2' : l2_loss}


def update_w_gc(i, opt_state, opt_update, get_params, x_b, f_b, m_b,
                max_grad_norm, l2reg):
  """Update the parameters w/ gradient clipped, gradient descent updates."""
  params = get_params(opt_state)

  def training_loss(params, x_b, f_b, m_b, l2reg):
    return loss(params, x_b, f_b, m_b, l2reg)['total']
  
  grads = grad(training_loss)(params, x_b, f_b, m_b, l2reg)
  clipped_grads = optimizers.clip_grads(grads, max_grad_norm)
  return opt_update(i, clipped_grads, opt_state)


loss_jit = jit(loss)
update_w_gc_jit = jit(update_w_gc, static_argnums=(2,3))


def run_trials(batched_run_fun, inputs_targets_h0s_fun, nbatches, batch_size):
  """Run a bunch of trials and save everything in a dictionary."""
  inputs = []
  hiddens = []
  outputs = []
  targets = []
  h0s = []
  key = random.PRNGKey(onp.random.randint(0, MAX_SEED_INT))
  for n in range(nbatches):
    key = random.fold_in(key, n)
    skeys = random.split(key, batch_size)
    input_b, target_b, h0s_b = inputs_targets_h0s_fun(skeys)
    if h0s_b is None:
      h_b, o_b = batched_run_fun(input_b)
    else:
      h_b, o_b = batched_run_fun(input_b, h0s_b)      
      h0s.append(h0s_b)
      
    inputs.append(input_b)
    hiddens.append(h_b)
    outputs.append(o_b)
    targets.append(target_b)
    
  trial_dict = {'inputs' : onp.vstack(inputs), 'hiddens' : onp.vstack(hiddens),
                'outputs' : onp.vstack(outputs), 'targets' : onp.vstack(targets)}
  if h0s_b is not None:
    trial_dict['h0s'] = onp.vstack(h0s)
  else:
    trial_dict['h0s'] = None
  return trial_dict


def plot_params(params):
  """ Plot the parameters of the vanilla RNN. """
  plt.figure(figsize=(16,8))
  plt.subplot(231)
  plt.stem(params['wO'][0,:],use_line_collection=True)
  plt.title('wO - output weights')
  
  plt.subplot(232)
  plt.stem(params['h0'],use_line_collection=True)
  plt.title('h0 - initial hidden state')
  
  plt.subplot(233)
  plt.imshow(params['wR'], interpolation=None)
  plt.title('wR - recurrent weights')
  plt.colorbar()
  
  plt.subplot(234)
  plt.imshow(params['wI'].T)
  plt.title('wI - input weights')
  
  plt.subplot(235)
  plt.stem(params['bR'],use_line_collection=True)
  plt.title('bR - recurrent biases')
  
  plt.subplot(236)
  evals, _ = onp.linalg.eig(params['wR'])
  x = onp.linspace(-1, 1, 1000)
  plt.plot(x, onp.sqrt(1-x**2), 'k')
  plt.plot(x, -onp.sqrt(1-x**2), 'k')
  plt.plot(onp.real(evals), onp.imag(evals), '.')
  plt.axis('equal')
  plt.title('Eigenvalues of wR')

  
def plot_examples(ntimesteps, rnn_internals, nexamples=1):
  """Plot some input/hidden/output triplets."""
  plt.figure(figsize=(nexamples*5, 12))
  for bidx in range(nexamples):
    plt.subplot(4, nexamples, bidx+1)
    plt.plot(rnn_internals['inputs'][bidx,:,0], 'b')
    plt.plot(rnn_internals['inputs'][bidx,:,2], 'k')    
    plt.xlim([0, ntimesteps])
    plt.title('Example %d' % (bidx))
    if bidx == 0:
      plt.ylabel('Input')

  for bidx in range(nexamples):
    plt.subplot(4, nexamples, 1*nexamples+bidx+1)
    plt.plot(rnn_internals['inputs'][bidx,:,1], 'b')
    plt.plot(rnn_internals['inputs'][bidx,:,3], 'k')    
    plt.xlim([0, ntimesteps])
    plt.title('Example %d' % (bidx))
    if bidx == 0:
      plt.ylabel('Input')
      
  ntoplot = 10
  closeness = 0.25
  for bidx in range(nexamples):
    plt.subplot(4, nexamples, 2*nexamples+bidx+1)
    plt.plot(rnn_internals['hiddens'][bidx, :, 0:ntoplot] +
             closeness * onp.arange(ntoplot), 'b')
    plt.xlim([0, ntimesteps])
    if bidx == 0:
      plt.ylabel('Hidden Units')
      
  for bidx in range(nexamples):
    plt.subplot(4, nexamples, 3*nexamples+bidx+1)
    plt.plot(rnn_internals['outputs'][bidx,:,:], 'r')
    plt.plot(rnn_internals['targets'][bidx,:,:], 'k')    
    plt.xlim([0, ntimesteps])
    plt.xlabel('Timesteps')
    if bidx == 0:
      plt.ylabel('Output')


# ## Fixed point finding routines

# In[8]:


"""Find the fixed points of a nonlinear system via numerical optimization."""

def find_fixed_points(rnn_fun, candidates, hps, do_print=True):
  """Top-level routine to find fixed points, keeping only valid fixed points.
  This function will:
    Add noise to the fixed point candidates ('noise_var')
    Optimize to find the closest fixed points / slow points (many hps, 
      see optimize_fps)
    Exclude any fixed points whose fixed point loss is above threshold ('fp_tol')
    Exclude any non-unique fixed points according to a tolerance ('unique_tol')
    Exclude any far-away "outlier" fixed points ('outlier_tol')
    
  This top level function runs at the CPU level, while the actual JAX optimization 
  for finding fixed points is dispatched to device.
  Arguments: 
    rnn_fun: one-step update function as a function of hidden state
    candidates: ndarray with shape npoints x ndims
    hps: dict of hyper parameters for fp optimization, including
      tolerances related to keeping fixed points
  
  Returns: 
    4-tuple of (kept fixed points sorted with slowest points first, 
      fixed point losses, indicies of kept fixed points, details of 
      optimization)"""

  npoints, dim = candidates.shape
  
  noise_var = hps['noise_var']
  if do_print and noise_var > 0.0:
    print("Adding noise to fixed point candidates.")
    candidates += onp.random.randn(npoints, dim) * onp.sqrt(noise_var)
    
  if do_print:
    print("Optimizing to find fixed points.")
  fps, opt_details = optimize_fps(rnn_fun, candidates, hps, do_print)

  if do_print and hps['fp_tol'] < onp.inf:  
    print("Excluding fixed points with squared speed above tolerance {:0.5f}.".format(hps['fp_tol']))
  fps, fp_kidxs = fixed_points_with_tolerance(rnn_fun, fps, hps['fp_tol'],
                                              do_print)
  if len(fp_kidxs) == 0:
    return onp.zeros([0, dim]), onp.zeros([0]), [], opt_details
  
  if do_print and hps['unique_tol'] > 0.0:  
    print("Excluding non-unique fixed points.")
  fps, unique_kidxs = keep_unique_fixed_points(fps, hps['unique_tol'],
                                               do_print)
  if len(unique_kidxs) == 0:
    return onp.zeros([0, dim]), onp.zeros([0]), [], opt_details
  
  if do_print and hps['outlier_tol'] < onp.inf:  
    print("Excluding outliers.")
  fps, outlier_kidxs = exclude_outliers(fps, hps['outlier_tol'],
                                        'euclidean', do_print) # TODO(sussillo) Make hp?
  if len(outlier_kidxs) == 0:
    return onp.zeros([0, dim]), onp.zeros([0]), [], opt_details

  if do_print:
    print('Sorting fixed points slowest first.')    
  losses = onp.array(get_fp_loss_fun(rnn_fun)(fps))# came back as jax.interpreters.xla.DeviceArray
  sort_idxs = onp.argsort(losses) 
  fps = fps[sort_idxs]
  losses = losses[sort_idxs]
  try:
    keep_idxs = fp_kidxs[unique_kidxs[outlier_kidxs[sort_idxs]]]
  except:
    import pdb; pdb.set_trace()
  return fps, losses, keep_idxs, opt_details


def get_fp_loss_fun(rnn_fun):
  """Return the per-example mean-squared-error fixed point loss.
  Arguments:
    rnn_fun : RNN one step update function for a single hidden state vector
      h_t -> h_t+1
  Returns: function that computes the loss for each example
  """
  batch_rnn_fun = vmap(rnn_fun, in_axes=(0,))
  return jit(lambda h : np.mean((h - batch_rnn_fun(h))**2, axis=1))


def get_total_fp_loss_fun(rnn_fun):
  """Return the MSE fixed point loss averaged across examples.
  Arguments:
    rnn_fun : RNN one step update function for a single hidden state vector
      h_t -> h_t+1
  Returns: function that computes the average loss over all examples.
  """
  fp_loss_fun = get_fp_loss_fun(rnn_fun)
  return jit(lambda h : np.mean(fp_loss_fun(h)))


def optimize_fp_core(batch_idx_start, num_batches, update_fun, opt_state):
  """Gradient updates to fixed points candidates in order to find fixed points.
  Uses lax.fori_loop instead of a Python loop to reduce JAX overhead. This 
    loop will be jit'd and run on device.
  Arguments:
    batch_idx_start: Where are we in the total number of batches
    num_batches: how many batches to run
    update_fun: the function that changes params based on grad of loss
    opt_state: the jax optimizer state, containing params and opt state
  Returns:
    opt_state: the jax optimizer state, containing params and optimizer state"""

  def run_update(batch_idx, opt_state):
    opt_state = update_fun(batch_idx, opt_state)
    return opt_state

  lower = batch_idx_start
  upper = batch_idx_start + num_batches
  return lax.fori_loop(lower, upper, run_update, opt_state)


optimize_fp_core_jit = jit(optimize_fp_core, static_argnums=(1, 2, 3))


def optimize_fps(rnn_fun, fp_candidates, hps, do_print=True):
  """Find fixed points of the rnn via optimization.
  This loop is at the cpu non-JAX level.
  Arguments:
    rnn_fun : RNN one step update function for a single hidden state vector
      h_t -> h_t+1, for which the fixed point candidates are trained to be 
      fixed points
    fp_candidates: np array with shape (batch size, state dim) of hidden states 
      of RNN to start training for fixed points
    hps: fixed point hyperparameters
    do_print: Print useful information? 
  Returns:
    np array of numerically optimized fixed points"""

  total_fp_loss_fun = get_total_fp_loss_fun(rnn_fun)

  def get_update_fun(opt_update, get_params):
    """Update the parameters using gradient descent.
    Arguments:
      opt_update: a function to update the optimizer state (from jax.optimizers)
      get_params: a function that extract parametrs from the optimizer state
    Returns:
      a 2-tuple (function which updates the parameters according to the 
        optimizer, a dictionary of details of the optimization)
    """
    def update(i, opt_state):
      params = get_params(opt_state)
      grads = grad(total_fp_loss_fun)(params)    
      return opt_update(i, grads, opt_state)

    return update

  # Build some functions used in optimization.
  decay_fun = optimizers.exponential_decay(hps['step_size'],
                                           hps['decay_steps'],
                                           hps['decay_factor'])
  opt_init, opt_update, get_params = optimizers.adam(step_size=decay_fun,
                                                     b1=hps['adam_b1'],
                                                     b2=hps['adam_b2'],
                                                     eps=hps['adam_eps'])
  opt_state = opt_init(fp_candidates)
  update_fun = get_update_fun(opt_update, get_params)

  # Run the optimization, pausing every so often to collect data and
  # print status.
  batch_size = fp_candidates.shape[0]
  num_batches = hps['num_batches']
  print_every = hps['opt_print_every']
  num_opt_loops = int(num_batches / print_every)
  fps = get_params(opt_state)
  fp_losses = []
  do_stop = False
  for oidx in range(num_opt_loops):
    if do_stop:
      break
    batch_idx_start = oidx * print_every
    start_time = time.time()
    opt_state = optimize_fp_core_jit(batch_idx_start, print_every, update_fun,
                                     opt_state)
    batch_time = time.time() - start_time

    # Training loss
    fps = get_params(opt_state)
    batch_pidx = batch_idx_start + print_every
    total_fp_loss = total_fp_loss_fun(fps)
    fp_losses.append(total_fp_loss)
    
    # Saving, printing.
    if do_print:
      s = "    Batches {}-{} in {:0.2f} sec, Step size: {:0.5f}, Training loss {:0.5f}"
      print(s.format(batch_idx_start+1, batch_pidx, batch_time,
                     decay_fun(batch_pidx), total_fp_loss))

    if total_fp_loss < hps['fp_opt_stop_tol']:
      do_stop = True
      if do_print:
        print('Stopping as mean training loss {:0.5f} is below tolerance {:0.5f}.'.format(total_fp_loss, hps['fp_opt_stop_tol']))
    optimizer_details = {'fp_losses' : fp_losses}    
  return fps, optimizer_details


def fixed_points_with_tolerance(rnn_fun, fps, tol=onp.inf, do_print=True):
  """Return fixed points with a fixed point loss under a given tolerance.
  
  Arguments: 
    rnn_fun: one-step update function as a function of hidden state
    fps: ndarray with shape npoints x ndims
    tols: loss tolerance over which fixed points are excluded
    do_print: Print useful information? 
  Returns: 
    2-tuple of kept fixed points, along with indicies of kept fixed points
  """
  fp_loss_fun = get_fp_loss_fun(rnn_fun)
  losses = fp_loss_fun(fps)
  lidxs = losses < tol
  keep_idxs = onp.where(lidxs)[0]
  fps_w_tol = fps[lidxs]
  
  if do_print:
    print("    Kept %d/%d fixed points with tolerance under %f." %
          (fps_w_tol.shape[0], fps.shape[0], tol))
  
  return fps_w_tol, keep_idxs
  

def keep_unique_fixed_points(fps, identical_tol=0.0, do_print=True):
  """Get unique fixed points by choosing a representative within tolerance.
  Args:
    fps: numpy array, FxN tensor of F fixed points of N dimension
    identical_tol: float, tolerance for determination of identical fixed points
    do_print: Print useful information? 
  Returns:
    2-tuple of UxN numpy array of U unique fixed points and the kept indices
  """
  keep_idxs = onp.arange(fps.shape[0])
  if identical_tol <= 0.0:
    return fps, keep_idxs
  if fps.shape[0] <= 1:
    return fps, keep_idxs
  
  nfps = fps.shape[0]
  example_idxs = onp.arange(nfps)
  all_drop_idxs = []

  # If point a and point b are within identical_tol of each other, and the
  # a is first in the list, we keep a.
  distances = squareform(pdist(fps, metric="euclidean"))
  for fidx in range(nfps-1):
    distances_f = distances[fidx, fidx+1:]
    drop_idxs = example_idxs[fidx+1:][distances_f <= identical_tol]
    all_drop_idxs += list(drop_idxs)
       
  unique_dropidxs = onp.unique(all_drop_idxs)
  keep_idxs = onp.setdiff1d(example_idxs, unique_dropidxs)
  if keep_idxs.shape[0] > 0:
    unique_fps = fps[keep_idxs, :]
  else:
    unique_fps = onp.array([], dtype=onp.int64)

  if do_print:
    print("    Kept %d/%d unique fixed points with uniqueness tolerance %f." %
          (unique_fps.shape[0], nfps, identical_tol))
    
  return unique_fps, keep_idxs


def exclude_outliers(data, outlier_dist=onp.inf, metric='euclidean', do_print=True):
  """Exclude points whose closest neighbor is further than threshold.
  Args:
    data: ndarray, matrix holding datapoints (num_points x num_features).
    outlier_dist: float, distance to determine outliers.
    metric: str or function, distance metric passed to scipy.spatial.pdist.
        Defaults to "euclidean"
    do_print: Print useful information? 
  Returns:
    2-tuple of (filtered_data: ndarray, matrix holding subset of datapoints,
      keep_idx: ndarray, vector of bools holding indices of kept datapoints).
  """
  if onp.isinf(outlier_dist):
    return data, onp.arange(len(data))
  if data.shape[0] <= 1:
    return data, onp.arange(len(data))

  # Compute pairwise distances between all fixed points.
  distances = squareform(pdist(data, metric=metric))

  # Find second smallest element in each column of the pairwise distance matrix.
  # This corresponds to the closest neighbor for each fixed point.
  closest_neighbor = onp.partition(distances, 1, axis=0)[1]

  # Return data with outliers removed and indices of kept datapoints.
  keep_idx = onp.where(closest_neighbor < outlier_dist)[0]
  data_to_keep = data[keep_idx]

  if do_print:
    print("    Kept %d/%d fixed points with within outlier tolerance %f." %
          (data_to_keep.shape[0], data.shape[0], outlier_dist))
  
  return data_to_keep, keep_idx                              


def compute_jacobians(rnn_fun, points):
  """Compute the jacobians of the rnn_fun at the points.
  This function uses JAX for the jacobian, and is computed on-device.
  Arguments:
    rnn_fun: RNN one step update function for a single hidden state vector
      h_t -> h_t+1
    points: np array npoints x dim, eval jacobian at this point.
  Returns: 
    npoints number of jacobians, np array with shape npoints x dim x dim
  """
  dFdh = jacrev(rnn_fun)
  batch_dFdh = jit(vmap(dFdh, in_axes=(0,)))
  return batch_dFdh(points)


def compute_eigenvalue_decomposition(Ms, sort_by='magnitude',
                                     do_compute_lefts=True):
  """Compute the eigenvalues of the matrix M. No assumptions are made on M.
  Arguments: 
    M: 3D np.array nmatrices x dim x dim matrix
    do_compute_lefts: Compute the left eigenvectors? Requires a pseudo-inverse 
      call.
  Returns: 
    list of dictionaries with eigenvalues components: sorted 
      eigenvalues, sorted right eigenvectors, and sored left eigenvectors 
      (as column vectors).
  """
  if sort_by == 'magnitude':
    sort_fun = onp.abs
  elif sort_by == 'real':
    sort_fun = onp.real
  else:
    assert False, "Not implemented yet."      
  
  decomps = []
  L = None  
  for M in Ms:
    evals, R = onp.linalg.eig(M)    
    indices = np.flipud(np.argsort(sort_fun(evals)))
    if do_compute_lefts:
      L = onp.linalg.pinv(R).T  # as columns      
      L = L[:, indices]
    indices=onp.reshape(indices,(1,len(indices)))
    decomps.append({'evals' : evals[indices], 'R' : onp.squeeze(R[:, indices]),  'L' : L})
  
  return decomps


# # Training the RNN

# # Input definition


rate = 40
T = 1.3
ntimesteps = 130
# If True +1/-1 target encoding at end. If False, cumulative integral defined 
# at each time step.
do_decision = True 
input_params = (rate, T, ntimesteps, do_decision)


# ## RNN Hyperparameters


u = 4         # Number of inputs to the RNN
n = N_UNITS       # Number of units in the RNN
o = 1         # Number of outputs in the RNN
param_scale = 0.8 # Scaling of the recurrent weight matrix
input_scale = 1.0
push_forward = 0.0


# ## Training Hyperparameters


batch_size = 256          # How many examples in each batch
eval_batch_size = 1024    # How large a batch for evaluating the RNN
step_size = 0.002          # initial learning rate
decay_factor = 0.99998     # decay the learning rate this much
# Gradient clipping is HUGELY important for training RNNs
max_grad_norm = 10.0      # max gradient norm before clipping, clip to this value.
l2reg = 0.0000           # amount of L2 regularization on the weights
adam_b1 = 0.9             # Adam parameters
adam_b2 = 0.999
adam_eps = 1e-1

# Create a decay function for the learning rate
decay_fun = lambda step: step_size
# decay_fun = optimizers.exponential_decay(step_size, decay_steps=1, 
#                                          decay_rate=decay_factor)

batch_idxs = onp.linspace(1, num_batchs)


# ## Fixed points stuff

# ## Preliminaries

# In[12]:


# We linearize around the context input values.
x_star_c1 = np.array((0.0, 0.0, 1.0, 0.0))
x_star_c2 = np.array((0.0, 0.0, 0.0, 1.0))
x_star = {'c1': x_star_c1, 'c2': x_star_c2}

# Make a one parameter function of thie hidden state, useful for jacobians.
rnn_fun_c1 = lambda h : vrnn(params, h, x_star_c1)
rnn_fun_c2 = lambda h : vrnn(params, h, x_star_c2)
rnn_fun = {'c1': rnn_fun_c1, 'c2': rnn_fun_c2}

batch_rnn_fun_c1 = vmap(rnn_fun_c1, in_axes=(0,))
batch_rnn_fun_c2 = vmap(rnn_fun_c2, in_axes=(0,))
batch_rnn_fun = {'c1': batch_rnn_fun_c1, 'c2': batch_rnn_fun_c2}

# Create some functions that define the fixed point loss.
fp_loss_fun_c1 = get_fp_loss_fun(rnn_fun_c1)
fp_loss_fun_c2 = get_fp_loss_fun(rnn_fun_c2)
fp_loss_fun = {'c1': fp_loss_fun_c1, 'c2': fp_loss_fun_c2}

total_fp_loss_fun_c1 = get_total_fp_loss_fun(rnn_fun_c1)
total_fp_loss_fun_c2 = get_total_fp_loss_fun(rnn_fun_c2)
total_fp_loss_fun = {'c1': total_fp_loss_fun_c1, 'c2': total_fp_loss_fun_c2}


# ## Fixed point optimization hyperparameters
# 

# In[13]:


fp_num_batches = 10000         # Total number of batches to train on.
fp_batch_size = 128          # How many examples in each batch
fp_step_size = 0.2          # initial learning rate
fp_decay_factor = 0.9999     # decay the learning rate this much
fp_decay_steps = 1           #
fp_adam_b1 = 0.9             # Adam parameters
fp_adam_b2 = 0.999
fp_adam_eps = 1e-5
fp_opt_print_every = 200   # Print training information during optimziation every so often

# Fixed point finding thresholds and other HPs
fp_noise_var = 0.0      # Gaussian noise added to fixed point candidates before optimization.
fp_opt_stop_tol = 0.00001  # Stop optimizing when the average value of the batch is below this value.
fp_tol = 0.00001        # Discard fps with squared speed larger than this value.
fp_unique_tol = 0.025   # tolerance for determination of identical fixed points
fp_outlier_tol = 1.0    # Anypoint whos closest fixed point is greater than tol is an outlier.

fp_tol = 0.0001 # This will be changed below.
fp_hps = {'num_batches': fp_num_batches, 
          'step_size': fp_step_size, 
          'decay_factor': fp_decay_factor, 
          'decay_steps': fp_decay_steps, 
          'adam_b1': fp_adam_b1, 'adam_b2': fp_adam_b2, 'adam_eps': fp_adam_eps,
          'noise_var': fp_noise_var, 
          'fp_opt_stop_tol': fp_tol, 
          'fp_tol': fp_tol, 
          'unique_tol': fp_unique_tol, 
          'outlier_tol': fp_outlier_tol, 
          'opt_print_every': fp_opt_print_every}


# ## Train the RNN

# In[14]:


def inputs_targets_no_h0s(keys):
    inputs_b, targets_b, masks_b = build_inputs_and_targets_jit(input_params, keys)
    h0s_b = None # Use trained h0
    return inputs_b, targets_b, h0s_b

rnn_run = lambda inputs: batched_rnn_run(params, inputs)
give_trained_h0 = lambda batch_size : np.array([params['h0']] * batch_size)




# In[ ]:


print("NEW MODEL!!!")
print(N_UNITS)
print(RANDOM_SEED)
print("  ")

path = "models/fixed_lr/model_stage_" + str(N_UNITS) + "_" + str(RANDOM_SEED)
if not os.path.exists(path):
  os.mkdirs(path)

key = random.PRNGKey(RANDOM_SEED)

init_params = random_vrnn_params(key, u, n, o, g=param_scale, 
                                 h=push_forward, input_scale=input_scale)
# Initialize the optimizer.  Please see jax/experimental/optimizers.py
opt_init, opt_update, get_params = optimizers.adam(decay_fun, adam_b1, adam_b2, adam_eps)
opt_state = opt_init(init_params)

# Parameters through training -- initialise with values before any training on full task
wIall = [init_params['wI']]
wRall = [init_params['wR']]
wOall = [init_params['wO']]
bRall = [init_params['bR']]
bOall = [init_params['bO']]
h0all = [init_params['h0']]

# Run the optimization loop, first jit'd call will take a minute.
start_time = time.time()
all_train_losses = []
all_params = []
for batch in range(num_batchs):
    key = random.fold_in(key, batch)
    skeys = random.split(key, batch_size)
    inputs, targets, masks = build_inputs_and_targets_jit(input_params, skeys)
    opt_state = update_w_gc_jit(batch, opt_state, opt_update, get_params, inputs,
                                targets, masks, max_grad_norm, l2reg)
    if (batch + 1) % print_every == 0:
        params = get_params(opt_state)
        all_train_losses.append(loss_jit(params, inputs, targets, masks, l2reg))
        # Marino added this line to keep track of the weights throughout training
        all_params.append(params)
        train_loss = all_train_losses[-1]['total']
        batch_time = time.time() - start_time
        step_size = decay_fun(batch)
        s = "Batch {} in {:0.2f} sec, step size: {:0.5f}, training loss {:0.4f}"
        print(s.format(batch, batch_time, step_size, train_loss))
        start_time = time.time()

# List of dicts to dict of lists
all_train_losses = {k: [dic[k] for dic in all_train_losses] for k in all_train_losses[0]}
losse=all_train_losses['total']

for jidx in range(int(num_batchs/print_every)):
    wI = all_params[jidx]['wI']                
    wIall.append(wI)
    wO = all_params[jidx]['wO']                
    wOall.append(wO)
    wR = all_params[jidx]['wR']                
    wRall.append(wR)
    bR = all_params[jidx]['bR']
    bRall.append(bR)
    bO = all_params[jidx]['bO']
    bOall.append(bO)
    h0 = all_params[jidx]['h0']
    h0all.append(h0)


onp.save(path + "/bO.npy",params['bO'])
onp.save(path + "/wI.npy",params['wI'])
onp.save(path + "/wO.npy",params['wO'])
onp.save(path + "/wR.npy",params['wR'])
onp.save(path + "/bR.npy",params['bR'])
onp.save(path + "/h0.npy",params['h0'])



onp.save(path + "/all_train_losses.npy", losse)


onp.save(path + "/wIall.npy",wIall)
onp.save(path + "/wOall.npy",wOall)
onp.save(path + "/wRall.npy",wRall)
onp.save(path + "/bRall.npy",bRall)
onp.save(path + "/bOall.npy",bOall)
onp.save(path + "/h0all.npy",h0all)


