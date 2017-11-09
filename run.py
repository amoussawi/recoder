# Copyright (c) 2017 NVIDIA Corporation
import torch
import argparse
from reco_encoder.data import input_layer
from reco_encoder.model import model
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn as nn
from torch.autograd import Variable
import copy
import time
from pathlib import Path
# from logger import Logger
from math import sqrt
import numpy as np

parser = argparse.ArgumentParser(description='RecoEncoder')
parser.add_argument('--lr', type=float, default=0.00001, metavar='N',
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.0, metavar='N',
                    help='L2 weight decay')
parser.add_argument('--drop_prob', type=float, default=0.0, metavar='N',
                    help='dropout drop probability')
parser.add_argument('--noise_prob', type=float, default=0.0, metavar='N',
                    help='noise probability')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='global batch size')
parser.add_argument('--summary_frequency', type=int, default=100, metavar='N',
                    help='how often to save summaries')
parser.add_argument('--aug_step', type=int, default=-1, metavar='N',
                    help='do data augmentation every X step')
parser.add_argument('--constrained', type=bool, default=False, metavar='N',
                    help='constrained autoencoder')
parser.add_argument('--num_epochs', type=int, default=50, metavar='N',
                    help='maximum number of epochs')
parser.add_argument('--optimizer', type=str, default="adam", metavar='N',
                    help='optimizer kind: adam, sgd, adagrad or rmsprop')
parser.add_argument('--hidden_layers', type=str, default="1024,512,512,128", metavar='N',
                    help='hidden layer sizes, comma-separated')
parser.add_argument('--gpu_ids', type=str, default="", metavar='N',
                    help='comma-separated gpu ids to use for data parallel training')
parser.add_argument('--path_to_full_train_data', type=str, default="", metavar='N',
                    help='Path to full training data')
parser.add_argument('--path_to_train_data', type=str, default="", metavar='N',
                    help='Path to training data')
parser.add_argument('--path_to_eval_data', type=str, default="", metavar='N',
                    help='Path to evaluation data')
parser.add_argument('--non_linearity_type', type=str, default="selu", metavar='N',
                    help='type of the non-linearity used in activations')
parser.add_argument('--logdir', type=str, default="logs", metavar='N',
                    help='where to save model and write logs')
parser.add_argument('--loss_func', type=str, default='MSE', metavar='N',
                    help='loss function used for training')
parser.add_argument('--last_layer_act', type=str, default='none', metavar='N',
                    help='activation function type used for last layer')
parser.add_argument('--data_delimiter', type=str, default=',', metavar='N',
                    help='data files delimiter')
parser.add_argument('--files_extension', type=str, default='.csv', metavar='N',
                    help='data files extension')
parser.add_argument('--efficient', type=bool, default=True, metavar='N',
                    help='use efficient method')


args = parser.parse_args()
print(args)

def do_eval(encoder, evaluation_data_layer):
  encoder.eval()
  denom = 0.0
  total_epoch_loss = 0.0
  for i, (eval, src) in enumerate(evaluation_data_layer.iterate_one_epoch_eval()):
    inputs = src
    targets = eval
    outputs,inputs,targets = encoder_forward(encoder, inputs, targets)
    loss, num_ratings = model.compute_loss(args.loss_func, outputs, targets)
    total_epoch_loss += loss.data[0]
    denom += num_ratings.data[0]
  return sqrt(total_epoch_loss / denom)

def log_var_and_grad_summaries(logger, layers, global_step, prefix, log_histograms=False):
  """
  Logs variable and grad stats for layer. Transfers data from GPU to CPU automatically
  :param logger: TB logger
  :param layers: param list
  :param global_step: global step for TB
  :param prefix: name prefix
  :param log_histograms: (default: False) whether or not log histograms
  :return:
  """
  for ind, w in enumerate(layers):
    # Variables
    w_var = w.data.cpu().numpy()
    logger.scalar_summary("Variables/FrobNorm/{}_{}".format(prefix, ind), np.linalg.norm(w_var),
                          global_step)
    if log_histograms:
      logger.histo_summary(tag="Variables/{}_{}".format(prefix, ind), values=w.data.cpu().numpy(),
                           step=global_step)

    # Gradients
    w_grad = w.grad.data.cpu().numpy()
    logger.scalar_summary("Gradients/FrobNorm/{}_{}".format(prefix, ind), np.linalg.norm(w_grad),
                          global_step)
    if log_histograms:
      logger.histo_summary(tag="Gradients/{}_{}".format(prefix, ind), values=w.grad.data.cpu().numpy(),
                         step=global_step)

def encoder_forward(encoder, inputs, outputs=None):
  if args.efficient:
    print('using efficient method')
    sparse_encoder = model.SparseBatchAutoEncoder(encoder, sparse_batch_in=inputs, sparse_batch_out=outputs)
    if outputs is None:
      return sparse_encoder(), sparse_encoder.reduced_batch_in
    else:
      return sparse_encoder(), sparse_encoder.reduced_batch_in, sparse_encoder.reduced_batch_out
  else:
    if not type(inputs) is Variable:
      inputs = Variable(inputs if args.gpu_ids == '' else inputs.cuda())
    if outputs is None:
      return encoder(inputs), inputs
    else:
      return encoder(inputs), inputs, outputs

def main():
  # logger = Logger(args.logdir)
  full_params = dict()
  full_params['batch_size'] = args.batch_size
  full_params['data_dir'] =  args.path_to_full_train_data
  full_params['major'] = 'users'
  full_params['itemIdInd'] = 1
  full_params['userIdInd'] = 0
  full_params['delimiter'] = args.data_delimiter
  full_params['extension'] = args.files_extension
  print("Loading full training data")
  full_data_layer = input_layer.UserItemRecDataProvider(params=full_params)
  user_id_map = full_data_layer.userIdMap
  item_id_map = full_data_layer.itemIdMap
  del full_data_layer # we don't full data any more
  print("Loading training data")
  params = copy.deepcopy(full_params)
  params['data_dir'] = args.path_to_train_data
  data_layer = input_layer.UserItemRecDataProvider(params=params,
                                                   user_id_map=user_id_map,
                                                   item_id_map=item_id_map)
  print("Data loaded")
  print("Total items found: {}".format(len(data_layer.data.keys())))
  print("Vector dim: {}".format(data_layer.vector_dim))

  # print("Loading eval data")
  # eval_params = copy.deepcopy(params)
  # # must set eval batch size to 1 to make sure no examples are missed
  # eval_params['data_dir'] = args.path_to_eval_data
  # eval_data_layer = input_layer.UserItemRecDataProvider(params=eval_params,
  #                                                       user_id_map=data_layer.userIdMap, # the mappings are provided
  #                                                       item_id_map=data_layer.itemIdMap)
  # eval_data_layer.src_data = data_layer.data
  rencoder = model.AutoEncoder(layer_sizes=[data_layer.vector_dim] + [int(l) for l in args.hidden_layers.split(',')],
                               activation_type=args.non_linearity_type,
                               is_constrained=args.constrained,
                               dp_drop_prob=args.drop_prob)

  model_checkpoint = args.logdir + "/model"
  path_to_model = Path(model_checkpoint)
  if path_to_model.is_file():
    print("Loading model from: {}".format(model_checkpoint))
    rencoder.load_state_dict(torch.load(model_checkpoint))

  print('######################################################')
  print('######################################################')
  print('############# AutoEncoder Model: #####################')
  print(rencoder)
  print('######################################################')
  print('######################################################')

  gpu_ids = [int(g) for g in args.gpu_ids.split(',')] if args.gpu_ids != '' else []
  print('Using GPUs: {}'.format(gpu_ids))
  if len(gpu_ids)>1:
    rencoder = nn.DataParallel(rencoder,
                               device_ids=gpu_ids)
  if len(gpu_ids) > 0:
    rencoder = rencoder.cuda()

  if args.optimizer == "adam":
    optimizer = optim.Adam(rencoder.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
  elif args.optimizer == "adagrad":
    optimizer = optim.Adagrad(rencoder.parameters(),
                              lr=args.lr,
                              weight_decay=args.weight_decay)
  elif args.optimizer == "sgd":
    optimizer = optim.SGD(rencoder.parameters(),
                          lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[24, 36, 48, 66, 72], gamma=0.5)
  elif args.optimizer == "rmsprop":
    optimizer = optim.RMSprop(rencoder.parameters(),
                              lr=args.lr, momentum=0.9,
                              weight_decay=args.weight_decay)
  else:
    raise  ValueError('Unknown optimizer kind')

  t_loss = 0.0
  t_loss_denom = 0.0
  global_step = 0

  if args.noise_prob > 0.0:
    dp = nn.Dropout(p=args.noise_prob)

  for epoch in range(args.num_epochs):
    print('Doing epoch {} of {}'.format(epoch, args.num_epochs))
    e_start_time = time.time()
    rencoder.train()
    total_epoch_loss = 0.0
    denom = 0.0
    if args.optimizer == "sgd":
      scheduler.step()
    for i, mb in enumerate(data_layer.iterate_one_epoch()):
      inputs = mb
      optimizer.zero_grad()
      outputs,inputs = encoder_forward(rencoder,inputs)
      loss, num_ratings = model.compute_loss(args.loss_func ,outputs, inputs)
      loss = loss / num_ratings
      loss.backward()
      optimizer.step()
      global_step += 1
      t_loss += loss.data[0]
      t_loss_denom += 1

      if i % args.summary_frequency == 0:
        if args.loss_func == 'MSE':
          print('[%d, %5d] RMSE: %.7f' % (epoch, i, sqrt(t_loss / t_loss_denom)))
          # logger.scalar_summary("Training_RMSE", sqrt(t_loss/t_loss_denom), global_step)
        elif args.loss_func == 'SoftMarginLoss':
          print('[%d, %5d] SoftMarginLoss: %.7f' % (epoch, i, t_loss / t_loss_denom))
          # logger.scalar_summary("Training_SoftMarginLoss", t_loss/t_loss_denom, global_step)

        t_loss = 0
        t_loss_denom = 0.0
        # log_var_and_grad_summaries(logger, rencoder.encode_w, global_step, "Encode_W")
        # log_var_and_grad_summaries(logger, rencoder.encode_b, global_step, "Encode_b")
        # if not rencoder.is_constrained:
          # log_var_and_grad_summaries(logger, rencoder.decode_w, global_step, "Decode_W")
        # log_var_and_grad_summaries(logger, rencoder.decode_b, global_step, "Decode_b")

      total_epoch_loss += loss.data[0]
      denom += 1

      #if args.aug_step > 0 and i % args.aug_step == 0 and i > 0:
      if args.aug_step > 0:
        # Magic data augmentation trick happen here
        for t in range(args.aug_step):
          inputs = outputs.data
          if args.noise_prob > 0.0:
            inputs = dp(inputs)
          optimizer.zero_grad()
          outputs,inputs = encoder_forward(rencoder,inputs)
          loss, num_ratings = model.compute_loss(args.loss_func, outputs, inputs)
          loss = loss / num_ratings
          loss.backward()
          optimizer.step()

    e_end_time = time.time()
    print('Total epoch {} finished in {} seconds with TRAINING RMSE loss: {}'
          .format(epoch, e_end_time - e_start_time, sqrt(total_epoch_loss/denom)))
    # logger.scalar_summary("Training_RMSE_per_epoch", sqrt(total_epoch_loss/denom), epoch)
    # logger.scalar_summary("Epoch_time", e_end_time - e_start_time, epoch)
    # if epoch % 3 == 0 or epoch == args.num_epochs - 1:
    #   eval_loss = do_eval(rencoder, eval_data_layer)
    #   print('Epoch {} EVALUATION LOSS: {}'.format(epoch, eval_loss))
    #   # logger.scalar_summary("EVALUATION_RMSE", eval_loss, epoch)
    #   print("Saving model to {}".format(model_checkpoint + ".epoch_"+str(epoch)))
    #   torch.save(rencoder.state_dict(), model_checkpoint + ".epoch_"+str(epoch))

  print("Saving model to {}".format(model_checkpoint + ".last"))
  torch.save(rencoder.state_dict(), model_checkpoint + ".last")

if __name__ == '__main__':
  main()