import numpy as np
import tensorflow as tf
tf.enable_eager_execution()

import random
import pickle
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from target_lstm import TARGET_LSTM

#########################################################################################
#  Generator Lstm Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 20 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 120 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64

#########################################################################################
#  Discriminator CNN Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 200
positive_file = 'save/real_data.txt'
negative_file = 'save/generator_sample.txt'
eval_file = 'save/eval_file.txt'
generated_num = 10000


def generate_samples(trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate())

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)

def pre_train_epoch(trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)

def target_loss(target_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = target_lstm.pretrain(batch)
        nll.append(g_loss)

    return np.mean(nll)

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    gen_data_load = Gen_Data_loader(BATCH_SIZE)
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE) # For testing
    vocab_size = 5000
    dis_data_loader = Dis_dataloader(BATCH_SIZE)

    # generator
    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)

    # target latm
    target_params = pickle.load(open('./save/target_params_py3.pkl', "rb"))
    target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE,EMB_DIM, HIDDEN_DIM, SEQ_LENGTH,
                              START_TOKEN, target_params)

    discriminator = Discriminator(seq_len=20, num_classes=2, vocab_size=vocab_size,
                                  embed_size=dis_embedding_dim,
                                  filter_sizes=dis_filter_sizes, num_filters=dis_num_filters,
                                  l2_reg_lambda=dis_l2_reg_lambda)
    # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
    generate_samples(target_lstm, BATCH_SIZE, generated_num, positive_file)
    gen_data_load.create_batches(positive_file)


    log = open('save/experiment-log.txt', 'w')

    # pretrain generator
    log.write('pre-training...\n')
    for epoch in range(PRE_EPOCH_NUM):
        loss = pre_train_epoch(generator, gen_data_load)
        if epoch % 5 == 0:
            generate_samples(generator, BATCH_SIZE, generated_num, eval_file)
            likelihood_data_loader.create_batches(eval_file)
            test_loss = target_loss(target_lstm, likelihood_data_loader)
            print('pre-train epoch ', epoch, 'test_loss ', test_loss)
            buffer = 'epoch:\t' + str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
            log.write(buffer)

    # training and checkpointing
    # Create the checkpoint path and the checkpoint manager. This will be used to save checkpoints every n epochs.
    checkpoint_path = "./checkpoints/train"
    ckpt = tf.train.Checkpoint()

if __name__ == "__main__":
    main()





