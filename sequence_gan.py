import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

import random
import pickle
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from target_lstm import TARGET_LSTM
from rollout import ROLLOUT

#########################################################################################
#  Generator Lstm Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 20 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 10 # supervise (maximum likelihood estimation) epochs
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
generated_num = 100
vocab_size = 5000


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    #########################################################################################
    #  Generator, oracle(target-LSTM), discrimonator model.
    #########################################################################################
    print("loading generator, discriminator, oracle model.")
    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
    target_params = pickle.load(open('./save/target_params_py3.pkl', "rb"))
    target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE,EMB_DIM, HIDDEN_DIM, SEQ_LENGTH,
                              START_TOKEN, target_params)
    discriminator = Discriminator(seq_len=20, num_classes=2, vocab_size=vocab_size,
                                  embed_size=dis_embedding_dim,
                                  filter_sizes=dis_filter_sizes, num_filters=dis_num_filters,
                                  l2_reg_lambda=dis_l2_reg_lambda)

    #########################################################################################
    #  1. using oracle model(target-lstm) generator positive example
    #  2. using the positive example pre-train generator model.
    #  3. using pretrain generator generate evaluation example...
    #  4. using pretrain-generator generate negative example, and then pretrain discriminator.
    #  5. define rollout-policy using pretrained generator.
    #########################################################################################
    gen_data_load = Gen_Data_loader(BATCH_SIZE)
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE)  # For testing
    dis_data_loader = Dis_dataloader(BATCH_SIZE)

    # 1. use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
    print("-----------1. generate positive example using oracle model.--------")
    def generate_samples(gen_model, batch_size, generated_num, output_file):
        # Generate Samples
        generated_samples = []
        for _ in range(int(generated_num / batch_size)):                # 生成的样本数量是 batch 的整数倍
            generated_samples.extend(gen_model.generate())              # generare_num * [20]

        with open(output_file, 'w') as fout:
            for poem in generated_samples:
                buffer = ' '.join([str(x.numpy()) for x in poem]) + "\n"
                fout.write(buffer)

    generate_samples(target_lstm, BATCH_SIZE, generated_num, positive_file)
    gen_data_load.create_batches(positive_file)


    print("---------2. pre-training generator...\n, -------------"
          "---------3. and generate evaluation example...--------")
    # pretrain generator 预训练生成器
    def pre_train_epoch(trainable_model, data_loader):
        # Pre-train the generator using MLE for one epoch
        supervised_g_losses = []
        data_loader.reset_pointer()
        for it in range(data_loader.num_batch):
            batch = data_loader.next_batch()
            g_loss = trainable_model.pretrain_step(batch)
            # print(g_loss)
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
    for epoch in range(PRE_EPOCH_NUM):
        loss = pre_train_epoch(generator, gen_data_load)
        if epoch % 5 == 0:
            generate_samples(generator, BATCH_SIZE, generated_num, eval_file)  # 用预训练好的生成器，每5个epoch，生成得到验证集
            likelihood_data_loader.create_batches(eval_file)                   # 创建验证集
            test_loss = target_loss(target_lstm, likelihood_data_loader)       # 计算验证集的 loss
            print('pre-train epoch ', epoch, 'test_loss ', test_loss)
            print('epoch:\t' + str(epoch) + '\tnll:\t' + str(test_loss))

    print('-------- 4. Start pre-training discriminator...--------')
    # Train 3 epoch on the generated data and do this for 50 times
    for i in range(50):
        generate_samples(generator, BATCH_SIZE, generated_num, negative_file)
        dis_data_loader.load_train_data(positive_file, negative_file)
        d_loss = 0
        for _ in range(3):
            dis_data_loader.reset_pointer()
            for it in range(dis_data_loader.num_batch):
                x_batch, y_batch = dis_data_loader.next_batch()
                d_loss = discriminator.train_op(x_batch, y_batch)
            print("epoch\t:{} loss\t:{}".format(i, d_loss))

    print("-------- 5. define roll-out policy ---------------")
    rollout = ROLLOUT(generator, update_rate=0.8)

    #########################################################################################
    #  5. start adversarial training.
    #########################################################################################
    print("---------- 6. start Adversarial Training...")
    for total_epoch in range(TOTAL_BATCH):
        # train the generator for one step
        for it in range(1):
            samples = generator.generate()        # roll-policy部分的生成依旧用的是 pretrained generator.
            rewards = rollout.get_reward(samples, 16, discriminator)  # 基于 monte carlo 采样16，计算并累计 reward.

    # # training and checkpointing
    # # Create the checkpoint path and the checkpoint manager. This will be used to save checkpoints every n epochs.
    # checkpoint_path = "./checkpoints/train"
    # ckpt = tf.train.Checkpoint()

if __name__ == "__main__":
    main()





