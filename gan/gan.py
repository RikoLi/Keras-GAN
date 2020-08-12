# 实现代码fork自GitHub开源仓库: eriklindernoren/Keras-GAN
# 对基本GAN的模型搭建做介绍
# 使用工具：Keras（非tensorflow.keras）
# Commented by lijiachen

from __future__ import print_function, division

from keras.datasets import mnist    # 使用Keras提供的mnist手写数字数据集做测试
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np

class GAN():
    # 定义GAN类
    def __init__(self):
        # 构造函数
        self.img_rows = 28  # 希望生成的图像行数
        self.img_cols = 28  # 希望生成的图像列数
        self.channels = 1   # 希望生成的图像通道数
        self.img_shape = (self.img_rows, self.img_cols, self.channels)  # 输出图像size打包
        self.latent_dim = 100   # 作为输入的噪声维度，默认100-d向量

        optimizer = Adam(0.0002, 0.5)   # 优化器选择

        # 搭建鉴别器
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])   # 指定鉴别器损失函数 & 优化器

        # 搭建生成器
        self.generator = self.build_generator()

        # ########## #
        # 构建运算图  #
        # ######### #
        z = Input(shape=(self.latent_dim,)) # 噪声作为输入层
        img = self.generator(z)             # 噪声送入生成器生成G(z)

        # 在GAN中设置鉴别器参数为不可更新（为了单独训练生成器）
        self.discriminator.trainable = False

        validity = self.discriminator(img)  # G(z)送入鉴别器判断真实度
        self.combined = Model(z, validity)  # 运算图构建完成，将生成GAN的模型实例
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)  # 模型编译，指定损失函数 & 优化器


    def build_generator(self):
        # 生成器结构
        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh')) # 生成器输出层激活使用tanh
        model.add(Reshape(self.img_shape))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        # 鉴别器结构
        model = Sequential()

        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # 加载数据
        (X_train, _), (_, _) = mnist.load_data()

        # 预处理
        X_train = X_train / 127.5 - 1.  # scale到[-1,1]
        X_train = np.expand_dims(X_train, axis=3)

        # 设置标签
        valid = np.ones((batch_size, 1))    # 正样本标签
        fake = np.zeros((batch_size, 1))    # 负样本标签

        for epoch in range(epochs):

            # ---------------------
            #  训练鉴别器
            # ---------------------

            # 随机选择mini batch数据
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))   # 随机生成输入噪声向量

            # 先通过生成器得到G(z)
            gen_imgs = self.generator.predict(noise)

            # 用G(z)和真实数据训练鉴别器
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)    # 用真实数据训练鉴别器
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake) # 用G(z)训练鉴别器
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)                 # 计算总损失

            # ---------------------
            #  训练生成器
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))   # 随机生成输入噪声向量
            g_loss = self.combined.train_on_batch(noise, valid)             # 训练生成器，通过指定标签为正样本“欺骗”鉴别器

            # 打印损失
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # 定期采样可视化生成结果
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # scale至[0,1]
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=30000, batch_size=32, sample_interval=200)
