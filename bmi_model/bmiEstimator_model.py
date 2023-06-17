import tensorflow as tf
import numpy as np

class BMI_Estimator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.feature_size = 512
        self.dropout_rate = 0.5
        self.learning_rate = 1e-3

        self.normalization = tf.keras.layers.BatchNormalization()
        self.flatten = tf.keras.layers.Flatten()

        self.dense1 = tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu)
        self.dense2 = tf.keras.layers.Dense(256, activation=tf.nn.leaky_relu)
        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense3 = tf.keras.layers.Dense(128, activation=tf.nn.leaky_relu)
        self.dense4 = tf.keras.layers.Dense(64, activation=tf.nn.leaky_relu)
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense5 = tf.keras.layers.Dense(1)

        # Define placeholders
        self.X = tf.placeholder(tf.float32, shape=(None, 512))
        self.Y = tf.placeholder(tf.float32, shape=(None,))

        # Get the output from the model
        self.outputs = self.call(self.X)

        # Define the loss
        self.loss = tf.losses.mean_squared_error(self.Y, self.outputs)

        # Define the optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)

        # Define the training operation
        self.train_op = self.optimizer.minimize(self.loss)

        # Initialize the variables
        self.init = tf.global_variables_initializer()

    def call(self, inputs):
        x = self.normalization(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dropout1(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dropout2(x)
        output = self.dense5(x)
        return output

    def train(self, train_data_loader, epochs):
        with tf.Session() as sess:
            sess.run(self.init)

            for epoch in range(epochs):
                total_loss = 0
                for i, (inputs, labels) in enumerate(train_data_loader):
                    _, loss_val = sess.run([self.train_op, self.loss], feed_dict={self.X: inputs, self.Y: labels})
                    total_loss += loss_val
                average_loss = total_loss / len(train_data_loader)

                print('Epoch {} Average Loss {}'.format(epoch, average_loss))
                self.save(sess, 'model_weights_epoch{}.ckpt'.format(epoch))

    def save(self, sess, file_name):
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        saver.save(sess, file_name)

    def load(self, file_name):
        sess = tf.Session()
        saver = tf.train.Saver(var_list=tf.trainable_variables())
        saver.restore(sess, file_name)
        return sess