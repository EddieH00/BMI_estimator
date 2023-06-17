import tensorflow as tf
import os

class BMI_Estimator:
    def __init__(self):
        self.feature_size = 512
        self.num_hidden_units_1 = 512
        self.num_hidden_units_2 = 256
        self.num_hidden_units_3 = 128
        self.num_hidden_units_4 = 64
        self.dropout_rate = 0.5
        self.learning_rate = 1e-3
        
        self._build_model()
        self.saver = tf.train.Saver()    

    def _build_model(self):
        self.features = tf.placeholder(tf.float32, shape=[None, self.feature_size])
        self.labels = tf.placeholder(tf.float32, shape=[None, 1])

        # Reshape features to a 4D tensor for compatibility with convolutional layers
        features_reshape = tf.reshape(self.features, [-1, 16, 32, 1])  # Assuming a feature_size of 512

        # Add normalization layer (e.g., batch normalization)
        normalized_features = tf.layers.batch_normalization(features_reshape)

        # Add max pooling layer
        # max_pooled_features = tf.layers.max_pooling2d(normalized_features, pool_size=(2, 2), strides=(2, 2))

        # Flatten the max pooled features for compatibility with fully connected layers
        flattened_features = tf.layers.flatten(normalized_features)

        # Continue with the existing code
        fc1 = tf.layers.dense(flattened_features, self.num_hidden_units_1, activation=tf.nn.leaky_relu)
        fc2 = tf.layers.dense(fc1, self.num_hidden_units_2, activation=tf.nn.leaky_relu)
        dropout = tf.layers.dropout(fc2, rate=self.dropout_rate)
        fc3 = tf.layers.dense(dropout, self.num_hidden_units_3, activation=tf.nn.leaky_relu)
        fc4 = tf.layers.dense(fc3, self.num_hidden_units_4, activation=tf.nn.leaky_relu)
        dropout2 = tf.layers.dropout(fc4, rate=self.dropout_rate)
        self.predictions = tf.layers.dense(dropout2, 1)
        
        #self.loss = tf.losses.mean_squared_error(self.labels, self.predictions)
        self.loss = tf.reduce_mean(tf.abs(self.labels - self.predictions))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
    
    def train(self, sess, features, labels):
        feed_dict = {self.features: features, self.labels: labels}
        _, loss = sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
        return loss
    
    def predict(self, sess, features):
        feed_dict = {self.features: features}
        return sess.run(self.predictions, feed_dict=feed_dict)

    def save_model(self, sess, epoch, base_path="model_checkpoint"):
        # Define the save path
        save_path = "{}/model_epoch_{}/".format(base_path, epoch)

        # Check if the save directory exists, if not, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Save the model in the directory specific to the epoch
        self.saver.save(sess, save_path + "model.ckpt")
        
    def load_model(self, sess, model_path):
        self.saver.restore(sess, model_path)

    @classmethod
    def load(cls, checkpoint_dir):
        model = cls()
            
        sess = tf.Session()
                
        # Get latest checkpoint
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
                
        if checkpoint_path is not None:
            model.saver.restore(sess, checkpoint_path)
        else:
            raise ValueError("No checkpoint found in directory: {}".format(checkpoint_dir))
                    
        model.sess = sess
                
        return model
