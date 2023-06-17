import numpy as np

class DataGenerator:
    def __init__(self, embeddings, labels, batch_size=32):
        self.embeddings = embeddings
        self.labels = labels
        self.batch_size = batch_size
        self.num_samples = len(self.embeddings)
        self.num_batches = self.num_samples // self.batch_size

    def generate_batches(self, shuffle=True):
        indices = np.arange(self.num_samples)

        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            batch_embeddings = self.embeddings[batch_indices]
            batch_labels = self.labels[batch_indices]
            
            # reshape the labels to (batch_size, 1)
            batch_labels = np.reshape(batch_labels, (-1, 1))
            yield batch_embeddings, batch_labels

    def get_generator(self):
        return self.generate_batches()
