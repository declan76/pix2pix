import tensorflow as tf

class Dataset:
    """
    Dataset class for creating a TensorFlow dataset from a data loader.
    """
    
    def __init__(self, data_loader, buffer_size=400, batch_size=1):
        """
        Initializes the Dataset with the given data loader, buffer size, and batch size.
        
        Args:
        - data_loader (DataLoader): An instance of the DataLoader class to load image data.
        - buffer_size (int, optional): Size of the buffer for shuffling the dataset. Defaults to 400.
        - batch_size (int, optional): Number of samples per batch. Defaults to 1.
        """
        self.data_loader = data_loader
        self.buffer_size = buffer_size
        self.batch_size  = batch_size

    def create_dataset(self):
        """
        Creates a TensorFlow dataset using the data loader.
        
        Returns:
        - tf.data.Dataset: A TensorFlow dataset containing image pairs.
        """
        num_pairs = len(self.data_loader.pairs)
        dataset   = tf.data.Dataset.range(num_pairs)
        dataset   = dataset.map(lambda idx: tf.py_function(self.data_loader.load_image_pair, [idx], [tf.string, tf.float32, tf.float32]))
        dataset   = dataset.batch(self.batch_size)
        
        return dataset
