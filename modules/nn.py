class NNInterpolator:
    def __init__(self, params):
        self.params = params
    
    def train(self, data_train, data_test):
        # Set up the network
        depth = self.params.get("depth", 3)
        epochs = self.params.get("epochs", 100)
        # etc... Add more parameters as needed for comparison
        
        q_train, w_train = data_train
        q_test, w_test = data_test

        # Do the training loop here

        pass

    def predict(self, q: float, w: float) -> np.ndarray:
        """
        Predict the value of the response function in the point (q,w)
        Should return an array of shape (5), corresponding to the 5 curves

        """
        pass