import tensorflow as tf 
import utils

class Shallow:
    def __init__(self):
        self.reuse = False
        self.name = 'Shallow'        
        
    def __call__(self, input):
        with tf.variable_scope(self.name):
            d1 = utils.dense_layer(input, 6833, self.reuse, 'relu', name='layer1')
            output = utils.dense_layer(d1, 1, self.reuse, 'linear', name='output')
        self.reuse = True
        
        return output
class Deep:
    def __init__(self):
        self.reuse = False
        self.name = 'Deep'        
        
    def __call__(self, input):
        with tf.variable_scope(self.name):
            d1 = utils.dense_layer(input, 100, self.reuse, 'relu', name='layer1')
            d2 = utils.dense_layer(d1, 100, self.reuse, 'relu', name='layer2')
            d3 = utils.dense_layer(d2, 100, self.reuse, 'relu', name='layer3')
            output = utils.dense_layer(d3, 1, self.reuse, 'linear', name='output')
        self.reuse = True
        
        return output