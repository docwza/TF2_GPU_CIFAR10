import datetime
import tensorflow as tf

class Model(object):
    def __init__(self, model_arch, LR, LReps, opt, shape_input, shape_output, dropout,):
        self.model = self.get_model(model_arch, shape_input, shape_output, dropout)
        ###add desired optimizer here
        if opt =='adam':
            opt = tf.keras.optimizers.Adam(lr=LR, epsilon=LReps)
        elif opt == 'rmsprop':
            opt = tf.keras.optimizers.Adam(lr=LR, epsilon=LReps)

        self.model.compile(optimizer=opt,
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        self.val_loss = []
        self.val_acc = []

    def run(self, epoch, batch, x, y, x_test, y_test):
        self.train(epoch, batch, x, y)
        self.test(x_test, y_test)

    def train(self, epoch, batch, x, y):
        """
        x is input batch
        y is target output batch
        """
        history = self.model.fit(x, y, epochs = epoch, batch_size = batch)
        print(history.history)

    def test(self, x, y):
        self.model.evaluate(x, y)

    def get_model(self, model_arch, shape_input, shape_output, dropout):
        print(shape_input)
        blocks = [ tf.keras.layers.Input(shape=shape_input) ]
        conv_arch, fc_arch = self.decode_arch(model_arch) 
        ###loop all conv filters
        for conv in conv_arch:
            blocks.extend(self.conv_block(conv))
            if dropout > 0:
                blocks.append(tf.keras.layers.Dropout(dropout))

        blocks.append( tf.keras.layers.Flatten() ) 
        ###loop all fc layers
        for fc in fc_arch:
            blocks.extend(self.fc_block(self.decode_block(fc)))
            if dropout > 0:
                blocks.append(tf.keras.layers.Dropout(dropout))

        blocks.append(tf.keras.layers.Dense(shape_output, activation='softmax'))
        model = tf.keras.models.Sequential(blocks)
        return model

    def conv_block(self, arch):
        n_filters, k_size, s_stride, max_pool = self.decode_block(arch)
        block = [tf.keras.layers.Conv2D(n_filters, (k_size, k_size), (s_stride, s_stride), padding='same', activation='relu' ) for i in range(2)]
        if  max_pool > 1:
            block.append( tf.keras.layers.MaxPooling2D((max_pool, max_pool)) ) 
        return block

    def fc_block(self, n):
        return [ tf.keras.layers.Dense(n, activation='relu')  ]
  
    def decode_arch(self, arch):
        layers = ['F', 'C']
        for l in layers:
            layer_split = arch.split('/')
            blocks = [ s for s in layer_split if s.count(l) == 1 ]
            if l == 'F':
                fc_arch = blocks
            elif l == 'C':
                conv_arch = blocks

        print(conv_arch)
        print(fc_arch)
        return conv_arch, fc_arch

    def decode_block(self, arch):
        if arch[:1] == 'C':
            arch = arch[1:]
            args = arch.split('_')
            n_filters = args[0]
            k_size = args[1]
            s_stride = args[2]
            max_pool = args[3]
            return int(n_filters), int(k_size), int(s_stride), int(max_pool)
        elif arch[:1] == 'F':
            ###specifies number of neurons
            return int(arch[1:])

    #def load(self):

    #def save(self):

if __name__ == '__main__':
    #test Model class
    model_arch = "C48_3_1_2/C96_3_1_2/C192_3_1_2/F512/F256"
    lr = 0.0005
    lreps = 0.001
    opt = 'adam'
    input_shape = (32,32,3)
    output_shape = 10
    dropout = 0.33

    model = Model(model_arch, lr, lreps, opt, input_shape, output_shape, dropout )
