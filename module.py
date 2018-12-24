import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers
import numpy as np

# todo https://www.tensorflow.org/tutorials/keras/save_and_restore_models
# read a nice formed training pipeline


class model1(tf.keras.Model):
    def __init__(self, args):
        super(model1, self).__init__()
        self.base_model_name = args.base_model_name
        assert self.base_model_name in ['VGG16']
        self.BATCH_SIZE = args.BATCH_SIZE
        self.dataset_dir = args.dataset_dir
        self.checkpoint_dir = args.checkpoint_dir
        self.build_shape = args.build_shape
        # self.saver = tf.train.Saver()

        # contruct layers
        self.base_model = VGG16(weights='imagenet', include_top=False,
                                input_shape=(128, 128, 3))
        self.base_model.trainable = False
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(2, activation='softmax')

        # built model once layers constructed
        self._build_model()

    def call(self, inputs):
        base_model = self.base_model(inputs)
        base_model.trainable = False
        flatten = self.flatten(base_model)
        dense1 = self.dense1(flatten)
        dense2 = self.dense2(dense1)
        prediction = dense2
        return prediction

    def _build_model(self):
        self.build(self.build_shape)
        self.summary()
        self.compile(optimizer=tf.train.AdamOptimizer(),
                     loss=tf.keras.losses.sparse_categorical_crossentropy,
                     shuffle = True)
        print('compile done')

    def train(self, args):
        pass
        # todo rewrite the train function and test function
        '''do the data augmentation and training process'''
        optimizer = tf.train.AdamOptimizer(args.lr)
        '''
        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
        counter = 1
        for epoch in range(args.epoch):
            ds_091_L3, img_count = process_slided_img_to_tf(dataset_dir=args.dataset_dir,
                                                  levelNO=args.levelNO,
                                                  batch_size=args.BATCH_SIZE)
            history1 = model1.fit(ds_091_L3,
                                  steps_per_epoch=int(np.ceil(img_count / args.BATCH_SIZE)))

            counter += 1
            if np.mod(counter, args.save_freq) == 2:
                self.save(args.checkpoint_dir, counter)
        '''


        '''
        example code for training function
        for epoch in range(EPOCHS):
        start = time.time()
        
        hidden = encoder.initialize_hidden_state()
        total_loss = 0
        
        for (batch, (inp, targ)) in enumerate(dataset):
            loss = 0
            
            with tf.GradientTape() as tape:
                enc_output, enc_hidden = encoder(inp, hidden)
                
                dec_hidden = enc_hidden
                
                dec_input = tf.expand_dims([targ_lang.word2idx['<start>']] * BATCH_SIZE, 1)       
                
                # Teacher forcing - feeding the target as the next input
                for t in range(1, targ.shape[1]):
                    # passing enc_output to the decoder
                    predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
                    
                    loss += loss_function(targ[:, t], predictions)
                    
                    # using teacher forcing
                    dec_input = tf.expand_dims(targ[:, t], 1)
            
            batch_loss = (loss / int(targ.shape[1]))
            
            total_loss += batch_loss
            
            variables = encoder.variables + decoder.variables
            
            gradients = tape.gradient(loss, variables)
            
            optimizer.apply_gradients(zip(gradients, variables))
            
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
          checkpoint.save(file_prefix = checkpoint_prefix)
        
        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / N_BATCH))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
        
        '''


    def save(self, checkpoint_dir, step):
        pass
        '''
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
        '''

def modeling(training_data, img_count, args):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False

    model1 = models.Sequential()
    model1.add(base_model)
    model1.add(layers.Flatten())
    model1.add(layers.Dense(32, activation='relu'))
    model1.add(layers.Dense(2, activation='softmax'))
    model1.summary()

    model1.compile(optimizer=tf.train.AdamOptimizer(),
                   loss=tf.keras.losses.sparse_categorical_crossentropy,
                   metrics=['acc'])

    history1 = model1.fit(training_data, epochs=10, steps_per_epoch=int(np.ceil(img_count / args.BATCH_SIZE)))
    return history1


def create_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False

    model1 = models.Sequential()
    model1.add(base_model)
    model1.add(layers.Flatten())
    model1.add(layers.Dense(32, activation='relu'))
    model1.add(layers.Dense(2, activation='softmax'))
    # model1.summary()

    model1.compile(optimizer=tf.train.AdamOptimizer(),
                   loss=tf.keras.losses.sparse_categorical_crossentropy,
                   metrics=['acc'])

    return model1

def train_model(args, model, training_data, img_count):
    # include the epoch in the file name. (uses `str.format`)
    checkpoint_path = args.checkpoint_dir + '\\' + "cp-{epoch:04d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path, verbose=1, save_weights_only=True,
        period=5)

    history = model.fit(training_data, epochs=20, callbacks=[cp_callback],
                         steps_per_epoch=int(np.ceil(img_count / args.BATCH_SIZE)),
                         verbose=1)

    return history, model

def restore_model_from_latest_ckpt(args):
    model_test = create_model()
    try:
        latest = tf.train.latest_checkpoint(args.checkpoint_dir)
    except:
        print('checkpoint load failed, please train the model first, then run the test phase')
    try:
        model_test.load_weights(latest)
    except:
        print('model load failed, check the checkpointer file for more details')
    return model_test


