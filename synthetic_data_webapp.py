################## LOAD PACKAGES #############################
import streamlit as st
import pandas as pd
import numpy as np

import imblearn
from imblearn.over_sampling import SMOTE

import os
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras import Model

from tensorflow.keras.optimizers import Adam

################## END LOAD PACKAGES #############################################

################### HELPER FUNCTIONS ################################
def convert_df(df):
# IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

class GAN():
    
    def __init__(self, gan_args):
        [self.batch_size, lr, self.noise_dim, self.data_dim, layers_dim] = gan_args

        self.generator = Generator(self.batch_size).\
            build_model(input_shape=(self.noise_dim,), dim=layers_dim, data_dim=self.data_dim)

        self.discriminator = Discriminator(self.batch_size).\
            build_model(input_shape=(self.data_dim,), dim=layers_dim)

        optimizer = Adam(lr, 0.5)

        # Build and compile the discriminator
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.noise_dim,))
        record = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(record)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def get_data_batch(self, train, batch_size, seed=0):
        # # random sampling - optional
        # np.random.seed(seed)
        # x = train.loc[ np.random.choice(train.index, batch_size) ].values
        # iterate through shuffled indices, so every sample gets covered evenly

        start_i = (batch_size * seed) % len(train)
        stop_i = start_i + batch_size
        shuffle_seed = (batch_size * seed) // len(train)
        np.random.seed(shuffle_seed)
        train_ix = np.random.choice(list(train.index), replace=False, size=len(train))  
        train_ix = list(train_ix) + list(train_ix)  # duplicate to cover ranges past the end of the set
        x = train.loc[train_ix[start_i: stop_i]].values
        return np.reshape(x, (batch_size, -1))
        
    def train(self, data, train_arguments):
        [cache_prefix, epochs, sample_interval] = train_arguments
        
        data_cols = data.columns

        # Adversarial ground truths
        valid = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        for epoch in range(epochs):    
            # ---------------------
            #  Train Discriminator
            # ---------------------
            batch_data = self.get_data_batch(data, self.batch_size)
            noise = tf.random.normal((self.batch_size, self.noise_dim))

            # Generate a batch of new images
            gen_data = self.generator.predict(noise)
    
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(batch_data, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_data, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
            # ---------------------
            #  Train Generator
            # ---------------------
            noise = tf.random.normal((self.batch_size, self.noise_dim))
            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)
    
            # Plot the progress (optional)
            # print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
    
            # If at save interval => save generated events
            if epoch % sample_interval == 0:
                #Test here data generation step
                # save model checkpoints (Re-eval if needed)
                model_checkpoint_base_name = cache_prefix + '_{}_model_weights_step_{}.h5'
                self.generator.save_weights(model_checkpoint_base_name.format('generator', epoch))
                self.discriminator.save_weights(model_checkpoint_base_name.format('discriminator', epoch))

                #Here is generating the data
                z = tf.random.normal((432, self.noise_dim))
                gen_data = self.generator(z)
                print('generated_data')

    def save(self, path, name):
        assert os.path.isdir(path) == True, \
            "Please provide a valid path. Path must be a directory."
        model_path = os.path.join(path, name)
        self.generator.save_weights(model_path)  # Load the generator
        return
    
    def load(self, path):
        assert os.path.isdir(path) == True, \
            "Please provide a valid path. Path must be a directory."
        self.generator = Generator(self.batch_size)
        self.generator = self.generator.load_weights(path)
        return self.generator
    
class Generator():
    def __init__(self, batch_size):
        self.batch_size=batch_size
        
    def build_model(self, input_shape, dim, data_dim):
        input= Input(shape=input_shape, batch_size=self.batch_size)
        cnt = 1
        for i in range(gen_size): 
            x = Dense(dim*(cnt*2-2), activation='relu')(input)
            cnt+=1
        x = Dense(data_dim)(x)
        return Model(inputs=input, outputs=x)

class Discriminator():
    def __init__(self,batch_size):
        self.batch_size=batch_size
    
    def build_model(self, input_shape, dim):
        input = Input(shape=input_shape, batch_size=self.batch_size)
        cnt = dcrm_size
        for i in range(dcrm_size):
            x = Dense(dim*(2*cnt-2), activation='relu')(input)
            x = Dropout(0.1)(x)
            cnt-=1
            if cnt <= 0:
                break
        x = Dense(dim, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)
        return Model(inputs=input, outputs=x)


#################### END HELPER FUNCTIONS ##############################

#################### STREAMLIT APP #######################################

st.title("Synthetic Data Generator")

upload_file = st.file_uploader("Please upload an .xlsx file", type = 'xlsx')

if upload_file is not None:
    data_raw = pd.read_excel(upload_file, index_col = 0)
    data = data_raw.copy()
    
    with st.form(key='smote'):
        smote_submit = st.form_submit_button("Synthesize with SMOTE")
        
        k_neighbors= st.slider('Select K-Neighbors', 0, int(data.shape[0]/100), 5)
        number_new_obs = st.number_input('Number of Samples to Generate with SMOTE',
                                        min_value = 1,
                                        max_value = int(data.shape[0]))
        
        primary_col = st.selectbox(
            "Select Column with which to resample around.",
            options=data.columns.tolist(),
        )
        
    
    with st.form(key='gan'):
        gan_submit = st.form_submit_button("Synthesize with GAN")
        
        # Controls
        noise_dim = 32
        dim = 128
        batch_size = 32

        log_step = 100
        epochs = st.number_input('Number of Epochs to Train (Larger number will result in longer wait time)',
                                        min_value = 1,
                                        max_value = 5000,
                                value= 50)
        learning_rate = 5e-4
        samples_to_gen = st.number_input('Number of Samples to Generate with GAN',
                                        min_value = 1,
                                        max_value = int(data.shape[0]))

        ## Generator & Discrim Custom
        gen_size = st.slider('Select Generator Depth (Larger number will result in longer wait time)', 0, 5, 2)
        dcrm_size = st.slider('Select Discriminator Depth (Larger number will result in longer wait time)', 0, 5, 2)
        
    
    # SMOTE Section
    if smote_submit:
        number_new_obs = int(number_new_obs)
        # create artificial data with SMOTE
        oversample = SMOTE(k_neighbors=k_neighbors, sampling_strategy='all')
        X_train_smote, y_train_smote = oversample.fit_resample(data.drop(primary_col, axis = 1), data[primary_col])
        final_data = pd.concat([X_train_smote, y_train_smote], axis = 1)
        
        # check size, if request is more than we need to oversample again
        size_check = final_data.shape[0]- data.shape[0]

        # Continuously oversample until we meet number of new data points requested
        while size_check < number_new_obs:
            oversample = SMOTE(random_state= random_state, k_neighbors=k_neighbors, sampling_strategy='all')
            X_train_smote, y_train_smote = oversample.fit_resample(final_data.drop(primary_col, axis = 1), final_data[primary_col])
            int_data = pd.concat([X_train_smote, y_train_smote], axis = 1)
            final_data = pd.concat([final_data, int_data])
            size_check = final_data.shape[0]- data.shape[0]


        if size_check >= number_new_obs:
            final_data = final_data.iloc[:data.shape[0]+number_new_obs]
       
        csv = convert_df(final_data)
        
        name = upload_file.name.split('.')[0]
        st.download_button(
            label=f"Download Appended SMOTE Synthetic Data as CSV",
            data=csv,
            file_name=f'{name}_SMOTE.csv',
            mime='text/csv',
            )
        
    if gan_submit:
        epochs = int(epochs) + 1
        samples_to_gen = int(samples_to_gen0
        # If non-numerical data exists create hybrid vector column
        cat_cols = data_raw.loc[:,(data_raw.applymap(type) == str).all(0)].columns.tolist()
        if len(cat_cols) > 0:
            data_raw[cat_cols] = data_raw[cat_cols].astype('category') # [0,1]
            
        # Normalize Data for faster/better GAN fitting
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data_raw), columns = data_raw.columns.tolist(), index = data_raw.index.tolist())
        
        data_cols = data.columns
        
        #Define the GAN and training parameters
        data[data_cols] = data[data_cols]

        gan_args = [batch_size, learning_rate, noise_dim, data.shape[1], dim]
        train_args = ['', epochs, log_step]
        
        model = GAN

        #Training the GAN model chosen
        synthesizer = model(gan_args)
        synthesizer.train(data, train_args)
        
        syn_data = np.random.normal(size=(samples_to_gen, noise_dim))
        gen_data =  synthesizer.generator.predict(syn_data)
        
        final_data = pd.DataFrame(scaler.inverse_transform(gen_data), columns= data.columns.tolist())
        
        final_data = final_data.round(1)
        
        # Combine with original data
        final_data = pd.concat([data_raw, final_data], axis = 0)
        
        csv = convert_df(final_data)
        
        name = upload_file.name.split('.')[0]
        st.download_button(
            label=f"Download Appended GAN Synthetic Data as CSV",
            data=csv,
            file_name=f'{name}_GAN.csv',
            mime='text/csv',
            )
        
        
    
    
        
