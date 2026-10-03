# -*- coding: utf-8 -*-
"""
Attention-based multimodal fusion model for single fine-particle classification
================================================================================
Fuses TEM particle images (DenseNet169 backbone) with elemental composition and
particle size metadata (Random Forest) through self- and
cross-attention, and evaluates the fused model with five fold cross-validation.

This is the best-performing model reported in: "Advancing automatic classification of single fine particles
via multimodal deep learning framework", Scientific Reports.
================================================================================
"""

import argparse
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.densenet import DenseNet169, preprocess_input
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, Concatenate, Input, Dropout, Multiply, BatchNormalization,
                                     Layer, Reshape, Lambda, Conv2D, LayerNormalization, MultiHeadAttention)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.model_selection import train_test_split
from openpyxl import Workbook
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
import random
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from scipy.interpolate import interp1d
from sklearn.metrics import roc_auc_score
from tensorflow.keras.utils import plot_model
import time
import warnings
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import joblib


# =============================================================================
# ATTENTION-BASED MULTIMODAL FUSION 
# =============================================================================
# Pipeline, as described in the manuscript:
#   1. Numerical modality  : particle size + 45-element composition  ==> Random Forest#                            
#   2. Morphological modality: TEM image ==> DenseNet169 (fully fine-tuned) ==> GAP ==> Dense(64)
#   3. Fusion              : self-attention per modality, bidirectional multi-head
#                            cross-attention (8 heads), LayerNorm, concat,
#                            Dense(64, ReLU) + Dropout, softmax over 7 particle types
#   4. Evaluation          : stratified 5fold CV; SMOTE + image augmentation applied to
#                            the training split only; early stopping
# =============================================================================


# Hyperparameters used for the results reported in the manuscript
# Learning rate, batch size and dropout were tuned to minimise validation loss

img_height = 227 ####change
img_width = 227####change
img_channels = 3####change
batch_size = 16####change # Try: 8, 32, 64
epochs = 70####change # Try: 50, 100, 150
num_folds = 5
learning_rate = 1e-4 ####change   # Try: 1e-3, 1e-5, 5e-4
decay_rate = learning_rate // epochs####change
input_size = (227, 227, 3)####change
ratio = 1  ####change  # SMOTE ratio: 1 = oversample every minority class up to the majority-class count

################################## VARIABLES (per-fold result containers) ########################################################
#to store traning time
timetaken=[]


#Train test
test_accuracies=[]
val_accuracies=[]
train_accuracies =[]
test_losses =[]
val_losses =[]
train_losses =[]

fold_data = {}
classification_reports = []
confusion_matrices = []
sensitivity_all = []
specificity_all = []
TNR_all = []
PPV_all = []
NPV_all = []
FPR_all = []
FNR_all = []
FDR_all = []

all_true_labels = []
all_predicted_probs = []
fff = []
ttt = []
aucc = []
all_fpr = []
all_tpr = []
all_roc_auc = []
all_true_classes = []
all_predicted_classes = []


fold_train_accuracies =[]
fold_train_losses =[]
fold_val_accuracies =[]
fold_val_losses=[]
all_fold_results = []


# ------------------------------------------------------------------ paths ----
parser = argparse.ArgumentParser(description='Attention-based multimodal fusion model (5-fold CV).')
parser.add_argument('--metadata', required=True,
                    help='Excel file with elemental fractions, diameter_um, Final_particle_type and Image Path columns')
parser.add_argument('--output', default='results',
                    help='Directory where models, figures and metrics are written (default: results/)')
args = parser.parse_args()

output = args.output
os.makedirs(output, exist_ok=True)
df = pd.read_excel(args.metadata, engine='openpyxl')

# Elemental columns from EDX (wt%). 46 columns are listed including Cu; Cu is removed below because
# it originates from the Cu TEM grid (Methods: "Elemental composition"), leaving the 45 elements used in the paper.

columns_of_interest = ['Ag', 'Al', 'Ar', 'As', 'Au','Ba', 'Br', 'C', 'Ca', 'Cd', 'Cl', 'Co', 'Cr','Cu', 'F', 'Fe', 
                      'Ga', 'Hg', 'In', 'K', 'Mg', 'Mn', 'Mo', 'N', 'Na', 'Ni', 'O', 'Os', 'P', 'Pb', 
                      'Pd', 'Rb', 'Rh', 'S', 'Sb', 'Sc', 'Si', 'Sn', 'Sr', 'Te', 'Ti', 'V', 'W', 'Y', 'Zn', 'Zr'] #45 withoiut cu #46 with cu

# Re-normalise the 45 non-Cu elements so that each particle's composition sums to 100 wt%, then convert to fractions (0-1)
elements_wo_cu = [elem for elem in columns_of_interest if elem != 'Cu']
df[elements_wo_cu] = df[elements_wo_cu].div(df[elements_wo_cu].sum(axis=1), axis=0) * 100

if 'Cu' in df.columns:
    df = df.drop(columns=['Cu'])

df_numeric = df[elements_wo_cu]
df_normalized_fraction = df_numeric / 100
df_normalized_fraction = df_normalized_fraction.fillna(0)

numeric_features = df_normalized_fraction .values

# Particle diameter (um, from TEM scale bar; Methods: "Particle size") scaled to [0, 1] with min-max scaling
label_encoder = LabelEncoder()
diameter_scaler = MinMaxScaler()
diameter_values = df['diameter_um'].values.reshape(-1, 1)
labels_particlesize = diameter_scaler.fit_transform(diameter_values)

# Ground-truth labels: the seven particle types (dust, fly ash, metals, organic-rich, sea spray, soot, sulfur-rich), one-hot encoded
labels_particletype = to_categorical(label_encoder.fit_transform(df['Final_particle_type']))

# Numerical modality = elemental fractions (45) + scaled diameter (1)
metadata_features = np.hstack([numeric_features, labels_particlesize])#use both or either to checke the diff in performance

# image paths
image_paths = df['Image Path'].values


########################################################################################################################################
################################## Feature extraction using RF ##################################################################
# Methods: "the Random Forest (RF) model was utilized to derive key features from the particle size and elemental composition".
# RF was the best unimodal model on the numerical data (Fig. 2). Its 7 class probabilities are linearly interpolated to a
# 64-dimensional vector so the numerical branch has the same width as the 64-unit image embedding before attention fusion.
def extract_rf_features(train_metadata, model):
    rf_preds = model.predict_proba(train_metadata)
    interpolated = np.zeros((rf_preds.shape[0], 64))
    for i in range(rf_preds.shape[0]):
        interpolated[i] = np.interp(
            np.linspace(0, 1, 64),
            np.linspace(0, 1, rf_preds.shape[1]),
            rf_preds[i]
        )
    return interpolated.astype(np.float32)


# Custom data generator yielding ([image_batch, numerical_features], labels).
# Image augmentation (Methods: "Data partitioning and augmentation"): random rotation up to 180 deg, horizontal and
# vertical flips, zoom 0.2. Applied ONLY when is_training=True, i.e. only to the training split of each fold.
def custom_data_generator(image_paths, metadata_features, labels, batch_size, is_training=True): #If is_training is False, it creates a ImageDataGenerator without augmentation, meaning it will only rescale images.
    data_gen = ImageDataGenerator(
        zoom_range=0.2,#####change 
        horizontal_flip=True,#####change 
        vertical_flip=True, #####change 
        rotation_range=180,#####change 
    ) if is_training else ImageDataGenerator()

    while True:
        if is_training:
            indices = np.random.permutation(len(image_paths)) #shuffles the indices of the images randomly
        else:
            indices = np.arange(len(image_paths)) #keeps the indices in sequential orde

        for start in range(0, len(image_paths), batch_size):
            end = min(start + batch_size, len(image_paths))
            batch_indices = indices[start:end]

            batch_images = []
            batch_metadata = metadata_features[batch_indices]
            batch_labels = labels[batch_indices]

            for i in batch_indices:
                image = cv2.imread(image_paths[i])

                # skip unreadable images
                if image is None:
                    print(f"Error: Could not load image at path: {image_paths[i]}")
                    continue  # skip to the next image in batch

                image = cv2.resize(image, (img_width, img_height))
                image = image / 255.0 #normalizes the image pixel values to from 0 and 1
                batch_images.append(image)

            batch_images = np.array(batch_images)
            augmented_batch_images = next(data_gen.flow(batch_images, batch_size=batch_size, shuffle=False))

            yield [augmented_batch_images, batch_metadata], batch_labels

# count the number of samples for each final particle type
particle_type_counts = np.sum(labels_particletype, axis=0)

#  maximum count in particle type for balacing
#max_count = int(np.max(particle_type_counts))  #change accrodingly

################################## SMOTE ########################################################################################
# Methods: "SMOTE was utilized to balance the numerical modalities (particle size and elemental composition)".
# Synthetic samples are generated in the numerical feature space within each particle class; applied to the training split only.
def apply_smote(X, y, ratio=1, k=5):  #1 = match the majority class)1 = match the majority class)
    
    y_labels = np.argmax(y, axis=1)
    unique, counts = np.unique(y_labels, return_counts=True)
    majority_count = max(counts)


    target_counts = {
        cls: int(majority_count * ratio) if count < int(majority_count * ratio) else count
        for cls, count in zip(unique, counts)
    }

    smote = SMOTE(sampling_strategy=target_counts, k_neighbors=k, random_state=42)
    X_res, y_res_labels = smote.fit_resample(X, y_labels)
    
    # Reconvert to one-hot
    num_classes = y.shape[1]
    y_res = np.eye(num_classes)[y_res_labels]
    return X_res, y_res 

# Build the balanced multimodal training set. Each synthetic numerical sample produced by SMOTE is paired with a TEM image
# randomly drawn from the training particles of the same class; because image augmentation is applied stochastically
# in the data generator at every epoch, repeated draws of the same image yield different augmented views.
def augment_data(image_paths, metadata_features, labels, ratio=1.0):
  
    #0ne-hot encoded labels to class indices
    label_indices = np.argmax(labels, axis=1)
    
    #SMOTE
    metadata_features_resampled, labels_resampled = apply_smote(metadata_features, labels, ratio=ratio)
    label_indices_resampled = np.argmax(labels_resampled, axis=1)
    # Convert resampled indices back to one-hot encoding
    labels_resampled = to_categorical(label_indices_resampled, num_classes=labels.shape[1])
    
    # Handle image paths for synthetic samples
    augmented_image_paths = []
    original_indices = {}
    
    # Create a mapping of original samples for each class
    for i in range(len(label_indices)):
        class_idx = label_indices[i]
        if class_idx not in original_indices:
            original_indices[class_idx] = []
        original_indices[class_idx].append(i)
    
    # For each resampled instance, either use original path or select from same class
    for i, class_idx in enumerate(label_indices_resampled):
        if i < len(image_paths):  # Original sample
            augmented_image_paths.append(image_paths[i])
        else:  # Synthetic sample
            # Select a random image path from the same class
            original_idx = np.random.choice(original_indices[class_idx])
            augmented_image_paths.append(image_paths[original_idx])
    
    augmented_image_paths = np.array(augmented_image_paths)
    
    return augmented_image_paths, metadata_features_resampled, labels_resampled


##################################  MODEL ARCHITECTURE (Fig. 7, attention-based fusion) ##########################################

def create_model(input_shape, metadata_shape, num_classes):
  
    # --- Morphological branch: ImageNet-pretrained DenseNet169 (best unimodal CNN, Fig. 2), all layers fine-tuned ---
    img_input = Input(shape=input_shape, name="image_input")
    base_model = DenseNet169(weights='imagenet', include_top=False, input_tensor=img_input)

    for layer in base_model.layers:
        layer.trainable = True
    img_features = base_model.output
    img_features_flattened = GlobalAveragePooling2D()(img_features)
    img_features_flattened = Dense(64, trainable=True)(img_features_flattened)
    img_features_flattened = Dropout(0.4, trainable=True)(img_features_flattened)
    img_features_reshaped = Reshape((1, 1, img_features_flattened.shape[1]))(img_features_flattened)

    # --- Numerical branch: 64-dim RF probability features (size + elemental composition) ---
    metadata_input = Input(shape=(metadata_shape,), name="metadata_input")  # shape=(64,)
    metadata_input_reshaped = Reshape((1, 1, metadata_input.shape[1]))(metadata_input)
    #metadata_features_seq = Reshape((1, 64))(metadata_input)

    # --- Self-attention within each modality ("prioritise its most informative intra-modal patterns") ---
    img_attention = tf.keras.layers.Attention()([img_features_reshaped, img_features_reshaped])
    metadata_attention = tf.keras.layers.Attention()([metadata_input_reshaped, metadata_input_reshaped])

    # --- Bidirectional multi-head cross-attention, 8 heads ("empirically optimized to eight") ---
    attention_image_to_metadata = MultiHeadAttention(num_heads=8, key_dim=img_features_flattened.shape[1])(
        img_attention, metadata_attention )
    attention_metadata_to_image = MultiHeadAttention(num_heads=8, key_dim=metadata_input.shape[1])(
        metadata_attention , img_attention)

    # --- Residual concatenation with the original features + layer normalisation ---
    norm_metadata = tf.keras.layers.LayerNormalization()(tf.keras.layers.Concatenate()([attention_image_to_metadata, metadata_input_reshaped])) 
    norm_image = tf.keras.layers.LayerNormalization()(tf.keras.layers.Concatenate()([attention_metadata_to_image, img_features_reshaped]))

    # --- Fuse both attended modalities ---
    concatenated = tf.keras.layers.Concatenate()([norm_metadata, norm_image]) #try with Mulitply cocnatenate too # shape = (batch, 2, features)
    concatenated_flattened =  tf.keras.layers.Flatten()(concatenated)

    # Classification head: Dense(64, ReLU) + Dropout, then softmax over the 7 particle types
    x = Dense(64, activation='relu', trainable = True)(concatenated_flattened)
    x = Dropout(0.4, trainable = True)(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=[img_input, metadata_input], outputs=output)
    return model


################################## CROSS-VALIDATION ##############################################################################
# Methods: "a stratified five-fold cross-validation approach was employed. During each iteration, one fold (20%) was reserved
# for testing, while the remaining four folds were utilized for training and validation (allocating 20% of this training
# subset specifically for validation)."
kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=123)
class_indices_all = np.argmax(labels_particletype, axis=1)
input_shape = (img_height, img_width, img_channels)
#metadata_shape = metadata_features.shape[1]
metadata_shape = 64
num_classes = labels_particletype.shape[1]
fold = 0

# Record the exact train / validation / test membership of every particle per fold (to verify there is no data leakage)
wb_cv = Workbook()
ws_cv = wb_cv.active
ws_cv.title = "Cross-Validation Splits"
#morph_class_org = np.unique(df['morph_class'])
#particle_size_org = np.unique(df['particle_size'])
particle_type_org = np.unique(df['Final_particle_type'])
ws_cv.append(["Fold", "Type", "Image Path", *columns_of_interest,   *particle_type_org])
ws_cv_augmented = wb_cv.create_sheet(title="Augmented Data")
ws_cv_augmented.append(["Fold", "Type", "Image Path", *columns_of_interest,   *particle_type_org])


all_fpr = {i: [] for i in range(num_classes)}
all_tpr = {i: [] for i in range(num_classes)}
all_roc_auc = {i: [] for i in range(num_classes)}

for train_index, test_index in kf.split(image_paths, class_indices_all):
    fold += 1
    print('-------------------------------------------------------------------------------------')
    print(f'--------------------Processing fold {fold}---------------------------------------')
    print('-------------------------------------------------------------------------------------')

    train_image_paths, test_image_paths = image_paths[train_index], image_paths[test_index]
    train_metadata_features, test_metadata_features = metadata_features[train_index], metadata_features[test_index]
    train_labels, test_labels = labels_particletype[train_index], labels_particletype[test_index]

    train_image_paths, val_image_paths, train_metadata_features, val_metadata_features, train_labels, val_labels = \
        train_test_split(train_image_paths, train_metadata_features, train_labels, test_size=0.2, random_state=123,
                         stratify=np.argmax(train_labels, axis=1))

    # Balance the TRAINING split only (SMOTE on numerical data + paired images); validation and test remain untouched
    train_image_paths_aug, train_metadata_features_aug, train_labels_aug = augment_data(
        train_image_paths, train_metadata_features, train_labels, ratio)

    # Store the info on train, test, and validation data
    for img_path, metadata_info, type_label in zip(train_image_paths, train_metadata_features, train_labels):
        ws_cv.append([fold, "train", img_path, *metadata_info, *type_label])

    for img_path, metadata_info, type_label in zip(val_image_paths, val_metadata_features, val_labels):
        ws_cv.append([fold, "val", img_path, *metadata_info, *type_label])

    for img_path, metadata_info, type_label in zip(test_image_paths, test_metadata_features, test_labels):
        ws_cv.append([fold, "test", img_path, *metadata_info, *type_label])

    # save augmented training data information
    for img_path, metadata_info, type_label in zip(train_image_paths_aug, train_metadata_features_aug, train_labels_aug):
        ws_cv_augmented.append([fold, "augmented", img_path, *metadata_info, *type_label])

    # shuffle the augmented data
    shuffle_indices = np.random.permutation(len(train_image_paths_aug))
    train_image_paths_aug = train_image_paths_aug[shuffle_indices]
    train_metadata_features_aug = train_metadata_features_aug[shuffle_indices]
    train_labels_aug = train_labels_aug[shuffle_indices]


    # Fit the Random Forest on the balanced training metadata of this fold (RF hyperparameters: Table S2)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1)
    rf_model.fit(train_metadata_features_aug, np.argmax(train_labels_aug, axis=1))
    # Save the model
    output_dir = output

    rf_model_filename = f'random_forest_model_Fold{fold}.joblib'
    rf_model_path = os.path.join(output_dir, rf_model_filename)
    # Save the model using joblib
    joblib.dump(rf_model, rf_model_path)

    # Convert numerical data of all three splits to 64-dim RF probability features (the RF never sees validation/test labels)
    rf_train_features = extract_rf_features(train_metadata_features_aug, rf_model)
    rf_val_features = extract_rf_features(val_metadata_features, rf_model)
    rf_test_features = extract_rf_features(test_metadata_features, rf_model)
    # Create data generators with RF features as metadata input
    train_generator = custom_data_generator(train_image_paths_aug, rf_train_features, train_labels_aug, batch_size)
    val_generator = custom_data_generator(val_image_paths, rf_val_features, val_labels, batch_size, is_training=False)
    test_generator = custom_data_generator(test_image_paths, rf_test_features, test_labels, batch_size, is_training=False)

        
    # Training time per fold (reported in Table S3)
    start = time.time() 
    
    
    #current_metadata_shape = train_metadata_features_aug.shape[1]   #updated
    
    # Build the attention-based fusion model
    model = create_model(input_shape, metadata_shape, num_classes)
    
    checkpoint_filename = f'checkpoint_model_{epochs}_{batch_size}_{learning_rate}__Fold{fold}.h5'
    checkpoint_path = os.path.join(output, checkpoint_filename)
    checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        verbose=1
    )

    # Methods: "an early stopping mechanism with a patience of 10 epochs was implemented"
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    # Methods: Adam optimiser with categorical cross-entropy loss. Note: decay_rate evaluates to 0 (integer division), so no LR decay is applied.
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=decay_rate)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    weights_filename = f'best_weight_{epochs}_{batch_size}_{learning_rate}__Fold{fold}.h5'
    weights_filepath = os.path.join(output, weights_filename)
    model.save_weights(weights_filepath)

    model_filename = f'best_model_{epochs}_{batch_size}_{learning_rate}__Fold{fold}.h5'
    model.save(os.path.join(output, model_filename))

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_image_paths_aug) // batch_size,
        validation_data=val_generator,
        validation_steps=len(val_image_paths) // batch_size,
        epochs=epochs,
        callbacks=[checkpoint_callback, early_stopping]
    )
    
        
    # Training / validation accuracy and loss curves per fold
    plt.figure(figsize=(12, 9))
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')    
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.title(f'Training and validation accuracy curves (Fold {fold})')
    loss_fig = os.path.join(output, f'accuracy_{epochs}_{batch_size}_{learning_rate}_Fold_{fold}.png')
    plt.savefig(loss_fig, dpi=150)  
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 9))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')    
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.title(f'Training and validation loss curves (Fold {fold})')
    loss_fig = os.path.join(output, f'loss_{epochs}_{batch_size}_{learning_rate}_Fold_{fold}.png')
    plt.savefig(loss_fig,  dpi=150)  
    plt.tight_layout()
    plt.show()
    
    
    #saving the values of training loss acc and validation loss acc in excel
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']
    training_accuracy = history.history['accuracy']
    validation_accuracy = history.history['val_accuracy']


    # Evaluate on the held-out test fold (overall accuracy reported in Fig. 3)
    test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_image_paths) // batch_size)
    print(f'Fold {fold} - Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')   
    
    test_accuracies.append(test_accuracy)
    test_losses.append(test_loss)
    
    val_loss, val_accuracy = model.evaluate(val_generator, steps=len(val_image_paths) // batch_size)
    print(f'Fold {fold} - val Loss: {val_loss}, val Accuracy: {val_accuracy}')

    val_accuracies.append(val_accuracy)
    val_losses.append(val_loss)

    
    train_loss, train_accuracy = model.evaluate(train_generator, steps=len(train_image_paths) // batch_size)
    train_accuracies.append(train_accuracy)
    train_losses.append(train_loss)  


     #Store history of each fold
    fold_train_loss = history.history['loss']
    fold_train_acc = history.history['accuracy']
    fold_val_loss = history.history['val_loss']
    fold_val_acc = history.history['val_accuracy']
    # Determine the number of epochs actually trained
    num_epochs_trained = len(fold_train_loss)
    for epoch in range(num_epochs_trained):
        print(
            f"Epoch {epoch + 1}: Train Loss = {fold_train_loss[epoch]:.4f}, Train Acc = {fold_train_acc[epoch]:.4f}, Val Loss = {fold_val_loss[epoch]:.4f}, Val Acc = {fold_val_acc[epoch]:.4f}")

    # Append fold training accuracy and loss to respective lists
    fold_train_accuracies.append(history.history['accuracy'])
    fold_train_losses.append(history.history['loss'])

    # Append fold validation accuracy and loss to respective lists
    fold_val_accuracies.append(history.history['val_accuracy'])
    fold_val_losses.append(history.history['val_loss'])
    
     
    fold_results = {
        'Fold': [fold] * num_epochs_trained ,
        'Epoch': list(range(1, num_epochs_trained  + 1)),
        'Train Loss': history.history['loss'],
        'Train Accuracy': history.history['accuracy'],
        'Validation Loss': history.history['val_loss'],
        'Validation Accuracy': history.history['val_accuracy']
    }
    
    all_fold_results.append(pd.DataFrame(fold_results))
    
    test_generator_full = custom_data_generator(test_image_paths, rf_test_features, test_labels, len(test_image_paths), is_training=False)
    [test_images, test_metadata], test_labels = next(test_generator_full)
    predictions = model.predict([test_images, test_metadata])
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels, axis=1)
    class_labels = label_encoder.classes_


    # Confusion matrix for this fold (counts and row-normalised %), Methods: "Performance evaluation"
    
    #not normalize        
    cm = confusion_matrix(true_classes, predicted_classes, labels=np.arange(len(class_labels)))    
    fig, ax = plt.subplots(figsize=(16, 12))
    cax = ax.matshow(np.ones_like(cm), cmap="gray", interpolation="nearest", vmin=0, vmax=1)
    cbar = fig.colorbar(cax)
    cbar.ax.tick_params(labelsize=30, which='major', width=2.5, length=5)
    cbar.set_label("Percentage within class", fontsize=32, fontname="Arial")   
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels, fontsize=28) #23.5
    ax.set_yticklabels(class_labels, fontsize=28)    #23.5    
    ax.set_xlabel("Predicted labels", fontsize=32, fontname="Arial")
    ax.set_ylabel("True labels", fontsize=32, fontname="Arial") 
    
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=34)    
    
    confusion_fig = os.path.join(output, f'Confusionmatrix_{epochs}_{batch_size}_{learning_rate}__earlystopping_10_earlystopping_10_Fold{fold}.png')
   
    plt.savefig(confusion_fig, dpi = 150)
    plt.tight_layout()
    plt.show()


    #normalize
    list(class_labels) #check to see the sequence for labelling the axis label
    #class_labels = ['Flyash', 'Metal', 'S-rich', 'Sea spray', 'Soot']
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  #https://scikit-learn.org/0.20/auto_examples/model_selection/plot_confusion_matrix.html
    cm_norm = np.round(np.array(cm_norm), 1) #with decimal point 1        
    fig, ax = plt.subplots(figsize=(16, 12))
    cax = ax.matshow(cm_norm , cmap='gray_r', vmin=0, vmax=100)
    cbar = fig.colorbar(cax)  #https://matplotlib.org/stable/users/explain/colors/colorbar_only.html
    cbar.ax.tick_params(labelsize=30, which='major', width=2.5, length = 5)
    cbar.set_label("Percentage within class", fontsize=32, fontname="Arial") 
    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels, fontsize=28) #23.5
    ax.set_yticklabels(class_labels, fontsize=28)    #23.5    
    ax.set_xlabel("Predicted labels", fontsize=32, fontname="Arial")
    ax.set_ylabel("True labels", fontsize=32, fontname="Arial") 
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            value = f"{cm_norm[i, j]}%"
            color = "white" if cm_norm[i, j] > 80 else "black"
            ax.text(j, i, value, ha="center", va="center", color=color, fontsize=34, fontname="Arial")
    
    confusion_fig_norm = os.path.join(output, f'Confusionmatrix_{epochs}_{batch_size}_{learning_rate}__earlystopping_10_earlystopping_10_Fold{fold}_normalized.png')
    plt.tight_layout()
    plt.savefig(confusion_fig_norm, dpi = 150)
    
    plt.show()
    
    confusion_matrices.append(cm_norm) #concatenating all the data of each folds 
#   


    # One-vs-rest ROC curves and AUC per particle type
    #calling again few variables 
    
    label_encoder = LabelEncoder()
    label_encoder.fit(df['Final_particle_type'])
    label_mapping = {label: label_encoder.inverse_transform([label])[0] for label in range(num_classes)}
    
    test_generator_full = custom_data_generator(test_image_paths, rf_test_features, test_labels, len(test_image_paths), is_training=False)
    [test_images, test_metadata], test_labels = next(test_generator_full)
    predictions = model.predict([test_images, test_metadata])
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(test_labels, axis=1)
    class_labels = label_encoder.classes_

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve((true_classes == i).astype(int), predictions[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        mean_fpr = np.linspace(0, 1, 100)  # Fixed number of points for interpolation
        interp_tpr = interp1d(fpr[i], tpr[i], kind='linear', fill_value="extrapolate")
        mean_tpr = interp_tpr(mean_fpr)
        mean_tpr[0] = 0.0  # Start from 0
        all_fpr[i].append(mean_fpr)
        all_tpr[i].append(mean_tpr)
        all_roc_auc[i].append(roc_auc[i])

    all_true_classes.append(true_classes)
    all_predicted_classes.append(predicted_classes)
    
  
    plt.figure(figsize=(15, 12))
    marker_styles = ['o', 's', '^', 'X', "D", '*', 'v', 'H']
    skip_points = 5  # Number of points to skip between markers
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], lw=2, color='black', label='{0} (AUC = {1:.2f})'.format(label_mapping[i], roc_auc[i]), marker=marker_styles[i], markersize=19, markevery=skip_points)

    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.xlabel('False Positive Rate', fontsize=30)
    plt.ylabel('True Positive Rate', fontsize=30)
    plt.title('ROC-AUC Curve')
    plt.legend(loc="lower right", fontsize=30, markerscale=1.3)
    roc_fig = os.path.join(output, f'Roc_{epochs}_{batch_size}_{learning_rate}__earlystopping_10_earlystopping_10_Fold{fold}_normal.png')
   
    plt.savefig(roc_fig, dpi = 150)
    plt.show()
    
    
    # Per-class precision / recall / F1
    predicted_labels = [label_mapping[class_index] for class_index in predicted_classes]
    true_labels = [label_mapping[class_index] for class_index in true_classes]
    report = classification_report(true_labels, predicted_labels ) #output_dict = True
    print(report)
    txt = f'classification_report_fold{fold}.txt'
    class_report_path = os.path.join(output, txt)
    with open(class_report_path, 'w') as file:
        file.write(str(report))


    # Sensitivity (true positive rate) and specificity (true negative rate) per class, as reported in Fig. 3
    tp = np.diag(cm)
    fp = np.sum(cm, axis=0) - np.diag(cm)
    fn = np.sum(cm, axis=1) - np.diag(cm)
    tn = np.sum(cm) - (tp + fp + fn)     
    sensitivity = tp/ (tp+fn) #similar to recall and TPR #better check the result with recall from classification report for valdiation
    specificity = tn/ (tn+fp)   
    TNR = tn/(tn+fp)
    PPV = tp/(tp+fp)
    NPV = tn/(tn+fn)
    FPR = fp/(fp+tn)
    FNR = fn/(tp+fn)
    FDR = fp/(tp+fp) 
    
 
    txt = f'Specificty_senstivity_fold{fold}.txt'
    report_path = os.path.join(output, txt)    
    for i in range(len(class_labels)):
        print(f"{class_labels[i]} - sensitivity: {sensitivity[i]:.2f}, specificity: {specificity[i]:.2f}, TNR: {TNR[i]:.2f}, PPV : {PPV[i]:.2f}, NPV: {NPV[i]:.2f}, FPR: {FPR[i]:.2f}, FNR: {TNR[i]:.2f}, FDR: {FDR[i]:.2f}")        
        with open(report_path, "w") as file: #https://docs.python.org/2/library/functions.html#open
            for i in range(len(class_labels)):
                file.write(f"{class_labels[i]} - Sensitivity: {sensitivity[i]:.2f}, Specificity: {specificity[i]:.2f}, TNR: {TNR[i]:.2f}, PPV: {PPV[i]:.2f}, NPV: {NPV[i]:.2f}, FPR: {FPR[i]:.2f}, FNR: {FNR[i]:.2f}, FDR: {FDR[i]:.2f}\n")


    end = stop = time.time()
    
    trainingtime = f'{stop - start}s'
    
    timetaken.append(trainingtime)


################################## SAVE RESULTS (aggregated over the 5 folds) ####################################################

cross_validation_splits = os.path.join(output, 'cross_validation_splits.xlsx' )
wb_cv.save(cross_validation_splits)   ####save only image infor

Cross_Validation_Splits_Augmented = os.path.join(output, 'cross_validation_splits_augumented.xlsx' )
wb_cv.save(Cross_Validation_Splits_Augmented )


#save the traning time for each fold
#excel_writer_test.save()
all_traningtime = pd.DataFrame(timetaken)
save_time = os.path.join(output, f'Trainingtime_{epochs}_{batch_size}_{learning_rate}__earlystopping_10_.xlsx')
all_traningtime.to_excel(save_time, index = False)


# Calculate and print average scores for all folds and save
with open(os.path.join(output, f'scores_{epochs}_{batch_size}_{learning_rate}_earlystopping_10_.txt'), 'w') as file:
    file.write('------------------------------------------------------------------------\n')
    file.write('Score per fold\n')
    for i in range(0, len(test_accuracies)):
        file.write('--------------------------------------------------------------\n')
        file.write(
            f'>Fold {i+1} - Test Loss : {test_losses[i]} - Test Accuracy : {test_accuracies[i]}% - val Loss : {val_losses[i]} - val Accuracy : {val_accuracies[i]}% - Train Loss : {train_losses[i]} - Train Accuracy : {train_accuracies[i]}%\n')

    file.write('------------------------------------------------------------------------\n')
    average_accuracy = np.mean(test_accuracies)
    average_loss = np.mean(test_losses)
    average_val_accuracy = np.mean(val_accuracies)
    average_val_loss = np.mean(val_losses)
    average_train_accuracy = np.mean(train_accuracies)
    average_train_loss = np.mean(train_losses)

    file.write("Average scores for all folds:\n")
    file.write(
        f"> Test Accuracy: {average_accuracy * 100:.2f}% (+- {np.std(test_accuracies) * 100:.2f}%)\n")
    file.write(f"> Loss: {average_loss:.6f}\n")

    file.write(
        f"> val Accuracy: {average_val_accuracy * 100:.2f}% (+- {np.std(val_accuracies) * 100:.2f}%)\n")
    file.write(f"> Loss: {average_val_loss:.6f}\n")

    file.write(
        f"> Train Accuracy: {average_train_accuracy * 100:.2f}% (+- {np.std(train_accuracies) * 100:.2f}%)\n")
    file.write(f"> Loss: {average_train_loss:.6f}\n")

##Display

print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(test_accuracies)):
    print('--------------------------------------------------------------')
    print(
        f'>Fold {i+1} - Test Loss : {test_losses[i]} - Test Accuracy : {test_accuracies[i]}% - val Loss : {val_losses[i]} - val Accuracy : {val_accuracies[i]}% - Train Loss : {train_losses[i]} - Train Accuracy : {train_accuracies[i]}% ')

print('------------------------------------------------------------------------')

print("Average scores for all folds:")
print(
    f"> Test Accuracy: {average_accuracy * 100:.2f}% (+- {np.std(test_accuracies) * 100:.2f}%)")
print(f"> Loss: {average_loss:.6f}")

print(
    f"> val Accuracy: {average_val_accuracy * 100:.2f}% (+- {np.std(val_accuracies) * 100:.2f}%)")
print(f"> Loss: {average_val_loss:.6f}")

print(
    f"> Train Accuracy: {average_train_accuracy * 100:.2f}% (+- {np.std(train_accuracies) * 100:.2f}%)")
print(f"> Loss: {average_train_loss:.6f}")   


# Fold-averaged, row-normalised confusion matrix (Fig. 4)
total_cm = sum(confusion_matrices)
total_cm_df = pd.DataFrame(total_cm, columns=class_labels, index=class_labels)
average_cm = total_cm / num_folds
average_cm = np.round(np.array(average_cm ), 1)
average_cm_df = pd.DataFrame(average_cm, columns=class_labels, index=class_labels)

fig, ax = plt.subplots(figsize=(16, 12))
cax = ax.matshow(average_cm , cmap='gray_r', vmin=0, vmax=100)
cbar = fig.colorbar(cax)  #https://matplotlib.org/stable/users/explain/colors/colorbar_only.html
cbar.ax.tick_params(labelsize=30, which='major', width=2.5, length = 5)
cbar.set_label("Percentage within class", fontsize=32, fontname="Arial") 
ax.set_xticks(np.arange(len(class_labels)))
ax.set_yticks(np.arange(len(class_labels)))
ax.set_xticklabels(class_labels, fontsize=28) #23.5
ax.set_yticklabels(class_labels, fontsize=28)    #23.5
ax.set_xlabel("Predicted labels", fontsize=32, fontname="Arial")
ax.set_ylabel("True labels", fontsize=32, fontname="Arial")
for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        value = f"{average_cm[i, j]}%"
        color = "white" if average_cm[i, j] > 80 else "black"
        ax.text(j, i, value, ha="center", va="center", color=color, fontsize=34, fontname="Arial")

confusion_fig_avg = os.path.join(output, f'Confusionmatrix_{epochs}_{batch_size}_{learning_rate}__earlystopping_10_earlystopping_10_Fold_avg.png')

plt.tight_layout()
plt.savefig(confusion_fig_avg, dpi = 150)
plt.show()

avg_fpr = {}
avg_tpr = {}
avg_roc_auc = {}


for i in range(num_classes):
    avg_fpr[i] = np.mean(all_fpr[i], axis=0)
    avg_tpr[i] = np.mean(all_tpr[i], axis=0)
    avg_roc_auc[i] = np.mean(all_roc_auc[i])


plt.figure(figsize=(15, 12))
marker_styles = ['o', 's', '^', 'X', "D", '*', 'v', 'H']
skip_points = 5  # Number of points to skip between markers

for idx, i in enumerate(range(num_classes)):
    plt.plot(avg_fpr[i], avg_tpr[i], color='black', lw=2, marker=marker_styles[i],  markersize=19,
             label=' %s (AUC = %0.2f)' % (label_mapping[i], avg_roc_auc[i]), markevery=skip_points)

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.xlabel('False Positive Rate', fontsize=30)
plt.ylabel('True Positive Rate', fontsize=30)
plt.title('ROC-AUC Curve')
plt.legend(loc="lower right", fontsize=30, markerscale=1.3)
roc_fig_avg = os.path.join(output, f'AvgRoc_{epochs}_{batch_size}_{learning_rate}__earlystopping_10_earlystopping_10_Fold{fold}_normal.png')
   
plt.savefig(roc_fig_avg, dpi = 150)
plt.show()


##Save the history of loss and accuracy of each fold#############
final_results_df = pd.concat(all_fold_results, ignore_index=True)

# Save the combined results to an Excel file
results_filename = os.path.join(output, f'loss_acc_curve_{epochs}_{batch_size}_{learning_rate}__earlystopping_10_.xlsx')
final_results_df.to_excel(results_filename, index=False)

print(f"All folds results saved to {results_filename}")
