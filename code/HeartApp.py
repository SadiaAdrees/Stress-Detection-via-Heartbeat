import numpy as np
import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Disable the PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the data
dfs = [pd.read_csv('F:/7th Semester/Distributed Computing/project/ptbdb_' + x + '.csv') for x in ['normal', 'abnormal']]
for df in dfs:
    df.columns = list(range(len(df.columns)))
data = pd.concat(dfs, axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True)
data = data.rename({187: 'Label'}, axis=1)
y = data['Label'].copy()
X = data.drop('Label', axis=1).copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=1)

# Display bar chart of heartbeats distribution before training
fig_before_training, ax_before_training = plt.subplots(figsize=(10, 6))
sns.countplot(x='Label', data=data, hue='Label', palette={0: 'blue', 1: 'orange'})
plt.title('Heartbeats Distribution Before Training')
plt.xlabel('Label')
plt.ylabel('Count')
plt.legend(title='Label', labels=['Normal', 'Abnormal'])
st.pyplot(fig_before_training)

# Build and train the model
inputs = tf.keras.Input(shape=(X_train.shape[1],))
expand = tf.expand_dims(inputs, axis=2)
gru = tf.keras.layers.GRU(256, return_sequences=True)(expand)
flatten = tf.keras.layers.Flatten()(gru)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(flatten)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(name='auc')
    ]
)

# history = model.fit(
#     X_train,
#     y_train,
#     validation_split=0.2,
#     batch_size=32,
#     epochs=100,
#     callbacks=[
#         tf.keras.callbacks.EarlyStopping(
#             monitor='val_loss',
#             patience=5,
#             restore_best_weights=True
#         )
#     ]
# )


# Build and train the model (if not already trained)
if not os.path.exists('my_model.h5'):
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        batch_size=32,
        epochs=100,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
    )
    # Save the trained model
    model.save('my_model.h5')
else:
    # Load the pre-trained model
    model = tf.keras.models.load_model('my_model.h5')

# Streamlit app
st.title('Heartbeat Classification Demo')

# Sidebar for model information
st.sidebar.subheader('Model Information:')
st.sidebar.text(model.summary())
results = model.evaluate(X_test, y_test, verbose=0)
st.sidebar.text("Test Accuracy: {:.2f}%".format(results[1] * 100))
st.sidebar.text("     Test AUC: {:.4f}".format(results[2]))

# Display classification report
st.subheader('Classification Report:')
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)
st.text(classification_report(y_test, y_pred_binary))

# Display confusion matrix heatmap
st.subheader('Confusion Matrix:')
class_names = ['Normal', 'Abnormal']
conf_matrix = confusion_matrix(y_test, y_pred_binary)
# Create a heatmap using seaborn
fig_conf_matrix, ax_conf_matrix = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names, ax=ax_conf_matrix)

# Add labels and title
ax_conf_matrix.set_xlabel("Predicted Labels")
ax_conf_matrix.set_ylabel("True Labels")
ax_conf_matrix.set_title("Confusion Matrix")

# Show the plot using st.pyplot() by passing the figure explicitly
st.pyplot(fig_conf_matrix)

# Display other relevant information or visualizations

# Bar chart showing the distribution of predicted probabilities
st.subheader('Distribution of Predicted Probabilities:')
fig_proba, ax_proba = plt.subplots()
ax_proba.hist(y_pred, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
ax_proba.set_xlabel('Predicted Probability')
ax_proba.set_ylabel('Frequency')
ax_proba.set_title('Distribution of Predicted Probabilities')
st.pyplot(fig_proba)

# Input for predicting a single heartbeat sequence
st.subheader('Predict Single Heartbeat:')
heartbeat_input = st.text_input('Enter a heartbeat sequence (comma-separated values):', '0.5,0.6,0.7,0.8')
heartbeat_values = [float(value) for value in heartbeat_input.split(',')]

if st.button('Predict'):
    # Ensure the input sequence has the correct length (187 in this case)
    if len(heartbeat_values) != 187:
        st.warning('Please enter a sequence of 187 values.')
    else:
        # Prepare input for the model
        heartbeat_array = np.array(heartbeat_values).reshape(1, -1, 1)

        # Make prediction
        prediction = model.predict(heartbeat_array)

        # Display prediction result
        if prediction[0, 0] > 0.5:
            st.success('Prediction: Abnormal Heartbeat')
        else:
            st.success('Prediction: Normal Heartbeat')

