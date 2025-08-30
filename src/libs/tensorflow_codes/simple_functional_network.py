from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from sklearn.decomposition import PCA
import tensorflow as tf
import pandas as pd

K.clear_session()

input1 = Input(shape=(4,))
input2 = Input(shape=(2,))
out1 = Dense(16, activation='relu')(input1)
out2 = Dense(16, activation='relu')(input2)
out = Concatenate(axis=1)([out1, out2])
out = Dense(16, activation='relu')(out)
output = Dense(3, activation='softmax')(out)
model = Model(inputs=[input1, input2], outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()


from sklearn.decomposition import PCA

iris_df = pd.read_csv('iris.data', header=None)
iris_df.columns = ['sepal_length', 'sepal_width', 'petal_width',
'petal_length', 'label']
iris_df["label"] = iris_df["label"].map({'Iris-setosa':0, 'Iris-versicolor':1,
'Iris-virginica':2})

iris_df = iris_df.sample(frac=1.0, random_state=4321)
x = iris_df[["sepal_length", "sepal_width", "petal_width", "petal_length"]]
x = x - x.mean(axis=0)
y = tf.one_hot(iris_df["label"], depth=3)
pca_model = PCA(n_components=2, random_state=4321)
x_pca = pca_model.fit_transform(x)
model.fit([x, x_pca], y, batch_size=64, epochs=25)
