import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, SimpleRNN, BatchNormalization
from keras import backend as K
from tensorflow.keras.optimizers import SGD


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_score(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

veri = pd.read_csv('train.csv')
veri = veri.drop(['id'], axis=1)

label_encoder = LabelEncoder().fit(veri.species)
labels = label_encoder.transform(veri.species)
classes = list(label_encoder.classes_)

X = veri.drop(['species'], axis=1)
y = labels
nb_features = 192
nb_classes = len(classes)

sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(SimpleRNN(512, input_shape=(X_train.shape[1],1)))
model.add(Activation("relu"))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(2048, activation="relu"))
model.add(Dense(1024, activation="relu"))
model.add(Dense(nb_classes, activation="sigmoid"))
model.summary()

opt = SGD(learning_rate=0.01, decay=0.001, momentum=0.9, nesterov=True)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy",f1_score])

score = model.fit(X_train, y_train, epochs=25, validation_data=(X_test,y_test))

print(("Ortalama Eğitim Kaybı: ", np.mean(model.history.history["loss"])))
print(("Ortalama Eğitim Başarımı: ", np.mean(model.history.history["accuracy"])))
print(("Ortalama Doğrulama Kaybı: ", np.mean(model.history.history["val_loss"])))
print(("Ortalama Doğrulama Başarımı: ", np.mean(model.history.history["val_accuracy"])))


plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.title("Model Başarımları")
plt.ylabel("Başarım")
plt.xlabel("Epok Sayısı")
plt.legend(["Eğitim", "Test"], loc="upper left")
plt.show()

plt.plot(model.history.history['loss'], color="g")
plt.plot(model.history.history['val_loss'], color="r")
plt.title("Model Kayıpları")
plt.ylabel("Kayıp")
plt.xlabel("Epok Sayısı")
plt.legend(["Eğitim", "Test"], loc="upper left")
plt.show()

loss, accuracy, f1_score = model.evaluate(X_test, y_test)
print("F1 Skor: ",f1_score)


