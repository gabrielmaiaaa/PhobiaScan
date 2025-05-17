import os
from models.train_model import trainModel
from src.utils import plotGraficos

hist, model, validation_generator, train_generator, name = trainModel()

dir = 'models/tests'

tamanho = 0

while os.path.exists(f"{dir}/{tamanho}_{name}"):
    tamanho += 1

newDir = f'{dir}/{tamanho}_{name}'
os.makedirs(newDir,exist_ok=True)

## Carregando a última acurácia do último modelo
last_val_loss = hist.history['val_loss'][-1]
last_epoch = hist.history['val_loss'].index(last_val_loss)
last_val_acc = hist.history['val_accuracy'][last_epoch]

final_model_filename_last = f"{newDir}/Last_{last_epoch}_loss_{last_val_loss:.4f}_acc_{last_val_acc:.2f}.keras"
model.save(final_model_filename_last)

plotGraficos(final_model_filename_last, validation_generator, train_generator, last_val_acc, 'Last', newDir)

## Carregando a melhor acurácia do modelo
best_val_loss = min(hist.history['val_loss'])
best_epoch = hist.history['val_loss'].index(best_val_loss)
best_val_acc = hist.history['val_accuracy'][best_epoch]

final_model_filename_best = f"{newDir}/Best_{best_epoch}_loss_{best_val_loss:.4f}_acc_{best_val_acc:.2f}.keras"
model.save(final_model_filename_best)


plotGraficos(final_model_filename_best, validation_generator, train_generator, best_val_acc, 'Best', newDir)