import os
import time
from models.train_model import trainModel
from models.hyperparamentos import hyperparametro
from src.utils import plotGraficos, saveCsv, saveTxt

l2_regularization = [0.0001, 0.001, 0.005, 0.01, 0.05]
dropout = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
# batch_size = [16, 32, 64]
for l2 in l2_regularization:
#     for taxaDropout in dropout:
#         for batch in batch_size:
# l2 = 0.002789250672351807
# taxaDropout = 0.3
# dropout = [0.2, 0.7, 0.8]
# factors = [0.1, 0.2, 0.3, 0.4, 0.5]
# min_lrs = [1e-5, 1e-6, 1e-7]
# l2 = 0.001
    # taxaDropout = 0.2
    factor = 0.2
    min_lr = 1e-6
    for taxaDropout in dropout:
#     for factor in factors:
#         for min_lr in min_lrs:
        inicio = time.perf_counter()
        hist, model,validation_generator,train_generator,name,patience,patienceReduce,batch_size = trainModel(l2, taxaDropout, factor, min_lr)
        fim = time.perf_counter()

        dir = 'models/tests/' + name

        tamanho = 0

        while os.path.exists(f"{dir}/{name}_{tamanho}"):
            tamanho += 1

        newDir = f'{dir}/{name}_{tamanho}'
        os.makedirs(newDir,exist_ok=True)

        ## Carregando a última acurácia do último modelo
        last_val_loss = hist.history['val_loss'][-1]
        last_epoch = hist.history['val_loss'].index(last_val_loss)
        last_acc = hist.history['accuracy'][last_epoch]
        last_loss = hist.history['loss'][last_epoch]
        last_val_acc = hist.history['val_accuracy'][last_epoch]

        final_model_filename_last = f"{newDir}/Last_{last_epoch+1}_loss_{last_val_loss:.4f}_acc_{last_val_acc:.2f}.keras"
        model.save(final_model_filename_last)

        plotGraficos(final_model_filename_last, validation_generator, train_generator, last_val_acc, 'Last', newDir, hist)

        ## Carregando a melhor acurácia do modelo
        best_val_loss = min(hist.history['val_loss'])
        best_epoch = hist.history['val_loss'].index(best_val_loss)
        best_acc = hist.history['accuracy'][best_epoch]
        best_loss = hist.history['loss'][best_epoch]
        best_val_acc = hist.history['val_accuracy'][best_epoch]

        final_model_filename_best = f"{newDir}/Best_{best_epoch+1}_loss_{best_val_loss:.4f}_acc_{best_val_acc:.2f}.keras"
        model.save(final_model_filename_best)


        plotGraficos(final_model_filename_best, validation_generator, train_generator, best_val_acc, 'Best', newDir, hist)

        best = {
            'acc': best_acc,
            'loss': best_loss,
            'val_acc': best_val_acc,
            'val_loss': best_val_loss,
            'epoch': best_epoch+1
        }
        last = {
            'acc': last_acc,
            'loss': last_loss,
            'val_acc': last_val_acc,
            'val_loss': last_val_loss,
            'epoch': last_epoch+1
        }

        saveTxt(newDir, best, last, l2, taxaDropout, fim-inicio)
        saveCsv(name,best,last,newDir,l2,taxaDropout,min_lr,factor,patience,patienceReduce,batch_size,fim-inicio)

# hyperparametro()