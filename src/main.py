import os
import time
from models.train_model import trainModel
from src.utils import plotGraficos, saveCsv, saveTxt, manipularCsv, test

# docker build -t tf-gpu-custom .
# docker run --gpus all -it -v "C:\Users\gmara\Documents\Sourcetree\PhobiaScan:/tf/PhobiaScan" -w /tf/PhobiaScan tf-gpu-custom python -m src.main

# docker run --gpus all -it -v "C:\Users\gmara\.nv:/root/.nv" -v "C:\Users\gmara\Documents\Sourcetree\PhobiaScan:/tf/PhobiaScan" -w /tf/PhobiaScan tf-gpu-custom python -m src.fer2013

l2_regularization = [0.0001]
dropout = [0.3,0.4,0.6]
min_lrs = [1e-6]
factors = [0.1,0.2,0.3]
for l2 in l2_regularization:
    for taxaDropout in dropout:
        for factor in factors:
            for min_lr in min_lrs:
                inicio = time.perf_counter()
                hist, model,validation_generator,train_generator,name,patience,batch_size,patienceReduce = trainModel(l2, taxaDropout, factor, min_lr)
                fim = time.perf_counter()

                dir = 'models/tests/' + name
                os.makedirs(dir,exist_ok=True)

                tamanho = 0

                while os.path.exists(f"{dir}/{name}_{tamanho}"):
                    tamanho += 1

                newDir = f'{dir}/{name}_{tamanho}'
                os.makedirs(newDir,exist_ok=True)

                last_val_loss = hist.history['val_loss'][-1]
                last_epoch = hist.history['val_loss'].index(last_val_loss)
                last_acc = hist.history['accuracy'][last_epoch]
                last_loss = hist.history['loss'][last_epoch]
                last_val_acc = hist.history['val_accuracy'][last_epoch]

                final_model_filename_last = f"{newDir}/Last_{last_epoch+1}_loss_{last_val_loss:.4f}_acc_{last_val_acc:.2f}.keras"
                model.save(final_model_filename_last)

                plotGraficos(final_model_filename_last, validation_generator, train_generator, last_val_acc, 'Last', newDir, hist)

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

# test('models/tests/AffectnetGray/AffectnetGray_12/Best_20_loss_1.1719_acc_0.58.keras')

manipularCsv()