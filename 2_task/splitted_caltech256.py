!pip install pandas
!pip install torchsampler
#скачивание архива

#!rm -rf /content/drive/MyDrive/archive
!wget -P /content/drive/MyDrive https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar
import os
import tarfile
os.mkdir("/content/drive/MyDrive/archive")

#распаковка

file = tarfile.open("/content/drive/MyDrive/256_ObjectCategories.tar")
file.extractall(r"/content/drive/MyDrive/archive")
os.rename(r"/content/drive/MyDrive/archive/256_ObjectCategories",r"/content/drive/MyDrive/archive/train")
file.extractall(r"/content/drive/MyDrive/archive")
os.rename(r"/content/drive/MyDrive/archive/256_ObjectCategories",r"/content/drive/MyDrive/archive/val")
file.close()

#!pip install optuna
#директории для разбитого архива
import os
import shutil
#!rm -rf /content/drive/MyDrive/splitted_archive
os.mkdir("/content/drive/MyDrive/splitted_archive")
os.mkdir("/content/drive/MyDrive/splitted_archive/train")
os.mkdir("/content/drive/MyDrive/splitted_archive/val")
file_train = open("/content/drive/MyDrive/train_lst.txt")
file_val = open("/content/drive/MyDrive/val_lst.txt")

#перенос файлов

for line in file_train:
    try:
        list_str = line.split()
        list_str.pop()
        list_str.pop()
        path_file = "/content/drive/MyDrive/archive/train" + list_str[0][20:]
        folder = list_str[0].split('/')[1]
        try:
            os.mkdir("/content/drive/MyDrive/splitted_archive/train/" + folder)
            shutil.copy(path_file,"/content/drive/MyDrive/splitted_archive/train" + list_str[0][20:])
        except:
            shutil.copy(path_file,"/content/drive/MyDrive/splitted_archive/train" + list_str[0][20:])
    except EOFError as e:
        break

for line in file_val:
    try:
        list_str = line.split()
        list_str.pop()
        list_str.pop()
        path_file = "/content/drive/MyDrive/archive/val" + list_str[0][20:]
        folder = list_str[0].split('/')[1]
        try:
            os.mkdir("/content/drive/MyDrive/splitted_archive/val/" + folder)
            shutil.copy(path_file,"/content/drive/MyDrive/splitted_archive/val" + list_str[0][20:])
        except:
            shutil.copy(path_file,"/content/drive/MyDrive/splitted_archive/val" + list_str[0][20:])
    except EOFError as e:
        break

#подключение библиотек


#сам модуль
import torch
#пакет для нейронных сетей
import torch.nn as nn
#разные алгоритмы оптимизации
import torch.optim as optim
#методы для настройки скорости обучения в зависимости от количества эпох (настройка изменения градиентных шагов в ходе обучения)
from torch.optim import lr_scheduler
#библиотека GPU примитивов (вроде как ускорение с помощью GPU)
import torch.backends.cudnn as cudnn
#ну это просто numpy
import numpy as np
#бибилиотека используемая, для обработки изображений
import torchvision
#torchvision.dataset для загрузки данных (датасетов - наборов изображений)
#torchvision.models для существующих моделей (например alexnet), нам нужен resnet 18
#torchvision.transforms - простые функции преобразования изображений
from torchvision import datasets, models, transforms
#вывод изобржаний
import matplotlib.pyplot as plt
import time
import os
#модуль для работы с растовой графикой
from PIL import Image
#создание врменного каталога
#после завершения контекста или уничтожения объекта временного каталога,
#   временный каталог и его содержимое удаляются из фаловой системы
from tempfile import TemporaryDirectory
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
from torchsampler import ImbalancedDatasetSampler
from torch.utils.data import WeightedRandomSampler
from collections import Counter
import pandas as pd
#import optuna


#это позволяет использовать функцию cuDNN, которая будет тестировать несколько
#   различных способов вычисления сверток в cuDNN,
#   а затем использовать самый быстрый метод с этого момента

#также написано, что это дает программе возможность тратить немного доп. времени
#   при запуске каждого слоя свертки на поиск по всей сети,
#   что позволяет ускорить нейронную сеть
cudnn.benchmark = True


#включает интерактивный режим (требуется увидеть динамику изменения данных)
plt.ion()  # interactive mode

#путь к папке с набором данных
path = r"/content/drive/MyDrive/splitted_archive"



#контроль точки входа в программу
if __name__ == "__main__":

    #преобразования изображений
    data_transforms = {

        'train': transforms.Compose([ #конвейер преобразований (различные преобразования,
                                      #    которые необходимо выполнить в вашем наборе изображений)
            transforms.RandomResizedCrop(176), #случайная обрезка

            transforms.TrivialAugmentWide(), #изменение с помощью Trvial Augment

            transforms.ToTensor(), #преобразовать все в тензор изображения

            transforms.RandomErasing(p = 0.1), #случаное вырезание куска изображения


        ]),
        'val': transforms.Compose([ #конвейер преобразований (различные преобразования,
                                    #    которые необходимо выполнить в вашем наборе изображений)

            transforms.Resize((232,232)), #изменение изображения до размера 232x232 пикселей
            transforms.CenterCrop(224), #обрезать изображение от центра до изображения 224x224 пикселей

            transforms.ToTensor(), #преобразовать все в тензор изображения

        ]),
    }


    #загрузка изображений
    image_datasets = {x: datasets.ImageFolder(os.path.join(path, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}

    #названия классов
    class_names = image_datasets['train'].classes

    class_weights_train = []
    for root,subdir,files in os.walk(path + r"/train"):
        if len(files)>0:
            class_weights_train.append(len(image_datasets['train'].imgs)/len(files))

    #утилита загрузки изображений
    #image_datasets[x] - класс изображений, тут их 2
    #batch_size - количество выборок, используемых в обучении за одну итерацию
    #   batching the data - дословно дозирование информации, пакетная обработка данных
    #shuffle - перемешивание изображений
    #   если true - то все изображения класса перемешиваются и загружаются в пакеты
    #   если false - то изображения в пакеты попадают по порядку один за другим
    #num_workers -  вроде как распараллеливание
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=128,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    

    #устройство, на котором будут выполняться вычисления
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #функция вывода некоторых изображений из датасета, на которых будет тренироваться модель

    def imshow(inp, title=None):
        '''Display image for Tensor.'''

        #транспонирование данных
        inp = inp.numpy().transpose((1, 2, 0))

        #снова нормализация с мат. ожиданием [0.485, 0.456, 0.406] и СКО [0.229, 0.224, 0.225]
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean

        #обрезка изображения от 0 до 1, чтобы не было больших значений в массивах
        inp = np.clip(inp, 0, 1)

        #функция представления 2D растров, это могут быть картинки, двумерные массивы, матрицы
        plt.imshow(inp)

        #заголовки к картинкам
        if title is not None:
            plt.title(title)

        #пауза, чтобы графики обновились
        plt.pause(0.001)  # pause a bit so that plots are updated


    # Get a batch of training data

    #забираем первый пакет (batch)
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch

    #сетка из изображений
    out = torchvision.utils.make_grid(inputs)

    #вывод изображений
    imshow(out, title=[class_names[x][3:] for x in classes])

    #настройка обучения модели
    def train_model(model, criterion, optimizer, scheduler,mode_for_file,num_epochs=25):

        #время начала работы
        since = time.time()

        if mode_for_file == "finetuning":
          log_epochs = open("/content/drive/MyDrive/splitted_dataset256_finetuning_model_log_epochs.txt","w+")
        else:
          log_epochs = open("/content/drive/MyDrive/splitted_dataset256_fixed_feature_extractor_model_log_epochs.txt","w+")
        # временная директория, куда будет хранится последняя модель
        # Create a temporary directory to save training checkpoints

        with TemporaryDirectory() as tempdir:
            #соединение путя временный директории в файловую систему
            best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

            #сохранение первой модели
            torch.save(model.state_dict(), best_model_params_path)

            #лучшая точность
            best_acc = 0.0

            #эпохи
            for epoch in range(num_epochs):

                #вывод номера эпохи и отступа
                print(f'Epoch {epoch}/{num_epochs - 1}')
                print('-' * 10)

                log_epochs.write(f'Epoch {epoch}/{num_epochs - 1}\n')
                log_epochs.write('-' * 10 + '\n')

                #за каждую эпоху модель проходит два этапа: обучение и оценивается (достигается изменением поведения некоторых слоев)
                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:

                    #если этап обучения, то модель переводится в режим обучения
                    #если оценивания, то в режим оценки
                    if phase == 'train':
                        model.train()  # Set model to training mode
                    else:
                        model.eval()   # Set model to evaluate mode

                    #отслеживание потерь, чтобы потом рассчитать их среднее
                    running_loss = 0.0

                    #отслеживание правильных прогнозов, чтобы потом рассчитать точность модели
                    running_corrects = 0

                    # Iterate over data.
                    #цикл по данным
                    for inputs, labels in dataloaders[phase]:
                        #извлечение изображений
                        inputs = inputs.to(device)

                        #извлечение имен изображений
                        labels = labels.to(device)

                        # zero the parameter gradients
                        #обнуление всех градиентов оптимизатором
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        #torch.set_grad_enabled позволяет сохранять градиенты в зависимости от режима работы программы
                        #   train - будут сохранятся
                        #   validation - не будут
                        with torch.set_grad_enabled(phase == 'train'):

                            #подача входных данных
                            outputs = model(inputs)

                            #определение класса с помощью модели (resnet18)
                            _, preds = torch.max(outputs, 1)

                            #используя выходные данные и имена данных (изображений) вычисляются потери (ошибка определния класса)
                            loss = criterion(outputs, labels)

                            #если модель в режими обучения, вычисляются градиенты и с помощью
                            #   оптимизатора производится обновление параметров модели
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                #lrs.append(optimizer.param_groups[0]["lr"])
                                optimizer.step()

                        #обновляются статистические параметры модели, продолжая отслеживать потери и корректируя прогнозы
                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)

                    #обновляет значения параметров модели в соответствии с стратегией оптимизации
                    if phase == 'train':
                        scheduler.step()

                    #средняя точность и среднее ошибок (потерь) для данной эпохи
                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase]

                    #вывод средней точности и среднего ошибок (потерь)
                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                    log_epochs.write(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}\n')

                    # deep copy the model
                    #если режим оценки и точность данной эпохи лучше чем сохранная точность
                    #   модель и точность сохраняются
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(model.state_dict(), best_model_params_path)

                #отступ
                print()
                log_epochs.write('\n')

            #время выполнения цикла
            time_elapsed = time.time() - since

            #вывод времени выполнения цикла в минутах и секундах и вывод наибольшей достигнутой точности
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'Best val Acc: {best_acc:4f}')

            log_epochs.write(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s\n')
            log_epochs.write(f'Best val Acc: {best_acc:4f}\n')
            log_epochs.close()

            #захват лучшей модели
            model.load_state_dict(torch.load(best_model_params_path))
        return model

    #графическое представление модели
    def visualize_model(model, num_images=6):
        #судя по всему возвращает
        #   true - если обучалась
        #   fasle - если не обучалась
        was_training = model.training

        #переключение модели в режим оценки (изменение поведения слоев)
        model.eval()

        #счетчик изображений
        images_so_far = 0
        fig = plt.figure()

        #выключение вычисления градиентов
        with torch.no_grad():

            #загрузка изображений и их имен
            for i, (inputs, labels) in enumerate(dataloaders['val']):
                inputs = inputs.to(device)
                labels = labels.to(device)

                #подача входных данных
                outputs = model(inputs)

                #определение класса с помощью модели (resnet18)
                _, preds = torch.max(outputs, 1)

                #вывод изображений, тут сеткой, а в collab построчно
                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title(f'predicted: {class_names[preds[j]]}')
                    #поскольку imshow нельзя применить к данным графического процессора, его нужно сначала перенести в процессор
                    imshow(inputs.cpu().data[j])

                    #пакетный вывод изображений, если счетчик равен введенному количеству изобржений
                    #   num_images, происходит выход из цикла
                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return

            #установка режима работы модели в исходное состояние
            model.train(mode=was_training)


    #сверточная сеть finetunning
    #модель, которая обучается - resnet18
    model_ft = models.resnet18(pretrained = False) 
    #улучшение предварительно обученной модели путем небольших корректировок

    #количество выходов в последнем полностью подключенном слое
    num_ftrs = model_ft.fc.in_features


    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.

    #повторная инициализация линейного слоя с num_ftrs входами и выходами,
    #   количество которых равно количество классов в дата сете
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    #загрузка модели на GPU
    model_ft = model_ft.to(device)


    class_weights_train = torch.tensor(class_weights_train,dtype=torch.float).cuda()
    #функция потерь (перекрестная потеря энтропии между логитами и целевыми значениями) с весами классов,
    #   так как классы не сбалансированы
    criterion = nn.CrossEntropyLoss(weight=class_weights_train,label_smoothing=0.1)

    # Observe that all parameters are being optimized
    #опитимизатор
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.005, momentum=0.9, weight_decay=2e-05)


    # Decay LR by a factor of 0.1 every 7 epochs
    #планировщик
    #регулирует скорость обучения
    num_epochs = 100
    lr_warmup_epochs = 5
    lr_warmup_decay = 0.01

    main_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft,T_max = num_epochs-lr_warmup_epochs, eta_min = 0)
    warmup_scheduler = lr_scheduler.LinearLR(optimizer_ft, start_factor=lr_warmup_decay, total_iters= lr_warmup_epochs)
    exp_lr_scheduler = lr_scheduler.SequentialLR(optimizer_ft, schedulers=[warmup_scheduler, main_scheduler], milestones= [lr_warmup_epochs])

    #обучение модели
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,"finetuning",
                      num_epochs=100)

    #прогнозирование
    visualize_model(model_ft)

    #вывод на экран
    plt.ioff()
    plt.show()
