# License: BSD
# Author: Sasank Chilamkurthy


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
path = 'E:\datasets\hymenoptera_data'
# Data augmentation and normalization for training
# Just normalization for validation


#контроль точки входа в программу
if __name__ == "__main__":
    
    #преобразования изображений
    data_transforms = {
        
        'train': transforms.Compose([ #конвейер преобразований (различные преобразования, 
                                      #    которые необходимо выполнить в вашем наборе изображений)
                                      
            transforms.RandomResizedCrop(224), #Обрезать произвольную часть изображения
                                               #    и изменить ее размер до 224x224 пикселей
                                               
            transforms.RandomHorizontalFlip(), #горизонтально перевернуть данное изображение 
                                               #    случайным образом с заданной вероятностью (дефолт 0.5)
            
            transforms.ToTensor(), #преобразовать все в тензор изображения
            
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #нормализация изображений по трем каналам с 
            # мат. ожиданиями [0.485, 0.456, 0.406] и СКО [0.229, 0.224, 0.225]
             
        ]),
        'val': transforms.Compose([ #конвейер преобразований (различные преобразования, 
                                    #    которые необходимо выполнить в вашем наборе изображений)
                                    
            transforms.Resize(256), #изменение изображения до размера 256x256 пикселей
            transforms.CenterCrop(224), #обрезать изображение от центра до изображения 224x224 пикселей
            
            transforms.ToTensor(), #преобразовать все в тензор изображения
            
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #нормализация изображений по трем каналам с 
            # мат. ожиданиями [0.485, 0.456, 0.406] и СКО [0.229, 0.224, 0.225] 
            
        ]),
    }

    
    #загрузка изображений
    image_datasets = {x: datasets.ImageFolder(os.path.join(path, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    
    #утилита загрузки изображений
    #image_datasets[x] - класс изображений, тут их 2 
    #batch_size - количество выборок, используемых в обучении за одну итерацию
    #   batching the data - дословно дозирование информации, пакетная обработка данных
    #shuffle - перемешивание изображений
    #   если true - то все изображения класса перемешиваются и загружаются в пакеты
    #   если false - то изображения в пакеты попадают по порядку один за другим 
    #num_workers -  вроде как распараллеливание
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    
    #словарь, элементами которого являются количетсва изображений каждого класса
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    #названия классов 
    class_names = image_datasets['train'].classes

    #устройство, на котором будут выполняться вычисления
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    #функция вывода некоторых изображений из датасета, на которых будет тренироваться модель
    def imshow(inp, title=None):
        """Display image for Tensor."""
        
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
    imshow(out, title=[class_names[x] for x in classes])
    
    #настройка обучения модели
    def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
        
        #время начала работы 
        since = time.time()

        
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

                    # deep copy the model
                    #если режим оценки и точность данной эпохи лучше чем сохранная точность
                    #   модель и точность сохраняются
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(model.state_dict(), best_model_params_path)
                
                #отступ
                print()
            
            #время выполнения цикла
            time_elapsed = time.time() - since
            
            #вывод времени выполнения цикла в минутах и секундах и вывод наибольшей достигнутой точности 
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'Best val Acc: {best_acc:4f}')

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
    model_ft = models.resnet18(weights='IMAGENET1K_V1')
    
    #улучшение предварительно обученной модели путем небольших корректировок
    
    #количество выходов в последнем полностью подключенном слое
    num_ftrs = model_ft.fc.in_features
    
    
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
    
    #повторная инициализация линейного слоя с num_ftrs входами и выходами,
    #   количество которых равно количество классов в дата сете
    model_ft.fc = nn.Linear(num_ftrs, 2)

    #загрузка модели на GPU
    model_ft = model_ft.to(device)

    #функция потерь (перекрестная потеря энтропии между логитами и целевыми значениями)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    #опитимизатор
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    #планировщик
    #регулирует скорость обучения
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    #обучение модели
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)
    
    #прогнозирование
    visualize_model(model_ft)
    
    #вывод на экран 
    plt.ioff()
    plt.show()
    
    
    
    #сверточная сеть fixed feature extraction
    #используется resnet18 как основа
    model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    
    #замораживание слоев модели сети
    for param in model_conv.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    
    #повторная инициализация линейного слоя с num_ftrs входами и выходами,
    #   количество которых равно количество классов в дата сет
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    #загрузка модели на GPU
    model_conv = model_conv.to(device)

    #функция потерь (перекрестная потеря энтропии между логитами и целевыми значениями)
    criterion = nn.CrossEntropyLoss()

    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    #опитимизатор
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    #планировщик
    #регулирует скорость обучения
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    
    #обучение модели
    model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)
    
    #прогнозирование
    visualize_model(model_conv)

    #вывод на экран
    plt.ioff()
    plt.show()
    
    
    #графическое представление для какого-то своего отдельного изображения
    def visualize_model_predictions(model,img_path):
        
        #судя по всему возвращает
        #   true - если обучалась
        #   fasle - если не обучалась
        was_training = model.training
        
        #переключение модели в режим оценки (изменение поведения слоев)
        model.eval()

        #загрузка и преобразования изображения с помощью PILа
        img = Image.open(img_path)
        img = data_transforms['val'](img)
        img = img.unsqueeze(0)
        img = img.to(device)

        #выключение вычисления градиентов
        with torch.no_grad():
            
            #подача входных данных
            outputs = model(img)
            
           #определение класса с помощью модели (resnet18) 
            _, preds = torch.max(outputs, 1)

            ax = plt.subplot(2,2,1)
            ax.axis('off')
            ax.set_title(f'Predicted: {class_names[preds[0]]}')
            
            #поскольку imshow нельзя применить к данным графического процессора, его нужно сначала перенести в процессор
            imshow(img.cpu().data[0])

            #установка режима работы модели в исходное состояние
            model.train(mode=was_training)
    
    #вызов функции
    visualize_model_predictions(
        model_conv,
        
        #путь к изображению
        img_path=r'E:\datasets\hymenoptera_data\val\bees\72100438_73de9f17af.jpg'
    )

    #вывод на экран
    plt.ioff()
    plt.show()