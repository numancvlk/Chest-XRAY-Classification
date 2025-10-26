#LIBRARIES
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
from torchvision import transforms

from tqdm.auto import tqdm
from timeit import default_timer

#SCRIPTS
from Utils import loadCheckpoint, saveCheckpoint, getLoaders,printTrainTime,trainStep,testStep,validationStep,modelSummary, accuracy
from Model import DEVICE, getModel


#HYPERPARAMETERS
EPOCHS = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 16
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = True


#PATHS
TRAIN_IMAGES_DIR = "Dataset\\train"
TEST_IMAGES_DIR = "Dataset\\test"
VALIDATION_IMAGES_DIR = "Dataset\\val"



if __name__ == "__main__":
    torch.manual_seed(42)
    patience = 15
    bestLoss = float("inf")
    patienceCounter = 0

    trainTransform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),      # ±10 derece rotasyon
        transforms.ColorJitter(             # parlaklık, kontrast, doygunluk
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    testTransform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    validationTransform =  transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    #DATALOADERS
    trainDataLoader, testDataLoader, validationDataLoader = getLoaders(trainImageDir=TRAIN_IMAGES_DIR,
                                                                        valImageDir=VALIDATION_IMAGES_DIR,
                                                                        testImageDir=TEST_IMAGES_DIR,
                                                                        trainTransform=trainTransform,
                                                                        testTransform=testTransform,
                                                                        valTransform=validationTransform,
                                                                        batchSize=BATCH_SIZE,
                                                                        numWorkers=NUM_WORKERS,
                                                                        pinMemory=PIN_MEMORY)
    
    # MODEL - LOSSFN - SCHEDULER - OPTIMIZER
    myModel = getModel(numClasses=2,
                       device=DEVICE)
    
    lossFn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(params=myModel.parameters(),
                                  lr=LEARNING_RATE)
    
    scheduler = ReduceLROnPlateau(optimizer, 
                                mode='min', 
                                factor=0.1, 
                                patience=7)
    
    if LOAD_MODEL == True:
        loadCheckpoint(checkpointFile="myCheckpoint.pth",
                       model=myModel,
                       optimizer=optimizer)
    
    startTrainTimer = default_timer()

    for epoch in tqdm(range(EPOCHS)):

        trainStep(model=myModel,
                  dataLoader=trainDataLoader,
                  optimizer=optimizer,
                  lossFn=lossFn,
                  accFn=accuracy,
                  device=DEVICE)
        
        validationLoss = validationStep(model=myModel,
                       dataLoader=validationDataLoader,
                       lossFn=lossFn,
                       accFn=accuracy,
                       device=DEVICE)
        

        
        for paramsGroup in optimizer.param_groups:
            currentLR = paramsGroup["lr"]
            print(f"CURRENT LR = {currentLR}")
        
        scheduler.step(validationLoss)

        if validationLoss < bestLoss:
            bestLoss = validationLoss
            patienceCounter = 0
            saveCheckpoint(model=myModel,
                           optimizer=optimizer,
                           epoch=epoch)
        else:
            patienceCounter += 1
            print(f"{patienceCounter} epoch'tur gelişme yok.")

            if patienceCounter == patience:
                print("EARLY STOPPING TRIGGERED")
                break

    modelSum = modelSummary(model=myModel,
                 dataLoader=testDataLoader,
                 lossFn=lossFn,
                 accFn=accuracy,
                 device=DEVICE)
    
    print(modelSum)
    
    endTrainTimer = default_timer()

    printTrainTime(start=startTrainTimer,
                   end=endTrainTimer,
                   device=DEVICE)
    