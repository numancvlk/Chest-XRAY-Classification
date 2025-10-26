#LIBRARIES
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

#SCRIPTS
from Model import DEVICE

def printTrainTime(start,end,device):
    totalTime = end - start
    print(f"Total train time is {totalTime} on the {device}")


def saveCheckpoint(model,optimizer,epoch, checkpointFile = "myCheckpoint.pth"):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }

    torch.save(checkpoint, checkpointFile)
    print("MODEL CHECKPOINT ALINDI")

def loadCheckpoint(checkpointFile,model,optimizer):
    checkpoint = torch.load(checkpointFile, map_location=DEVICE)
        
    model.load_state_dict(checkpoint["model"])
        
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        
        epoch = checkpoint["epoch"]
        print(f"CHECKPOINT YUKLENDİ: Epoch {epoch} model ağırlıkları ve/veya optimizer durumu.")
        
        return epoch + 1
    
def accuracy(yTrue,yPred):
    correct = torch.eq(yTrue,yPred).sum().item()
    acc = (correct / len(yTrue)) * 100
    return acc

def getLoaders(trainImageDir,
               valImageDir,
               testImageDir,
               trainTransform,
               valTransform,
               testTransform,
               batchSize,
               numWorkers,
               pinMemory):
    
    trainDatas = ImageFolder(root=trainImageDir,
                             target_transform= None,
                             transform= trainTransform)
    
    testDatas = ImageFolder(root=testImageDir,
                            target_transform=None,
                            transform=testTransform)
    
    validationDatas = ImageFolder(root=valImageDir,
                                  target_transform=None,
                                  transform=valTransform)
    
    trainDataLoader = DataLoader(dataset=trainDatas,
                                 shuffle=True,
                                 batch_size=batchSize,
                                 num_workers=numWorkers,
                                 pin_memory=pinMemory)
    
    testDataLoader = DataLoader(dataset=testDatas,
                                shuffle=False,
                                batch_size=batchSize,
                                num_workers=numWorkers,
                                pin_memory=pinMemory)
    
    validationDataLoader = DataLoader(dataset=validationDatas,
                                      shuffle=False,
                                      batch_size=batchSize,
                                      num_workers=numWorkers,
                                      pin_memory=pinMemory)
    
    return trainDataLoader, testDataLoader, validationDataLoader


def trainStep(model: torch.nn.Module,
              dataLoader: torch.utils.data.DataLoader,
              optimizer: torch.optim.Optimizer,
              lossFn: torch.nn.Module,
              accFn,
              device: torch.device = DEVICE):
    
    model.train()
    trainLoss, trainAcc = 0,0

    for batch, (xTrain,yTrain) in enumerate(dataLoader):
        xTrain, yTrain = xTrain.to(device), yTrain.to(device)

        with torch.autocast(device_type="cuda"):
            trainPred = model(xTrain)
            loss = lossFn(trainPred,yTrain)
            trainLoss += loss.item()
            trainAcc += accFn(yTrue = yTrain, yPred = trainPred.argmax(dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    trainLoss /= len(dataLoader)
    trainAcc /= len(dataLoader)
    print(f"TRAIN LOSS = {trainLoss:.4f} | TRAIN ACCURACY = {trainAcc}%")

def testStep(model: torch.nn.Module,
              dataLoader: torch.utils.data.DataLoader,
              lossFn: torch.nn.Module,
              accFn,
              device: torch.device = DEVICE,
              returnLoss = False):
    
    testLoss, testAccuracy = 0,0
    
    model.eval()
    with torch.inference_mode():
        for xTest,yTest in dataLoader:
            xTest,yTest = xTest.to(device), yTest.to(device)

        
            testPred = model(xTest)

            loss = lossFn(testPred, yTest)
            testLoss += loss.item()

            acc = accFn(yTrue = yTest, yPred = testPred.argmax(dim=1))
            testAccuracy += acc
    
    testLoss /= len(dataLoader)
    testAccuracy /= len(dataLoader)
    print(f"TEST LOSS = {testLoss:.5f} | TEST ACCURACY = {testAccuracy}%")

    if returnLoss: #EARLY STOPPING İÇİN EKLEDİĞİMİZ BİR ŞEY
        return testLoss
    
def validationStep(model:torch.nn.Module,
                   dataLoader: torch.utils.data.DataLoader,
                   lossFn: torch.nn.Module,
                   accFn,
                   device: torch.device = DEVICE):
    model.eval()
    validationLoss, validationAcc = 0,0
    with torch.inference_mode(): 
        for xVal, yVal in dataLoader:
            xVal, yVal = xVal.to(device), yVal.to(device)

            yPred = model(xVal)
            loss = lossFn(yPred, yVal)
            validationLoss += loss.item()
            validationAcc += accFn(yTrue=yVal, yPred=yPred.argmax(dim=1))

    validationLoss /= len(dataLoader)
    validationAcc /= len(dataLoader)

    print(f"VALIDATION LOSS = {validationLoss:.4f} | VALIDATION ACC = {validationAcc}%")
    return validationLoss


def modelSummary(model: torch.nn.Module,
              dataLoader: torch.utils.data.DataLoader,
              lossFn: torch.nn.Module,
              accFn,
              device: torch.device = DEVICE):
    
    summaryLoss, summaryAccuracy = 0,0
    
    model.eval()
    with torch.inference_mode():
        for xTest,yTest in dataLoader:
            xTest,yTest = xTest.to(device), yTest.to(device)

        
            testPred = model(xTest)

            loss = lossFn(testPred, yTest)
            summaryLoss += loss.item()

            acc = accFn(yTrue = yTest, yPred = testPred.argmax(dim=1))
            summaryAccuracy += acc

    summaryLoss /= len(dataLoader)
    summaryAccuracy /= len(dataLoader)

    return {"MODEL NAME": model.__class__.__name__,
            "MODEL LOSS": summaryLoss,
            "MODEL ACCURACY": summaryAccuracy}