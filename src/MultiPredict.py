#LIBRARIES
import torch
import random
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import transforms

#SCRIPTS
from Model import DEVICE,getModel

testTransform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

testDatas = ImageFolder(
  root="Dataset\\test",
  target_transform=None,
  transform=testTransform
)

newModel = getModel(numClasses=2, device=DEVICE)
checkpoint = torch.load("myCheckpoint.pth", map_location=DEVICE)
newModel.load_state_dict(checkpoint["model"])

def makePredictions(model:torch.nn.Module,
                    data:list,
                    device:torch.device=DEVICE):

  predProbs = []
  model.to(device)
  model.eval()

  with torch.inference_mode():
    for sample in data:
      sample = torch.unsqueeze(sample,dim=0).to(device)

      predLogits = model(sample)

      predProb = torch.softmax(predLogits.squeeze(), dim=0)

      predProbs.append(predProb.cpu())

  return torch.stack(predProbs)

random.seed(12)
testSamples = []
testLabels = []

for sample, label in random.sample(list(testDatas), k=9):
  testSamples.append(sample)
  testLabels.append(label)


prediction = makePredictions(model=newModel,
                             data=testSamples,
                             device=DEVICE)

predictionClasses = prediction.argmax(dim=1)

# PLOT PREDICTIONS WITH COLORS
plt.figure(figsize=(9,9))
nrows = 3
ncols = 3

for i, sample in enumerate(testSamples):
    plt.subplot(nrows, ncols, i+1)

    # Görüntüyü göster
    plt.imshow(sample.permute(1,2,0), cmap="gray")
    plt.axis('off')  # Eksenleri gizle

    # Tahmin ve gerçek sınıf isimleri
    predLabel = testDatas.classes[predictionClasses[i]]  # modelin tahmini
    trueLabel = testDatas.classes[testLabels[i]]         # gerçek sınıf

    # Renk belirle
    color = "green" if predLabel == trueLabel else "red"

    # Başlık ekle
    plt.title(f"Prediction: {predLabel}\nTrue: {trueLabel}", color=color, fontsize=10)

plt.tight_layout()
plt.show()