#LIBRARIES
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

#SCRIPTS
from Model import DEVICE, getModel

classes = ["Normal", "PNEUMONIA"]  
print("Sınıf isimleri:", classes)

image_path = ""  

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(DEVICE)  

model = getModel(numClasses=2, device=DEVICE)
checkpoint = torch.load("myCheckpoint.pth", map_location=DEVICE)
model.load_state_dict(checkpoint["model"])
model.eval()

with torch.inference_mode():
    logits = model(image_tensor)
    probs = torch.softmax(logits, dim=1)
    predicted_class = probs.argmax(dim=1).item()

print(f"Tahmin sınıf index: {predicted_class}")
print(f"Tahmin sınıf adı: {classes[predicted_class]}")
print(f"Sınıf olasılıkları: {probs.cpu().numpy()}")

plt.figure(figsize=(5,5))
plt.imshow(image, cmap="gray")
plt.axis("off")
plt.title(f"Prediction: {classes[predicted_class]}\n"
          f"Probs: {probs.squeeze().cpu().numpy()}", fontsize=10)
plt.show()
