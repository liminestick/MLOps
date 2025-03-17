import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import os

# Создание директории для данных
os.makedirs("data", exist_ok=True)

# Загрузка датасета CIFAR-10
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet18 требует вход 224x224
    transforms.ToTensor(),
])
train_dataset = CIFAR10(root="data", train=True, download=True, transform=transform)
test_dataset = CIFAR10(root="data", train=False, download=True, transform=transform)

print("Датасет CIFAR-10 успешно загружен.")