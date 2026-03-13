import torch
import torch.nn as nn

# Modelo dummy (solo para pruebas)
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(224*224*3, 1)  # transforma imagen a un solo valor
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # aplanar la imagen
        x = torch.sigmoid(self.fc(x))
        return x

# Crear y guardar el modelo
model = DummyModel()
torch.save(model, "model/detector_model.pt")
print("Modelo dummy guardado en model/detector_model.pt")