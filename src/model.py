from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import AutoImageProcessor, ResNetForImageClassification
from transformers import AutoImageProcessor, AutoModel
from transformers import AutoProcessor, CLIPVisionModel
from torch.utils.data import DataLoader
import torch
from torch import nn
from datasets import Dataset
from tqdm import tqdm
import os

def get_model(model_name, device="cpu"):
    if model_name == "dino":
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
    elif model_name == "vit":
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)
    elif model_name == "resnet":
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50").to(device)
    elif model_name == "clip":
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")        
        model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    else:
        raise ValueError(f"Model name: {model_name} is not implemented.")
    return processor, model


def add_embeddings(data, model_name, batch_size=100, num_proc=4, environment="supervised", data_dir="./data"):
    data_emb_dir = os.path.join(data_dir, model_name)
    subfolders = [f.name for f in os.scandir(data_dir) if f.is_dir()]
    if (model_name in subfolders):
        environments = [f.name for f in os.scandir(data_emb_dir) if f.is_dir()]
        if (environment in environments):
            print(f"Embedding {model_name} already extracted.")
            return data
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    processor, model = get_model(model_name, device)
    model.eval().requires_grad_(False)
    # data = data.map(lambda x: {"emb1": encoder(x["image"], model, processor, device)}, batch_size=batch_size, batched=True, num_proc=num_proc)
    
    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        num_workers=num_proc,
        pin_memory=True,
        shuffle=False,
    )
    embeddings = []
    for batch in tqdm(dataloader):
        embedding = encoder(batch["image"], model, processor, device)
        embeddings.append(embedding)
    embeddings = torch.cat(embeddings, 0)
    embeddings = Dataset.from_dict({model_name: embeddings.tolist()})
    embeddings.set_format(type="torch", columns=[model_name])
    if not os.path.exists(data_emb_dir):
            os.makedirs(data_emb_dir)
    data_emb_env_dir = os.path.join(data_emb_dir, environment)
    embeddings.save_to_disk(data_emb_env_dir)
        
    return data


def encoder(x, model, processor, device):
    inputs = processor(images=x, return_tensors="pt").to(device)
    outputs = model(**inputs, output_hidden_states=True)
    model_name_full = model.config._name_or_path
    if ("vit" in model_name_full) or ("dino" in model_name_full):
        emb = outputs.hidden_states[-1][:, 0]
    elif ("resnet" in model_name_full):
        emb = outputs.hidden_states[-1].mean(dim=[2,3])
    else:
        raise ValueError(f"Unkown model class: {model_name_full}")
    return emb.to("cpu")

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.output_size = output_size
        self.model = nn.Sequential(
                            nn.Linear(input_size, hidden_size),
                            nn.ReLU(),
                            nn.Linear(hidden_size, output_size),
                            nn.Sigmoid()
                        )
    def forward(self, X):
        return self.model(X) # [0.8, 0.4]
    def probs(self, X):
        return self.model(X) # [0.8, 0.4]
    def pred(self, X):
        return self.model(X).round() # [1, 0]
