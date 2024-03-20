from transformers import ViTImageProcessor, AutoImageProcessor, AutoProcessor, SiglipImageProcessor
from transformers import ViTForImageClassification, ResNetForImageClassification, AutoModel, CLIPVisionModel, ViTMAEModel, SiglipVisionModel
from datasets import Dataset

import torch
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm
import os

def get_model(encoder_name, device="cpu"):
    if encoder_name == "dino":
        processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        model = AutoModel.from_pretrained('facebook/dinov2-base').to(device)
    elif encoder_name == "vit":
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)
    elif encoder_name == "vit_large":
        processor = SiglipImageProcessor.from_pretrained('google/siglip-base-patch16-512')
        model = SiglipVisionModel.from_pretrained('google/siglip-base-patch16-512').to(device)
    elif encoder_name == "resnet":
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50").to(device)
    elif encoder_name == "clip":
        processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")        
        model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    elif encoder_name == "clip_large":
        processor = AutoProcessor.from_pretrained('openai/clip-vit-large-patch14-336')
        model = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14-336').to(device)
    elif encoder_name == "mae":
        processor = AutoImageProcessor.from_pretrained('facebook/vit-mae-large')
        model = ViTMAEModel.from_pretrained('facebook/vit-mae-large').to(device)
    else:
        raise ValueError(f"Encoder name: {encoder_name} is not implemented.")
    return processor, model


def get_embeddings(data, encoder_name, batch_size=100, num_proc=4, environment="supervised", data_dir="./data", token="class", verbose=True):
    data_emb_dir = os.path.join(data_dir, "embeddings", token, encoder_name)
    if os.path.exists(data_emb_dir):
        environments = [f.name for f in os.scandir(data_emb_dir) if f.is_dir()]
        if (environment in environments):
            if verbose: print(f"Embeddings from encoder '{encoder_name}' token '{token}' already extracted.")
            data_emb_env_dir = os.path.join(data_emb_dir, environment)
            embeddings = Dataset.load_from_disk(data_emb_env_dir)
            return embeddings
    else:
        os.makedirs(data_emb_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    processor, model = get_model(encoder_name, device)
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
        embedding = encoder(batch["image"], model, processor, device, token)
        embeddings.append(embedding)
    embeddings = torch.cat(embeddings, 0)
    embeddings = Dataset.from_dict({encoder_name: embeddings.tolist()})
    embeddings.set_format(type="torch", columns=[encoder_name])
    data_emb_env_dir = os.path.join(data_emb_dir, environment)
    embeddings.save_to_disk(data_emb_env_dir)
    if verbose: print(f"Embeddings from encoder '{encoder_name}' token '{token}' computed and saved correctly.")
    return embeddings


def encoder(x, model, processor, device, token="class"):
    inputs = processor(images=x, return_tensors="pt").to(device)
    outputs = model(**inputs, output_hidden_states=True)
    encoder_name_full = model.config._name_or_path
    if ("vit" in encoder_name_full) or ("dino" in encoder_name_full) or ("siglip" in encoder_name_full):
        if token=="class":
            emb = outputs.hidden_states[-1][:, 0]
        elif token=="mean":
            emb = outputs.hidden_states[-1][:,1:].mean(dim=1)
        else:
            raise ValueError("Token criteria not recognized. Please select between: 'class', 'mean'.")
    elif ("resnet" in encoder_name_full):
        emb = outputs.hidden_states[-1].mean(dim=[2,3])
    else:
        raise ValueError(f"Unkown model class: {encoder_name_full}")
    return emb.to("cpu")

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, task):
        super().__init__()
        self.task = task
        if task=="all":
            output_size = 2
        elif task=="sum":
            output_size = 3
        else:
            output_size = 1
        self.output_size = output_size
        self.model = nn.Sequential(
                            nn.Linear(input_size, 2*hidden_size),
                            nn.ReLU(),
                            #nn.Linear(2*hidden_size, hidden_size),
                            #nn.ReLU(),
                            nn.Linear(2*hidden_size, output_size),
                        )
        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, X):
        return self.model(X) # [-1.8, 0.4]
    def probs(self, X):
        if self.task=="sum":
            return self.model(X).softmax(dim=-1) # [0.7, 0.1, 0.2]
        else:
            return self.model(X).sigmoid() # [0.8, 0.4]
    def pred(self, X):
        if self.task=="sum":
            return torch.argmax(self.model(X), dim=-1) # [0]
        else:
            return self.probs(X).round() # [1, 0]
    def cond_exp(self, X):
        if self.task=="sum":
            values = torch.tensor(range(3)).float().to(self.device)
            probs = self.probs(X)
            return torch.matmul(probs, values) # [0.5]
        else:
            return self.probs(X) # [0.8, 0.4]
        
    
