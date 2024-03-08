from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import AutoImageProcessor, ResNetForImageClassification
from transformers import AutoImageProcessor, AutoModel
from transformers import AutoProcessor, CLIPVisionModel
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

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


def add_embeddings(data, model_name, batch_size=100, num_proc=4, environment="train"):
    if model_name in data.features.keys():
        print(f"Embedding {model_name} already extracted.")
    else:
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
        data = data.add_column(model_name, embeddings.tolist())
        data.save_to_disk(f"./data/{environment}")

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
