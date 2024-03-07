from transformers import ViTImageProcessor, ViTForImageClassification


def add_embeddings(data, model_name, batch_size=100, num_proc=4, environment='train'):
    processor = ViTImageProcessor.from_pretrained(model_name)
    model = ViTForImageClassification.from_pretrained(model_name)
    data = data.map(lambda x: {"emb1": encoder(x['image'], model, processor)}, batch_size=batch_size, batched=True, num_proc=num_proc)
    data.save_to_disk(f"./data/{environment}")

def encoder(x, model, processor):
    inputs = processor(images=x, return_tensors="pt")
    outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[-1][:,0]  