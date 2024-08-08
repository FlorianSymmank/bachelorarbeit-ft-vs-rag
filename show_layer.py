from transformers import AutoModel

# Lade das Basismodell
model = AutoModel.from_pretrained("Qwen/Qwen2-0.5B-Instruct")

# Zeige alle Module an
for name, module in model.named_modules():
    print(name)