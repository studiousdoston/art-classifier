import gradio as gr
import spaces
import torch
from fastai.vision.all import create_vision_model, resnet34
from torchvision import transforms

categories = ["asian", "islamic", "renaissance"]

model = create_vision_model(resnet34, n_out=3, pretrained=False)
model.load_state_dict(torch.load("weights.pth", map_location="cpu"))
model.eval()

tfm = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


@spaces.GPU
def classify_image(img):
    x = tfm(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    return {c: float(p) for c, p in zip(categories, probs)}


# Returns {'asian': 0.99, 'islamic': 0.001, 'renaissance': 0.0004} for the Label bars

demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    title="Art Style Classifier",
    description="Upload a painting: Renaissance, Islamic (Middle-Eastern), or Asian.",
)
demo.launch()
