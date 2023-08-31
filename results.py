import torch
import PIL
from PIL import Image
import torchvision.transforms as transforms
from torchsummary import summary


if __name__ == '__main__':
    model_path = 'model.pth'
    model = torch.load(model_path)
    model.eval() 
    preprocess = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



    image_path = 'test_image/image.jpg' #add your test image to the folder test_image as image.jpg
    image = Image.open(image_path)

    input_preprocess = preprocess(image)
    input_batch = input_preprocess.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    _, predicted_class = torch.max(output, 1)
    print('Predicted class:', predicted_class.item())
    print('Probability:', probabilities[predicted_class].item())
