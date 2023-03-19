import torchvision
import torch
import argparse
from cv2 import imwrite
import detect_utils
#from basemodel.create_fasterrcnn_model import create_model
from PIL import Image

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input image/video')
parser.add_argument('-m', '--min-size', dest='min_size', default=800, help='minimum input size for the FasterRCNN network')
args = vars(parser.parse_args())


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#NUM_CLASSES = 10

# load model standard model from torchvision
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='COCO_V1',min_size=args['min_size'])
model.eval().to(device)
# load retrained model
'''
create_model = create_model["fasterrcnn_resnet50_fpn_v2"]
model = create_model(num_classes=NUM_CLASSES, coco_model=False)
checkpoint = torch.load("model/best_model (3).pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device).eval()
'''
image = Image.open(args['input'])


boxes, classes, labels = detect_utils.predict(image, model, device, 0.8)
detected_images = detect_utils.get_labels(boxes, classes)
print(boxes, classes)

# save result as image
image = detect_utils.draw_boxes(boxes, classes, labels, image)
#cv2.imshow('Image', image)
save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{args['min_size']}"

imwrite(f"outputs/{save_name}.jpg", image)
