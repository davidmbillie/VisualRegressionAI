import torch
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.transforms import functional as F
import cv2
import os
import json

class GridOcularisDataset(torch.utils.data.Dataset):
    def __init__(self, coco_json_path, image_dir, transforms=None):
        with open(coco_json_path) as f:
            coco_data = json.load(f)
        self.image_dir = image_dir
        self.images = {img['id']: img for img in coco_data['images']}
        self.annotations = {}
        for ann in coco_data['annotations']:
            self.annotations.setdefault(ann['image_id'], []).append(ann)
        self.categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
        self.transforms = transforms
        self.ids = list(self.images.keys())

    def __getitem__(self, index):
        img_id = self.ids[index]
        image_info = self.images[img_id]
        img_path = os.path.join(self.image_dir, image_info['file_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = F.to_tensor(img)

        annots = self.annotations.get(img_id, [])
        boxes = []
        labels = []
        for ann in annots:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64),
            'image_id': torch.tensor([img_id])
        }

        return img, target

    def __len__(self):
        return len(self.ids)

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    dataset = GridOcularisDataset("annotations/grid_ocularis_annotations.json", "images/")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    model = ssdlite320_mobilenet_v3_large(pretrained=True)
    model.train()

    # Determine in_channels dynamically
    model.backbone.eval()
    with torch.no_grad():
        dummy_input = torch.rand(1, 3, 320, 320)
        dummy_output = model.backbone(dummy_input)
        in_channels = [f.shape[1] for f in dummy_output.values()]
    model.backbone.train()

    num_classes = 3  # background + P + Y
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(in_channels, num_anchors, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    for epoch in range(5):  # Train for 5 epochs
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}, Loss: {losses.item():.4f}")

    # Save model
    torch.save(model.state_dict(), "models/grid_ocularis_model.pth")

if __name__ == "__main__":
    main()
