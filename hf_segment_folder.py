from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from PIL import Image
import requests
import numpy as np
import cv2
import glob
import os


# load MaskFormer fine-tuned on COCO panoptic segmentation
feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-base-coco")
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-coco")

root_pth = "/home/cankeles/bedlam_smpl/BEDLAM/nyu_human_scenes"
save_root = "/home/cankeles/bedlam_smpl/BEDLAM/nyu_human_masks"
folders = glob.glob(os.path.join(root_pth, "*"))

for folder in folders:
    for img_pth in glob.glob(os.path.join(folder, "*")):
        
        if "sync_depth" in img_pth:
            continue
        else:
            print(f"img_pth: {img_pth}")
        
        image = Image.open(img_pth)
        inputs = feature_extractor(images=image, return_tensors="pt")

        outputs = model(**inputs)
        # model predicts class_queries_logits of shape `(batch_size, num_queries)`
        # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
        class_queries_logits = outputs.class_queries_logits
        masks_queries_logits = outputs.masks_queries_logits

        # you can pass them to feature_extractor for postprocessing
        result = feature_extractor.post_process_panoptic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        # we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
        predicted_panoptic_map = result["segmentation"]

        binary_mask = np.zeros((np.array(image).shape[0], np.array(image).shape[1])).astype(np.uint8)

        for segment in result['segments_info']:
            if segment['label_id'] == 0:
                binary_mask += (predicted_panoptic_map.numpy() == segment['id']).astype(np.uint8)

        # Apply the binary mask to the original RGB image
        segmented_image = cv2.bitwise_and(np.array(image), np.array(image), mask=binary_mask)
        
        img_pth = os.path.normpath(img_pth).split(os.sep)

        save_pth = os.path.join(save_root, img_pth[-2]) #, f"{img_pth[-1][:-4]}_mask.jpg")
        if not os.path.exists(save_pth):
            os.mkdir(save_pth)
            print(f"mkdir {save_pth}")

        save_pth = os.path.join(save_pth, f"{img_pth[-1][:-4]}_mask.jpg")
        segmented_image = Image.fromarray(segmented_image)
        segmented_image.save(save_pth)

        print(f"Saved segmented img to: {save_pth}")
