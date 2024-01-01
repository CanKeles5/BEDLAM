from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation
from PIL import Image
import requests
import numpy as np
import cv2

# load MaskFormer fine-tuned on COCO panoptic segmentation
feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-base-coco")
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-base-coco")

#url = "http://images.cocodataset.org/val2017/000000039769.jpg"
#image = Image.open(requests.get(url, stream=True).raw)
image = Image.open("/home/cankeles/detectron2/projects/DensePose/nyu_human_scenes/furniture_store_0001d/rgb_00180.jpg")
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


# Create a binary mask for pixels equal to the specified integer 'n'
#binary_mask = (predicted_panoptic_map.numpy() == n).astype(np.uint8)

binary_mask = np.zeros((np.array(image).shape[0], np.array(image).shape[1])).astype(np.uint8)
print(f"shape of binary mask: {binary_mask.shape}")

for segment in result['segments_info']:
	if segment['label_id'] == 0:
		binary_mask += (predicted_panoptic_map.numpy() == segment['id']).astype(np.uint8)

print(f"binary min: {binary_mask.min()}, max: {binary_mask.max()}")
# Apply the binary mask to the original RGB image
segmented_image = cv2.bitwise_and(np.array(image), np.array(image), mask=binary_mask)

print(f"segmented_image shape: {segmented_image.shape}")

#cv2.imwrite('path_to_save_segmented_image.jpg', segmented_image)

segmented_image = Image.fromarray(segmented_image)
segmented_image.save('path_to_save_segmented_image.jpg')

print("Saved segmented img")
