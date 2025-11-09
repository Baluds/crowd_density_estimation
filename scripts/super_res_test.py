"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""

import argparse
import os
import glob
import cv2

from PIL import Image
import pandas as pd

import numpy as np
import torch as th
import torch.distributed as dist

from einops import rearrange

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from cc_utils.utils import *

from matplotlib import pyplot as plt

from tqdm import tqdm
import random
from numpy import asarray
import os
import threading

import os
import glob
import numpy as np
import torch as th
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
import multiprocessing


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    # print(f"Random seed set as {seed}")

# Assuming DataParameter is defined elsewhere
# from cc_utils.utils import DataParameter


def get_density_new(input_array):

    # Ensure the input is 2D (grayscale image)
  if len(input_array.shape) != 2:
      raise ValueError("The input array must be a 2D grayscale image.")

  # Create an output image with a white background
  output_image = np.zeros((input_array.shape[0], input_array.shape[1], 3), dtype=np.uint8)

  # Find black pixels (value 0) in the input array
  black_pixel_indices = np.column_stack(np.where(input_array == 255))

  # Draw red circles at black pixel locations
  for y, x in black_pixel_indices:
      cv2.circle(output_image, (x, y), radius=1, color=(0, 255, 0), thickness=2)  # Red circle

  return output_image




def main():
    args = create_argparser().parse_args()

    # Ensure that log_dir is set
    if args.log_dir is None:
        args.log_dir = 'output'  # Set a default output directory if not provided
        print(f"log_dir not specified. Using default: {args.log_dir}")

    dist_util.setup_dist()
    logger.configure(dir=args.log_dir)

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
        print(f"Created directory {args.log_dir}")

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cuda"), strict=False
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading data...")
    data = load_data_for_worker(args.data_dir, args.batch_size, args.file)

    logger.log("creating samples...")

    sub_name = 1
    df=pd.DataFrame(columns=['Image Name','Count'])
    for _ in tqdm(os.listdir(args.data_dir)):
        model_kwargs = next(data)

        original_image = model_kwargs['low_res'].clone()
        model_kwargs['original_image'] = original_image  # Save it in model_kwargs
        
        data_parameter = DataParameter(model_kwargs, args)
        
        # model_kwargs['low_res'] = model_kwargs['crops']
        # Move tensors to the correct device
        model_kwargs = {k: v.to(dist_util.dev()) if isinstance(v, th.Tensor) else v for k, v in model_kwargs.items()}
        
        if model_kwargs['low_res'].size(0) == 0:
            continue

        # Ensure data_parameter.name is set
        if not hasattr(data_parameter, 'name') or not data_parameter.name:
            data_parameter.name = model_kwargs.get('name', ['sample'])[0]

        while data_parameter.resample:

            data_parameter.update_cycle()
            # Call the p_sample_loop function and unpack the returned tuple
            # Call the p_sample_loop function and unpack the returned tuple
            samples= diffusion.p_sample_loop(
                model,
                (model_kwargs['low_res'].size(0), args.pred_channels, model_kwargs['low_res'].size(2), model_kwargs['low_res'].size(3)),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )

            # Now access the "sample" key from the samples dictionary
            samples = samples["sample"]

            #data_parameter.evaluate(samples, model_kwargs)

            # Process samples and save results
            samples_np = samples.cpu().numpy()
            for index, sample in enumerate(samples_np):
                sample_image = (sample * 255).transpose(1, 2, 0)
                # print(sample_image.shape)
                # np.save('sample_image.npy',sample_image)
                # sample_image = np.dot(sample_image[:, :, ::-1][..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

                # print("Sample_Image",sample_image[:, :, ::-1])
                sample_image_1=np.where(sample_image[:,:,0]>0,255,0)
                # sample_image_2=np.where(sample_image[:,:,0]>50,255,0)
                # sample_image_3=np.where(sample_image[:,:,0]>100,255,0)

                people_count=np.count_nonzero(sample_image_1==255)
                sample_image_1=get_density_new(sample_image_1)
                # sample_image_2=get_density_new(sample_image_2)
                # sample_image_3=get_density_new(sample_image_3)

                output_filename = os.path.join(args.log_dir, f'{data_parameter.name}')
                df.loc[len(df)]=[output_filename,people_count]
                # Define the text and its properties
                text = f"Count : {people_count}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5  # Adjust for size
                color = (255,255, 255)  # White color in BGR
                thickness = 1  # Thickness of the text

                # # Get the text size to calculate position
                # (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                # image_height, image_width, _ = sample_image_1.shape
                # x = (image_width - text_width) // 2
                # y = (image_height)//0.9

                # Put the text on the image
                cv2.putText(sample_image_1, text, (128,20),font, font_scale, color, thickness, cv2.LINE_AA)

                cv2.imwrite(os.path.join(args.log_dir, f'{data_parameter.name}'), sample_image_1)
                # cv2.imwrite(os.path.join(args.log_dir, f'2_{data_parameter.name}'), sample_image_2)
                # cv2.imwrite(os.path.join(args.log_dir, f'3_{data_parameter.name}'), sample_image_3)  # Save the generated image
                print(f'Saved image {output_filename}')
                sub_name += 1



            # Set resample to False to stop after one cycle
            data_parameter.resample = False

    df.to_csv('Count_Part_B.csv')    

    logger.log("sampling complete")


# def main():
#     args = create_argparser().parse_args()

#     if args.log_dir is None:
#         args.log_dir = 'output'
#         print(f"log_dir not specified. Using default: {args.log_dir}")

#     dist_util.setup_dist()
#     logger.configure(dir=args.log_dir)

#     if not os.path.exists(args.log_dir):
#         os.makedirs(args.log_dir)
#         print(f"Created directory {args.log_dir}")

#     logger.log("creating model...")
#     model, diffusion = sr_create_model_and_diffusion(
#         **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
#     )
#     model.load_state_dict(
#         dist_util.load_state_dict(args.model_path, map_location="cuda"), strict=False
#     )
#     model.to(dist_util.dev())
#     if args.use_fp16:
#         model.convert_to_fp16()
#     model.eval()

#     logger.log("loading data...")
#     data = load_data_for_worker_optimized(args.data_dir, args.batch_size, args.file)

#     logger.log("creating samples...")
#     sub_name = 1

#     for model_kwargs in tqdm(data):
#         model_kwargs = {k: v.to(dist_util.dev()) if isinstance(v, th.Tensor) else v for k, v in model_kwargs.items()}
        
#         if model_kwargs['low_res'].size(0) == 0:
#             continue

#         model_kwargs['original_image'] = model_kwargs['low_res'].clone()
#         data_parameter = DataParameter(model_kwargs, args)

#         if not hasattr(data_parameter, 'name') or not data_parameter.name:
#             data_parameter.name = model_kwargs.get('name', ['sample'])[0]

#         while data_parameter.resample:
#             data_parameter.update_cycle()
#             samples = diffusion.p_sample_loop(
#                 model,
#                 (model_kwargs['low_res'].size(0), args.pred_channels, model_kwargs['low_res'].size(2), model_kwargs['low_res'].size(3)),
#                 clip_denoised=args.clip_denoised,
#                 model_kwargs=model_kwargs,
#             )
#             samples = samples["sample"].cpu().numpy()
#             for index, sample in enumerate(samples):
#                 sample_image = (sample * 255).transpose(1, 2, 0)
#                 sample_image = np.where(sample_image[:, :, 0] > 100, 255, 0)
#                 people_count=np.count_nonzero(sample_image==255)
#                 sample_image = get_density_new(sample_image)
#                 output_filename = os.path.join(args.log_dir, f'{data_parameter.name}')

#                 text = f"Count : {people_count}"
#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 font_scale = 0.5  # Adjust for size
#                 color = (255, 255, 255)  # White color in BGR
#                 thickness = 1  # Thickness of the text



#                 cv2.putText(sample_image, text, (128,20),font, font_scale, color, thickness, cv2.LINE_AA)
#                 cv2.imwrite(output_filename, sample_image)
#                 sub_name += 1
#             data_parameter.resample = False

#     logger.log("sampling complete")


def evaluate_samples(samples, model_kwargs, crowd_count, order, result, mae, dims, cycles):

    samples = samples.cpu().numpy()
    for index in range(order.size):
        p_result, p_mae = evaluate_sample(samples[index], crowd_count[order[index]], name=f'{index}_{cycles}')
        if np.abs(p_mae) < np.abs(mae[order[index]]):
            result[order[index]] = p_result
            mae[order[index]] = p_mae
    
    indices = np.where(np.abs(mae[order]) > 0)
    order = order[indices]
    model_kwargs['low_res'] = model_kwargs['low_res'][indices]

    pred_count = combine_crops(result, model_kwargs, dims, mae)['pred_count']
    del model_kwargs['pred_count'], model_kwargs['result']
    
    resample = False if len(order) == 0 else True
    resample = False if np.sum(np.abs(mae[order])) < 25 else True

    print(f'mae: {mae}')
    print(f'cum mae: {np.sum(np.abs(mae[order]))} comb mae: {np.abs(pred_count - np.sum(crowd_count))} cycle: {cycles}')

    return model_kwargs, order, result, mae, resample


def evaluate_sample(sample, count, name=None):
    
    sample = sample.squeeze()
    sample = (sample + 1)
    sample = (sample / (sample.max() + 1e-8)) * 255
    sample = sample.clip(0, 255).astype(np.uint8)
    sample = remove_background(sample)
    
    pred_count = get_circle_count(sample, name=name, draw=True)

    return sample, pred_count - count


def remove_background(crop):
    def count_colors(image):

        colors_count = {}
        # Flattens the 2D single channel array so as to make it easier to iterate over it
        image = image.flatten()

        for i in range(len(image)):
            I = str(int(image[i]))
            if I in colors_count:
                colors_count[I] += 1
            else:
                colors_count[I] = 1
        
        return int(max(colors_count, key=colors_count.get)) + 5

    count = count_colors(crop)
    crop = crop * (crop > count)

    return crop


def get_circle_count(image, threshold=0, draw=False, name=None):

    # Denoising
    denoisedImg = cv2.fastNlMeansDenoising(image)

    # Threshold (binary image)
    th, threshedImg = cv2.threshold(denoisedImg, threshold, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Perform morphological transformations using an erosion and dilation as basic operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morphImg = cv2.morphologyEx(threshedImg, cv2.MORPH_OPEN, kernel)

    # Find and draw contours
    contours, _ = cv2.findContours(morphImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if draw:
        contoursImg = np.zeros_like(morphImg)
        contoursImg = np.repeat(contoursImg[:, :, np.newaxis], 3, -1)
        for point in contours:
            x, y = point.squeeze().mean(0)
            if x == 127.5 and y == 127.5:
                continue
            cv2.circle(contoursImg, (int(x), int(y)), radius=3, thickness=-1, color=(255, 255, 255))
        threshedImg = np.repeat(threshedImg[:, :, np.newaxis], 3, -1)
        morphImg = np.repeat(morphImg[:, :, np.newaxis], 3, -1)
        image = np.concatenate([contoursImg, threshedImg, morphImg], axis=1)
        if not os.path.exists('experiments/target_test'):
            os.makedirs('experiments/target_test')
        cv2.imwrite(f'experiments/target_test/{name}_image.jpg', image)
    return max(len(contours) - 1, 0)  # remove the border


def create_crops(model_kwargs, args):
    
    image = model_kwargs['low_res']
    density = model_kwargs['high_res']

    model_kwargs['dims'] = density.shape[-2:]

    # create a padded image
    image = create_padded_image(image, args.large_size)
    density = create_padded_image(density, args.large_size)

    model_kwargs['low_res'] = image
    model_kwargs['high_res'] = density

    model_kwargs['crowd_count'] = th.sum((model_kwargs['high_res'] + 1) * 0.5 * th.tensor(args.normalizer).view(1, -1, 1, 1), dim=(1, 2, 3)).cpu().numpy()
    model_kwargs['order'] = np.arange(model_kwargs['low_res'].size(0))

    model_kwargs = organize_crops(model_kwargs)
        
    return model_kwargs


def organize_crops(model_kwargs):
    indices = np.where(model_kwargs['crowd_count'] > 0)
    model_kwargs['order'] = model_kwargs['order'][indices]
    model_kwargs['low_res'] = model_kwargs['low_res'][indices]

    return model_kwargs
        

def create_padded_image(image, image_size):

    _, c, h, w = image.shape
    p1, p2 = (h - 1 + image_size) // image_size, (w - 1 + image_size) // image_size
    pad_image = th.full((1, c, p1 * image_size, p2 * image_size), -1, dtype=image.dtype)

    start_h, start_w = (p1 * image_size - h) // 2, (p2 * image_size - w) // 2
    end_h, end_w = h + start_h, w + start_w

    pad_image[:, :, start_h:end_h, start_w:end_w] = image
    pad_image = rearrange(pad_image, 'n c (p1 h) (p2 w) -> (n p1 p2) c h w', p1=p1, p2=p2)

    return pad_image


def combine_crops(crops, model_kwargs, dims, mae, image_size=256):

    crops = th.tensor(crops).squeeze()
    p1, p2 = (dims[0] - 1 + image_size) // image_size, (dims[1] - 1 + image_size) // image_size
    crops = rearrange(crops, '(p1 p2) h w -> (p1 h) (p2 w)', p1=p1, p2=p2)
    crops = crops.numpy()
    
    start_h, start_w = (crops.shape[0] - dims[0]) // 2, (crops.shape[1] - dims[1]) // 2
    end_h, end_w = start_h + dims[0], start_w + dims[1]
    model_kwargs['result'] = crops[start_h:end_h, start_w:end_w]

    model_kwargs['pred_count'] = get_circle_count(crops.astype(np.uint8))
    
    return model_kwargs



def save_visuals(model_kwargs, args):
    # Use the original full image
    crowd_img = model_kwargs["original_image"]

    # Ensure the image is in the correct format (uint8 and correct dimensions)
    if crowd_img.dtype != np.uint8:
        crowd_img = crowd_img.astype(np.uint8)
    if len(crowd_img.shape) == 2 or (len(crowd_img.shape) == 3 and crowd_img.shape[2] != 3):
        crowd_img = np.repeat(crowd_img[:, :, np.newaxis], 3, axis=-1)

    # Extract and process the model's predicted output
    sample = model_kwargs['result']
    if isinstance(sample, th.Tensor):
        sample = sample.cpu().numpy()
    sample = sample.squeeze()

    # Ensure sample is at least 3D
    if sample.ndim == 2:
        sample = sample[:, :, np.newaxis]

    sample = sample.clip(0, 255).astype(np.uint8)

    # Ensure the sample has 3 channels
    if args.pred_channels == 1 or sample.shape[2] == 1:
        sample = np.repeat(sample, 3, axis=-1)

    # Add a red gap between images for visual separation
    gap = 5
    red_gap = np.zeros((crowd_img.shape[0], gap, 3), dtype=int)
    red_gap[:, :, 0] = 255  # Set the red channel to maximum

    # Print shapes for debugging
    print("crowd_img shape:", crowd_img.shape)
    print("sample shape:", sample.shape)
    print("red_gap shape:", red_gap.shape)

    # Concatenate the images side by side
    req_image = np.concatenate([crowd_img, red_gap, sample], axis=1)

    # Construct the output filename
    path = f'{model_kwargs["name"][0].split(".")[0]}.jpg'
    print(f'Saving {path}')

    # Save the concatenated image
    cv2.imwrite(os.path.join(args.log_dir, path), req_image[:, :, ::-1])  # Convert RGB to BGR for OpenCV


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        per_samples=1,
        use_ddim=True,
        data_dir="",  # data directory
        model_path="",  # model path
        log_dir=None,  # output directory
        normalizer='',  # density normalizer
        pred_channels=3,
        thresh=200,  # threshold for circle count
        file='',  # specific file number to test
        overlap=0.5,  # overlapping ratio for image crops
        large_size=256,  # Added large_size parameter
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser



def load_data_for_worker(base_samples, batch_size, file_name, class_cond=False):
    if file_name == '':
        img_list = sorted(glob.glob(os.path.join(base_samples, '*.jpg')))
    else:
        img_list = sorted(glob.glob(os.path.join(base_samples, f'*/*/{file_name}-*.jpg')))

    image_arr = []
    for file in img_list:
        # Load and process the image
        image = Image.open(file)
        image_arr.append(np.asarray(image))

    rank = dist.get_rank()
    num_ranks = dist.get_world_size()
    buffer = []
    label_buffer = []
    name_buffer = []
    for i in range(rank, len(image_arr), num_ranks):
        buffer.append(image_arr[i])
        name_buffer.append(os.path.basename(img_list[i]))
        if class_cond:
            pass  # Placeholder for class labels if used
        if len(buffer) == batch_size:
            batch = th.from_numpy(np.stack(buffer)).float()
            batch = batch / 127.5 - 1.0
            batch = batch.permute(0, 3, 1, 2)
            res = dict(
                low_res=batch,
                name=name_buffer
            )
            if class_cond:
                res["y"] = th.from_numpy(np.stack(label_buffer))
            yield res
            buffer, label_buffer, name_buffer = [], [], []
    # If buffer is not empty after the loop, yield the remaining data
    if len(buffer) > 0:
        batch = th.from_numpy(np.stack(buffer)).float()
        batch = batch / 127.5 - 1.0
        batch = batch.permute(0, 3, 1, 2)
        res = dict(
            low_res=batch,
            name=name_buffer
        )
        if class_cond:
            res["y"] = th.from_numpy(np.stack(label_buffer))
        yield res

# class ImageDataset(Dataset):
#     def __init__(self, base_samples, file_name, class_cond=False):
#         if file_name == '':
#             self.img_list = sorted(glob.glob(os.path.join(base_samples, '*.jpg')))
#         else:
#             self.img_list = sorted(glob.glob(os.path.join(base_samples, f'*/*/{file_name}-*.jpg')))
#         self.class_cond = class_cond

#     def __len__(self):
#         return len(self.img_list)

#     def __getitem__(self, idx):
#         file = self.img_list[idx]
#         image = Image.open(file)
#         image = np.asarray(image)
#         image = image / 127.5 - 1.0  # Normalize image
#         image = np.transpose(image, (2, 0, 1))  # Convert to CxHxW
#         sample = {
#             'low_res': th.tensor(image, dtype=th.float32),
#             'name': os.path.basename(file)
#         }
#         if self.class_cond:
#             # Add class labels if needed
#             sample['y'] = 0  # Placeholder for class labels
#         return sample

# def load_data_for_worker_optimized(base_samples, batch_size, file_name, class_cond=False):
#     dataset = ImageDataset(base_samples, file_name, class_cond)
#     dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=multiprocessing.cpu_count(), shuffle=False)
#     for batch in dataloader:
#         yield batch


if __name__ == "__main__":
    main()
