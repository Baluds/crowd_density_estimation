import argparse
import os
import numpy as np
import torch as th
import cv2
from PIL import Image
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

def load_single_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return np.asarray(image)

def preprocess_image(image, image_size):
    image = cv2.resize(image, (image_size, image_size))
    image = image / 127.5 - 1.0
    return image

def create_argparser():
    defaults = dict(
        # Model architecture
        attention_resolutions="32,16,8",
        class_cond=False,
        diffusion_steps=1000,
        large_size=256,
        small_size=256,
        learn_sigma=True,
        noise_schedule="linear",
        num_channels=192,
        num_head_channels=64,
        num_res_blocks=2,
        resblock_updown=True,
        use_fp16=True,
        use_scale_shift_norm=True,
        
        # Training/Sampling parameters
        batch_size=1,
        model_path="",
        clip_denoised=True,
        num_samples=1,
        use_ddim=False,
        
        # Additional parameters
        image_size=256,
        log_dir="logs",
        normalizer=0.8,
        pred_channels=1,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def main(args):
    # Set up logging
    os.makedirs(args.log_dir, exist_ok=True)
    logger.configure(dir=args.log_dir)

    # Load the model
    logger.log("Creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # Load and preprocess the image
    image = load_single_image('/content/drive/MyDrive/cs682_project/crowddiff/data_sample/test_data/images/IMG_7.jpg')  # Replace with your image path
    processed_image = preprocess_image(image, args.large_size)

    # Prepare the input tensor
    input_tensor = th.from_numpy(processed_image).permute(2, 0, 1).unsqueeze(0).float().to(dist_util.dev())

    # Generate samples
    logger.log("Generating samples...")
    model_kwargs = {'low_res': input_tensor}
    sample = diffusion.p_sample_loop(
        model,
        (args.batch_size, args.pred_channels, args.large_size, args.large_size),
        clip_denoised=args.clip_denoised,
        model_kwargs=model_kwargs,
    )

    # Process and save the output
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous().cpu().numpy()
    
    output_path = os.path.join(args.log_dir, 'output_image.png')
    cv2.imwrite(output_path, sample[0])
    logger.log(f"Sample generation complete. Output saved as {output_path}")

if __name__ == "__main__":
    parser = create_argparser()
    args = parser.parse_args()
    main(args)