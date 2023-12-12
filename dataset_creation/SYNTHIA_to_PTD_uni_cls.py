import sys
import torch
import numpy as np
import gradio as gr
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat, rearrange
from pytorch_lightning import seed_everything
from imwatermark import WatermarkEncoder

from scripts.txt2img import put_watermark
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.data.util import AddMiDaS

torch.set_grad_enabled(False)


def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    print(config.model)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)
    return sampler


def make_batch_sd(
        image,
        txt,
        device,
        num_samples=1,
        model_type="dpt_hybrid"
):
    image = np.array(image.convert("RGB"))
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    # sample['jpg'] is tensor hwc in [-1, 1] at this point
    midas_trafo = AddMiDaS(model_type=model_type)
    batch = {
        "jpg": image,
        "txt": num_samples * [txt],
    }
    batch = midas_trafo(batch)
    batch["jpg"] = rearrange(batch["jpg"], 'h w c -> 1 c h w')
    batch["jpg"] = repeat(batch["jpg"].to(device=device),
                          "1 ... -> n ...", n=num_samples)
    batch["midas_in"] = repeat(torch.from_numpy(batch["midas_in"][None, ...]).to(
        device=device), "1 ... -> n ...", n=num_samples)
    return batch


def paint(sampler, image, prompt, t_enc, seed, scale, num_samples=1, callback=None,
          do_full_sample=False):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model
    seed_everything(seed)

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "SDV2"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    with torch.no_grad(),\
            torch.autocast("cuda"):
        batch = make_batch_sd(
            image, txt=prompt, device=device, num_samples=num_samples)
        z = model.get_first_stage_encoding(model.encode_first_stage(
            batch[model.first_stage_key]))  # move to latent space
        c = model.cond_stage_model.encode(batch["txt"])
        c_cat = list()
        for ck in model.concat_keys:
            cc = batch[ck]
            cc = model.depth_model(cc)
            depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3],
                                                                                           keepdim=True)
            display_depth = (cc - depth_min) / (depth_max - depth_min)
            depth_image = Image.fromarray(
                (display_depth[0, 0, ...].cpu().numpy() * 255.).astype(np.uint8))
            cc = torch.nn.functional.interpolate(
                cc,
                size=z.shape[2:],
                mode="bicubic",
                align_corners=False,
            )
            depth_min, depth_max = torch.amin(cc, dim=[1, 2, 3], keepdim=True), torch.amax(cc, dim=[1, 2, 3],
                                                                                           keepdim=True)
            cc = 2. * (cc - depth_min) / (depth_max - depth_min) - 1.
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)
        # cond
        cond = {"c_concat": [c_cat], "c_crossattn": [c]}

        # uncond cond
        uc_cross = model.get_unconditional_conditioning(num_samples, "")
        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
        if not do_full_sample:
            # encode (scaled latent)
            z_enc = sampler.stochastic_encode(
                z, torch.tensor([t_enc] * num_samples).to(model.device))
        else:
            z_enc = torch.randn_like(z)
        # decode it
        samples = sampler.decode(z_enc, cond, t_enc, unconditional_guidance_scale=scale,
                                 unconditional_conditioning=uc_full, callback=callback)
        x_samples_ddim = model.decode_first_stage(samples)
        result = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
        result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
    return [depth_image] + [put_watermark(Image.fromarray(img.astype(np.uint8)), wm_encoder) for img in result]


def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded


def predict(input_image, prompt, steps, num_samples, scale, seed, eta, strength):
    init_image = input_image.convert("RGB")
    image = pad_image(init_image)  # resize to integer multiple of 32

    sampler.make_schedule(steps, ddim_eta=eta, verbose=True)
    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    do_full_sample = strength == 1.
    t_enc = min(int(strength * steps), steps-1)
    result = paint(
        sampler=sampler,
        image=image,
        prompt=prompt,
        t_enc=t_enc,
        seed=seed,
        scale=scale,
        num_samples=num_samples,
        callback=None,
        do_full_sample=do_full_sample
    )
    return result


sampler = initialize_model(sys.argv[1], sys.argv[2])

from PIL import Image
import os
import random
def process_image(file_path, prompt, ddim_steps, num_samples, scale, seed, eta, strength):
    input_image = Image.open(file_path)
    width, height = input_image.size
    new_width = int(width)# * 0.5)
    new_height = int(height)# * 0.5)
    # Resize the image
    input_image = input_image.resize((new_width, new_height))

    result = predict(input_image, prompt, ddim_steps, num_samples, scale, seed, eta, strength)
    return result

def save_result(result, output_file_path, original_size):
    # Upsample the result back to the original size
    result_image = result[1].resize(original_size)
    result_image.save(output_file_path)

prompt_1 = "A high quality photo; europe" # of a german traffic scene"
prompt_2 = "A high quality photo; europe;Highway"
prompt_3 = "A high quality photo; europe;City"
prompt_4 = "A high quality photo; germany" # of a german traffic scene"
prompt_5 = "A high quality photo; germany;Highway"
prompt_6 = "A high quality photo; germany;City"
promts=[prompt_1,prompt_2,prompt_3,prompt_4,prompt_5,prompt_6]
ddim_steps = 25 #50
num_samples = 1
scale = 9 # 9
seed = 0
eta = 0
strength = 0.9

# Replace with the actual path to the folder containing PNG images
input_folder = 'Datasets/Synthia/train/RAND_CITYSCAPES/RGB'
input_folder_label = 'Synthia/train/RAND_CITYSCAPES/GT/LABELS'
# Replace with the actual path to the folder where you want to save the processed images
output_folder = 'pseudo_target_domain/SYNTHIA/uni_cls_rand_location'

CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
            'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
            'person', 'rider', 'car', 'truck', 'bus', 'tram/ train/ trolley', 'motorcycle',
            'bicycle') 
#  the following classes are not in Synthia terrain, truck, train, 
# List all PNG files in the input folder
png_files = [file for file in os.listdir(input_folder) if file.endswith('.png')]

hist = np.zeros(19)
hist[9] = 100000000000000
hist[14] = 100000000000000
hist[16] = 100000000000000

for png_file in png_files:
    file_path = os.path.join(input_folder, png_file)
    label_path = os.path.join(input_folder_label, png_file.replace('.png', '_labelTrainIds.png'))
    label = Image.open(label_path)
    label_array = np.array(label)
    classes_present = np.unique(label_array)
    classes_present = [i for i in classes_present if i != 255]
    addressed_classes = [CLASSES[i] for i in classes_present]
    addressed_classes_string = ', '.join(addressed_classes)
    print(classes_present, addressed_classes_string)

    # Update the histogram with the current image's class occurrences
    hist[classes_present] +=1
    current_least_often_cls = np.argmin(hist)
    current_least_often_cls_string = CLASSES[current_least_often_cls]
    hist[np.argmin(hist)] +=1

    # Process the image
    random.seed()
    prompt = random.choice(promts)+ ", " + current_least_often_cls_string + ", " + addressed_classes_string    
    print(prompt)
    result = process_image(file_path, prompt, ddim_steps, num_samples, scale, seed, eta, strength)
    
    # Get the original size of the image
    original_size = Image.open(file_path).size

    # Save the result in the output folder with the same filename
    output_file_path = os.path.join(output_folder, png_file)
    save_result(result, output_file_path, original_size)
