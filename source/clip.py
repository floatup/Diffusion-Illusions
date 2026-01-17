import torch
import numpy as np
from typing import Union, List
from transformers import AutoProcessor, CLIPModel, CLIPProcessor
import rp

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def _clip_normalize_image(image: torch.Tensor) -> torch.Tensor:
    assert isinstance(image, torch.Tensor), "image should be a torch.Tensor"
    assert image.ndim == 3, "image should be in CHW form"
    assert image.shape[0] == 3, "image should be rgb"
    assert image.min() >= 0 and image.max() <= 1, "image should have values between 0 and 1"

    mean = torch.tensor(clip_processor.feature_extractor.image_mean).to(image.device)
    std = torch.tensor(clip_processor.feature_extractor.image_std).to(image.device)

    norm_image = rp.torch_resize_image(image, (224, 224))
    norm_image = norm_image[None]
    norm_image = (norm_image - mean[None, :, None, None]) / std[None, :, None, None]
    return norm_image

def get_clip_logits(image: Union[torch.Tensor, np.ndarray], prompts: Union[List[str], str]) -> Union[torch.Tensor, np.ndarray]:
    """
    Takes a torch image and a list of prompt strings and returns a vector of log-likelihoods.
    The gradients can be propogated back into the image.

    Can also take in a numpy image. In this case, it will return a numpy vector instead of a torch vector.

    Parameters:
        image (Union[torch.Tensor, np.ndarray]): An image in CHW format if it's a torch.Tensor or in HWC format if it's a numpy array.
        prompts (List[str]): A list of prompt strings.

    Returns:
        Union[torch.Tensor, np.ndarray]: Vector of log-likelihoods in the form of a torch.Tensor if the input is a torch.Tensor or a numpy array if the input is a numpy array.
    """
    
    if isinstance(prompts, str):
        prompts=[prompts]
    
    if rp.is_image(image):
        # This block adds compatiability for numpy images
        # If given a numpy image, it will output a numpy vector of logits
        
        device = "cuda" # In the future, perhaps be smarter about the device selection
        image = rp.as_rgb_image  (image)
        image = rp.as_float_image(image)
        image = rp.as_torch_image(image)
        image = image.to(device)
        
        output = get_clip_logits(image, prompts)
        output = rp.as_numpy_array(output)
        assert output.ndim == 1, "output is a vector"
        
        return output
        
    #Input assertions
    assert isinstance(image,torch.Tensor), 'image should be a torch.Tensor'
    assert image.ndim==3, 'image should be in CHW form'
    assert image.shape[0]==3, 'image should be rgb'
    assert image.min()>=0, 'image should have values between 0 and 1'
    assert image.max()<=1, 'image should have values between 0 and 1'
    assert prompts, 'must have at least one prompt'
    assert all(isinstance(x,str) for x in prompts), 'all prompts must be strings'

    #This stupid clip_processor converts our image into a PIL image in the middle of its pipeline
    #This destroys the gradients; so the rest of the function will be spent fixing that
    image_hwc = image.permute(1,2,0) # (H,W,3)
    inputs = clip_processor(text=list(prompts), images=image_hwc.detach().cpu(), return_tensors="pt", padding=True)

    #There is a specific mean and std this clip_model expects
    #Normalize the image the way the clip_processor does
    norm_image = _clip_normalize_image(image)
    norm_image = norm_image.type_as(inputs["pixel_values"])
    inputs["pixel_values"] = norm_image
    
    #Put all input tensors on the same device
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            inputs[key]=inputs[key].to(image.device)

    #Calculate image-text similarity score
    outputs = clip_model.to(image.device)(**inputs) # Move the clip_model to the device we need on the fly
    logits_per_image = outputs.logits_per_image  # The image-text similarity score
    
    assert logits_per_image.shape == (1, len(prompts),)
    
    output = logits_per_image[0]
    assert output.shape == (len(prompts),)
    
    return output

def get_clip_image_similarity(
    image: Union[torch.Tensor, np.ndarray],
    prompt_image: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    """
    Returns cosine similarity between two images in CLIP image-embedding space.
    The gradients can be propagated back into the first image.
    """
    if rp.is_image(image):
        device = "cuda"
        image = rp.as_rgb_image(image)
        image = rp.as_float_image(image)
        image = rp.as_torch_image(image).to(device)
        output = get_clip_image_similarity(image, prompt_image)
        return rp.as_numpy_array(output)

    assert isinstance(image, torch.Tensor), "image should be a torch.Tensor"
    assert image.ndim == 3, "image should be in CHW form"

    if rp.is_image(prompt_image):
        prompt_image = rp.as_rgb_image(prompt_image)
        prompt_image = rp.as_float_image(prompt_image)
        prompt_image = rp.as_torch_image(prompt_image).to(image.device)
    assert isinstance(prompt_image, torch.Tensor), "prompt_image should be a torch.Tensor"
    assert prompt_image.ndim == 3, "prompt_image should be in CHW form"

    image_inputs = _clip_normalize_image(image)
    prompt_inputs = _clip_normalize_image(prompt_image.detach())

    image_features = clip_model.to(image.device).get_image_features(pixel_values=image_inputs)
    prompt_features = clip_model.to(image.device).get_image_features(pixel_values=prompt_inputs)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    prompt_features = prompt_features / prompt_features.norm(dim=-1, keepdim=True)

    similarity = (image_features * prompt_features).sum(dim=-1)
    assert similarity.shape == (1,)
    return similarity[0]
