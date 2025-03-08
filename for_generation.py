import json
import random

import pyrallis
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.for_pipeline import FORPipeline
from src.config import FOConfig
from src.dascore_metric import VQAModel, compute_dascores


def load_model(config: FOConfig):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if config.sd_2_1:
        stable_diffusion_version = "stabilityai/stable-diffusion-2-1-base"
    else:
        stable_diffusion_version = "CompVis/stable-diffusion-v1-4"
    stable = FORPipeline.from_pretrained(stable_diffusion_version).to(device)
    return stable


def run_on_prompt(prompt,
                  model: FORPipeline,
                  seed,
                  config: FOConfig,
                  run_standard_sd,
                  constraints,
                  parsed_input,
                  vqa_model):
        
    min_index = None
    last_attention_maps = None

    images = []
    losses_all = []
    
    if config.one_stage_gen:
        num_stages = 1
    else:
        num_stages = 2
    for stage in range(num_stages):
        print(f"Stage: {stage + 1}")
        outputs, losses, min_index, _, last_attention_maps = model.for_pipeline(vqa_model=vqa_model,
                                                                                           parsed_input=parsed_input,
                                                                                            constraints=constraints,
                                                                                            prompt=prompt,
                                                                                            attention_res=config.attention_res,
                                                                                            guidance_scale=config.guidance_scale,
                                                                                            num_inference_steps=config.n_inference_steps,
                                                                                            max_iter_to_alter=config.max_iter_to_alter,
                                                                                            run_standard_sd=run_standard_sd,
                                                                                            thresholds=config.thresholds,
                                                                                            scale_factor=config.scale_factor,
                                                                                            scale_range=config.scale_range,
                                                                                            seed=seed,
                                                                                            smooth_attentions=config.smooth_attentions,
                                                                                            sigma=config.sigma,
                                                                                            mode_loss = config.mode_loss,
                                                                                            kernel_size=config.kernel_size,
                                                                                            sd_2_1=config.sd_2_1,
                                                                                            min_index=min_index,
                                                                                            mode_fo=config.mode_fo,
                                                                                            multi_loss=config.multi_loss,
                                                                                            multi_object=config.multi_object,
                                                                                            last_attention_maps=last_attention_maps)
        
        images.append(outputs.images[0])
        losses_all.append(losses)

    scores = []
    for image in images:
        da_score, _ = compute_dascores([image], parsed_input[0]['parsed_input']['questions'], vqa_model) # dascore
        scores.append(da_score[0])
        
    torch.cuda.empty_cache()
    return images, scores, losses_all


def run_for(config: FOConfig):
    stable = load_model(config)
    vqa_model = VQAModel()

        
    output_folder = config.output_path
    output_folder.mkdir(exist_ok=True, parents=True)


    for seed in config.seeds:
        print(f"Seed: {seed}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


        prompt = "a yellow cow is on the right of a blue car"
        constraint = {
                "prompt": "a yellow cow is on the right of a blue car",
                "overlap": [],
                "subset": [
                    {
                        "adjective": "yellow",
                        "entity": "cow"
                    },
                    {
                        "adjective": "blue",
                        "entity": "car"
                    },
                    
            ],
                "spatial": [
                    {
                        "subject": "cow",
                        "spatial": "on the right of",
                        "entity": "car"
                    }
                ]
            }
        parsed_input = {
                    "prompt": "a yellow cow is on the right of a yellow car",
                    "parsed_input": {
                        "assertions": [
                            "there is a yellow cow in the image.",
                            "cow is on the right of a car."
                            "there is a blue car in the image."
                        ],
                        "questions": [
                            "is there a yellow cow in the image?",
                            "is there a cow on the right of a car?",
                            "is there a blue car in the image?"
                        ],
                        "entities": [
                            "cow",
                            "car"
                        ],
                        "type": [
                            "noun",
                            "relation",
                            "noun"
                        ]
                    },
                    "decompose_pormpt": "Decomposable-Caption: [a cow] and [a car]",
                    "id": 0
                }


            
        images, scores, losses_all = run_on_prompt(prompt=prompt,
                                                   model=stable,
                                                   seed=seed,
                                                   config=config,
                                                   run_standard_sd=config.run_standard_sd,
                                                   constraints=constraint,
                                                   parsed_input=[parsed_input],
                                                   vqa_model=vqa_model)  


        max_idx = np.argmax(scores)
        score = scores[max_idx]
        image = images[max_idx]
        losses = losses_all[max_idx]      


        prompt_output_path = config.output_path           
        prompt_output_path.mkdir(exist_ok=True, parents=True)

        images[0].save(prompt_output_path / f'first_{prompt}_{seed}.png') # first image 
            
        if not config.one_stage_gen:
            images[1].save(prompt_output_path / f'second_{prompt}_{seed}.png') # second image 


@pyrallis.wrap()
def main(config: FOConfig):
    run_for(config)
    
if __name__ == '__main__':
    main()