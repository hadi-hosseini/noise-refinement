from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

@dataclass
class FOConfig:
    # Whether to use Stable Diffusion v2.1
    sd_2_1: bool = False
    # Whether to run Standard Stable Diffusion 
    run_standard_sd: bool = False
    # Which token indices to alter with attend-and-excite
    token_indices: List[int] = None
    # Which random seeds to use when generating
    seeds: List[int] = field(default_factory=lambda: [140])
    # Path to save all outputs to
    output_path: Path = Path('./outputs/images')
    # Number of denoising steps
    n_inference_steps: int = 50
    # Text guidance scale
    guidance_scale: float = 7.5
    # Number of denoising steps to apply attend-and-excite
    max_iter_to_alter: int = 25
    # Resolution of UNet to compute attention maps over
    attention_res: int = 16
    # Dictionary defining the iterations and desired thresholds to apply iterative latent refinement in
    thresholds: Dict[int, float] = field(default_factory=lambda: {0: 0.05, 10: 0.5, 20: 0.8})
    # Scale factor for updating the denoised latent z_t
    scale_factor: int = 20
    # Start and end values used for scaling the scale factor - decays linearly with the denoising timestep
    scale_range: tuple = field(default_factory=lambda: (1.0, 0.5))
    # Whether to apply the Gaussian smoothing before computing the maximum attention value for each subject token
    smooth_attentions: bool = True
    # Standard deviation for the Gaussian smoothing
    sigma: float = 0.5
    # Kernel size for the Gaussian smoothing
    kernel_size: int = 3
    # Whether to save cross attention maps for the final results
    save_cross_attention_maps: bool = False
    # Specify which mode to run 1. comtie  2. attend-and-excite 3. stable diffusion
    mode_generation: int = 1
    # Specify which loss mode to run 1. entity loss  2. adjective loss 3. spatial relation loss 4. intransitive verb loss 5. all of them
    mode_loss: int = 1
    # Specify which dataset to run 1. Compbench 2. TIFA 3. HRS
    mode_dataset: int = 1
    # Specify which evaluator to run 1. comtie 2. TIFA 3. both
    mode_evaluator: int = 1
    # Specify which form of optimization 1. FR depends on one_stage_gen
    mode_fo: int = 1
    # Specify to on/off the multi loss 
    multi_loss: bool = True
    # Specify to on/off the multi object
    multi_object: bool = True
    #### One stage generation
    one_stage_gen = False

    tifa_vqa_model_name: str = "mplug-large"

    compbench_parsed_inputs_path = '/home/hadi/diffusion/fr/inputs/parsed_inputs_gpt4_mod.json'
    compbench_constraints_path = '/home/hadi/diffusion/fr/inputs/v4_constraints_mod_merge.json'
    

    hrs_parsed_inputs_path = "../inputs/parsed_inputs_gpt4_hrs_color_mod.json"
    hrs_constraints_path = "../inputs/hrs_color_constraints.json"
    

    def __post_init__(self):
        self.output_path.mkdir(exist_ok=True, parents=True)

