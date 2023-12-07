import argparse
import torch
import datetime
import json
import yaml
import os

from dataset_physio import get_dataloader
from utils import train, evaluate
from main_model import SampleType

import log_util

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)

parser.add_argument("--sample_type", type=str, default="csdi_ddpm_nose", choices=[x.name for x in SampleType], help="csdi_ddpm_nose|ddim_generalized")
parser.add_argument("--ddim_eta", type=float, default=0.0)
parser.add_argument("--num_timesteps", type=int, default=None)
parser.add_argument("--timesteps", type=int, default=None)
                
parser.add_argument("--noise_fn", type=str, default="gaussian")

parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--modelpath", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)

args = parser.parse_args()
print(args)


path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

if args.num_timesteps is not None:
    config["diffusion"]["num_steps"] = args.num_timesteps
if args.timesteps is not None:
    config["diffusion"]["timesteps"] = args.timesteps
    
sampletype = SampleType[args.sample_type]

if args.noise_fn == "simplex":
    from main_model_with_simplex_noise import CSDI_Physio, NoiseGenerator
    noise_gen = NoiseGenerator(NoiseGenerator.NoiseType.simplex)
    model = CSDI_Physio(config, args.device, noise_gen=noise_gen).to(args.device)
    folder_prefix = "simplex"
else:
    from main_model import CSDI_Physio
    model = CSDI_Physio(sampletype, args.ddim_eta, config, args.device).to(args.device)
    folder_prefix = "gaussian"

folder_prefix += "_noise"

if sampletype == SampleType.ddim_generalized:
    folder_prefix += f"_im_ddim_eta{args.ddim_eta}"
if args.modelpath:
    folder_prefix += "_imutation"
    
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = f"./save/{folder_prefix}/physio_fold{args.nfold}_missing{int(args.testmissingratio*100)}p_{current_time}/"

print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

logger = log_util.setup_logger(f"exe_physio", folder=foldername, filename=f"exe_physio")
logger.info(f"exe_physio.start")
logger.info("config:\n"+json.dumps(config, indent=4))

train_loader, valid_loader, test_loader = get_dataloader(
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
    logger=logger
)

if len(args.modelpath):
    model.load_state_dict(torch.load(args.modelpath))
elif len(args.modelfolder):
    model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))
else:
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
        logger=logger
    )

evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername, logger=logger)

logger.info(f"exe_physio.end")
