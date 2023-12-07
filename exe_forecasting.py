import argparse
import torch
import datetime
import json
import yaml
import os

from main_model import CSDI_Forecasting
from dataset_forecasting import get_dataloader
from utils import train, evaluate

import log_util

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base_forecasting.yaml")
parser.add_argument("--datatype", type=str, default="electricity")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)

parser.add_argument("--noise_fn", type=str, default="gaussian")

parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--modelpath", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

if args.datatype == 'electricity':
    target_dim = 370

config["model"]["is_unconditional"] = args.unconditional


if args.noise_fn == "simplex":
    from main_model_with_simplex_noise import CSDI_Forecasting, NoiseGenerator
    noise_gen = NoiseGenerator(NoiseGenerator.NoiseType.simplex)
    model = CSDI_Forecasting(config, args.device, target_dim, noise_gen=noise_gen).to(args.device)
    folder_prefix = "simplex"
else:
    from main_model import CSDI_Forecasting
    model = CSDI_Forecasting(config, args.device, target_dim).to(args.device)
    folder_prefix = "gaussian"


current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# foldername = "./save/forecasting_" + args.datatype + '_' + current_time + "/"
foldername = f"./save/{folder_prefix}_noise/forecasting_{args.datatype}_{current_time}/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

logger = log_util.setup_logger(f"exe_forecasting", folder=foldername, filename=f"exe_forecasting")
logger.info(f"exe_forecasting.start")
logger.info("config:\n"+json.dumps(config, indent=4))

train_loader, valid_loader, test_loader, scaler, mean_scaler = get_dataloader(
    datatype=args.datatype,
    device= args.device,
    batch_size=config["train"]["batch_size"],
    logger=logger
)

if len(args.modelpath):
    logger.info(f"modelpath: {args.modelpath}. load start")
    try:
        model.load_state_dict(torch.load(args.modelpath))
    except Exception as e:
        logger.info(f"modelpath: {args.modelpath}. load. exception:{str(e)}")
        exit(-1)
        
    logger.info(f"modelpath: {args.modelpath}. load end")
    
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

evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername,
    logger=logger
)

logger.info(f"exe_forecasting.end")
