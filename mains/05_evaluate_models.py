import os
import torch
from sklearn.metrics import classification_report
from utilities.config_parser import load_config
from utilities.logger import get_logger
from src.data.loader import get_loaders
from src.models.hybrid import HybridResNetLSTM
from torchvision.models.video import r3d_18

cfg = load_config()
logger = get_logger("05_EVAL", log_dir=cfg['paths']['logs_dir'])

def run_evaluation():
    cfg = load_config()
    _, val_loader = get_loaders(cfg)
    
    model_constructors = {
        "exp1": lambda: r3d_18(),
        "exp2": lambda: r3d_18(),
        "exp3": lambda: HybridResNetLSTM(num_classes=cfg['experiment']['num_classes'])
    }

    for key, path in cfg['evaluation']['model_weights'].items():
        if not os.path.exists(path):
            logger(f"⚠️ Weights missing for {key}")
            continue
            
        model = model_constructors[key]().cuda()
        if "exp1" in key or "exp2" in key:
            model.fc = torch.nn.Linear(model.fc.in_features, cfg['experiment']['num_classes']).cuda()
        
        model.load_state_dict(torch.load(path))
        model.eval()
        
        preds, labels = [], []
        with torch.no_grad():
            for i, (d, l) in enumerate(val_loader):
                out = model(d.cuda())
                preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                labels.extend(l.numpy())
                if cfg['experiment']['quick_test'] and i >= 1: break
        
        logger(f"Report for {key}:")
        logger(classification_report(labels, preds, target_names=cfg['experiment']['target_classes']))

if __name__ == "__main__":
    run_evaluation()