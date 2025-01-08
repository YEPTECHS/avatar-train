import os
import warnings
import logging

from dotenv import load_dotenv
from datetime import datetime
from accelerate.utils import ProjectConfiguration

from models.musetalk_dataset import MusetalkTrainDataset 
from models.trainer import Trainer
from models.config import config
from helpers import setup_wandb

# Load environment variables from.env file
load_dotenv()

# Ignore warnings
warnings.filterwarnings("ignore")

# Create a log folder
os.makedirs('logs', exist_ok=True)
train_start_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Output to both file and console
logging.basicConfig(filename=f'logs/{train_start_time}.log', filemode='w', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

logger = logging.getLogger(__name__)


def main():
    # 1. Settings
    project_config = ProjectConfiguration(
        total_limit=config.train.checkpoints_total_limit,
    )

    # 2. Initialize trainer
    trainer = Trainer(
        project_config=project_config,
    )
    logger.info("successfully initialized trainer")
    
    # 3. Setup wandb
    if config.train.wandb_enabled:
        if os.getenv("WANDB_KEY") is not None:
            setup_wandb()
        else:
            logger.warning("WANDB_KEY is not set, wandb is not enabled")
    
    # 4. Train
    logger.info("start training")
    trainer.train(
        validation_size=config.data.validation_size,
        resume_from_checkpoint=config.model.resume_from_checkpoint
    )


if __name__ == "__main__":
    main()