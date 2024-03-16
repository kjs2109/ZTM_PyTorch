import torch 
import pytorch_lightning as pl 
from model import NN 
from dataset import MnistDataModule 
import config
from callbacks import MyPrintingCallback
from pytorch_lightning.callbacks import EarlyStopping 
from pytorch_lightning.loggers import TensorBoardLogger  # tensorboard --logdir=tb_logs --bind_all
from pytorch_lightning.profilers import PyTorchProfiler  # pip install torch-tb-profiler 
# from pytorch_lightning.strategies import DeepSpeedStrategy 


if __name__ == "__main__": 
    
    logger = TensorBoardLogger(save_dir="tb_logs", name="mnist_model_v1") 
    profiler = PyTorchProfiler(on_trace_ready=torch.profiler.tensorboard_trace_handler("tb_logs/profiler0")) 
    # strategy = DeepSpeedStrategy()
    
    model = NN(input_size=config.INPUT_SIZE, learning_rate=config.LEARNING_RATE, num_classes=config.NUM_CLASSES).to(config.DEVICE) 
    data_module = MnistDataModule(data_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE, num_workers=0) 
    trainer = pl.Trainer(# profiler=profiler,  # "simple",
                         # strategy= strategy  # "ddp",   # multi-gpu 
                         logger=logger,
                         min_epochs=1, 
                         max_epochs=3, 
                         precision='bf16-mixed', 
                         callbacks=[MyPrintingCallback(), EarlyStopping(monitor="val_loss")])  

    trainer.fit(model, data_module)
    trainer.validate(model, data_module) 
    trainer.test(model, data_module)