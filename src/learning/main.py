import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from skeleton import Settings
from learning.frameworks.snn_torch import SNN_TORCH
from learning.data_pipeline import main as load_data  # Import the main() function

def main():
    cfg = Settings()
    
    # Get dataloaders from data_pipeline
    train_loader, test_loader = load_data()
    
    # Create model and pass loaders
    model = SNN_TORCH(cfg)
    trainer = model.get_trainer(train_loader)
    inference = model.get_inference(test_loader)
    
    print("\n✓ Model ready.")
    cfg.display()
    return model, trainer, inference

if __name__ == "__main__":
    cfg = Settings()
    model, trainer, inference = main()
    
    # Train the model
    results = trainer.train(checkpoint_dir="./checkpoints")
    
    print("\n✓ Training complete!")
    print(f"  Final loss: {results['loss_history'][-1]:.4f}")
    print(f"  Final accuracy: {results['accuracy_history'][-1]:.4f}")



"""
The main() function orchestrates the full pipeline. 
Note the try/finally pattern guaranteeing that the process group is torn down even if an exception occurs —
 without this, a crash on one rank can leave other ranks hanging indefinitely.
"""

def main():
    config = TrainingConfig.from_args()
    ctx = setup_distributed(config)
    logger = setup_logger(ctx.rank)


    torch.manual_seed(config.seed + ctx.rank)


    model = create_model(config, ctx.device)
    model = wrap_ddp(model, ctx.local_rank)


    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr,
                                 momentum=config.momentum,
                                 weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)
    scaler = torch.amp.GradScaler(enabled=config.use_amp)


    start_epoch = 1
    if config.resume_from:
        start_epoch = load_checkpoint(config.resume_from, model.module,
                                       optimizer, scaler, ctx.device) + 1


    dataset = SyntheticImageDataset(size=50000, image_size=config.image_size,
                                     num_classes=config.num_classes)
    loader, sampler = create_distributed_dataloader(dataset, config, ctx)
    criterion = nn.CrossEntropyLoss()


    try:
        for epoch in range(start_epoch, config.epochs + 1):
            sampler.set_epoch(epoch)
            tracker = train_one_epoch(model, loader, criterion, optimizer,
                                       scaler, ctx, config, epoch, logger)
            scheduler.step()


            avg_loss = all_reduce_scalar(tracker.average("loss"),
                                          ctx.world_size, ctx.device)


            if is_main_process(ctx.rank):
                log_epoch_summary(logger, epoch, {"loss": avg_loss})
                if epoch % config.save_every == 0:
                    save_checkpoint(f"checkpoints/epoch_{epoch}.pt",
                                     epoch, model, optimizer, scaler, ctx.rank)
    finally:
        cleanup_distributed()