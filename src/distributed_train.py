import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup_distributed(rank, world_size):
    """Инициализация распределенного обучения"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Очистка после распределенного обучения"""
    dist.destroy_process_group()

def train_ddp(rank, world_size, config):
    """Обучение с DistributedDataParallel"""
    
    # Инициализация
    setup_distributed(rank, world_size)
    
    # Создание модели
    model = create_resnet50(config).to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Распределенный sampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE // world_size,
        sampler=train_sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    # Обучение
    for epoch in range(config.EPOCHS):
        train_sampler.set_epoch(epoch)  # Важно для shuffle
        # ... обучение ...
    
    cleanup_distributed()

# Запуск
def main_ddp():
    world_size = torch.cuda.device_count()
    mp.spawn(train_ddp, args=(world_size, config), nprocs=world_size)