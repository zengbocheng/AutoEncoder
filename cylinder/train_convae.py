"""
@author: bochengz
@date: 2024/02/26
@email: bochengzeng@bochengz.top
"""
from common.config import Config
from kogger import Logger
from auto_encoder.data_handler import GridCylinderDataHandler
import sys
from auto_encoder.trainers import ConvAETrainer
from auto_encoder.models.cylinder import ConvAE
from torch.optim.lr_scheduler import ExponentialLR
import torch


def main():
    args = Config.get_parser().parse_args()
    config = Config.parse_yaml(yaml_filename=args.filename)
    config.device = torch.device('cuda')

    Logger.basic_config(filename=config.log_file)
    logger = Logger.get_logger(__name__)
    logger.info(config)

    def except_hook(exc_type, exc_value, exc_traceback):
        logger.error("Exception",
                     exc_info=(exc_type, exc_value, exc_traceback))
    sys.excepthook = except_hook

    data_handler = GridCylinderDataHandler()
    logger.info('Load training data...')
    tr_loader = data_handler.create_tr_loader(
        file_path=config.train_file,
        n_data=config.tr_n_data,
        time_window=config.tr_time_window,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    logger.info('Load validation data...')
    val_loader = data_handler.create_val_loader(
        file_path=config.val_file,
        n_data=config.val_n_data,
        time_window=config.val_time_window,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    # logger.info('Load test data...')
    # te_loader = data_handler.create_te_loader(
    #     file_path=config.te_file,
    #     n_data=config.te_n_data,
    #     time_window=config.te_time_window
    # )

    model = ConvAE(
        config.n_embed,
        config.embed_drop,
        config.layer_norm_eps
    ).to(config.device)
    model.apply_mu_std(data_handler.mu, data_handler.std, config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=1.0e-8)
    scheduler = ExponentialLR(optimizer, gamma=0.995)
    trainer = ConvAETrainer(
        model,
        optimizer,
        scheduler,
        config
    )
    trainer.train(tr_loader, val_loader)


if __name__ == '__main__':
    main()