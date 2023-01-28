import torch
import argparse
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import *

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',
                       type=int,
                       default=0)
    parser.add_argument('--data_path',
                        type=str,
                        default='/workspace/intent/newADB/data/',
                        help='where to prepare data')
    parser.add_argument("--dataset", 
                        default='stackoverflow', 
                        type=str, 
                        help="The name of the dataset to train selected")
    parser.add_argument("--known_cls_ratio",
                        default=0.5,
                        type=float,
                        help="The number of known classes")
    parser.add_argument("--labeled_ratio",
                        default=1.0,
                        type=float,
                        help="The ratio of labeled samples in the training set")
    parser.add_argument('--max_epoch',
                       type=int,
                        default=100,
                       help='maximum number of epochs to train')
    parser.add_argument('--num_gpus',
                       type=int,
                       default=1,
                       help='number of available gpus')
    parser.add_argument('--ckpt_path',
                       type=str,
                       default='/workspace/intent/newADB',
                       help='checkpoint file path')
    parser.add_argument('--model_save_path',
                       type=str,
                       default='checkpoints',
                       help='where to save checkpoint files')
    parser.add_argument('--max_seq_len',
                       type=int,
                       default=45,
                       help='maximum length of input sequence data')
    parser.add_argument('--batch_size',
                       type=int,
                       default=128,
                       help='batch size')
    parser.add_argument('--device',
                       type=int,
                       default=0,
                       help='device')
    # parser.add_argument('--mode',
    #                    type=str,
    #                    default='train',
    #                    help='mode')
    parser.add_argument('--num_workers',
                        type=int,
                        default=8,
                        help='num of worker for dataloader')
    parser.add_argument('--output_dir',
                       type=str,
                       default='/workspace/intent/newADB/results/',
                       help='output dir')
    parser.add_argument("--results_file_name", type=str, default = 'results.csv', help="The file name of all the results.")

    args = parser.parse_args()

    return args



def load_model_from_experiment(args):
        # hparams_file = experiment_folder + "/hparams.yaml"
        
        # hparams_file = "/workspace/intent/newADB/lightning_logs/version_0/hparams.yaml"

        # hparams = yaml.load(open(hparams_file).read(), Loader=yaml.FullLoader)

        checkpoints = [
            file
            for file in os.listdir("/workspace/intent/newADB/checkpoints/")
            if file.endswith(".ckpt")
        ]
        # checkpoint_path = ckpt_path + checkpoints[-1]
        # model = BERTfeature.load_from_checkpoint(
        #     checkpoint_path, hparams=Namespace(**hparams)
        # )
        print("name", checkpoints[0])
        checkpoint_path = args.ckpt_path + "/checkpoints/" + checkpoints[0]
        model = BERTfeature.load_from_checkpoint(checkpoint_path, args =args)
        print("checkpoint_path", checkpoint_path)
        model.eval()
        model.freeze()
        return model

def main():
    args = parse_arguments()
    # seed_everything(args.seed, workers=True)

    dm = ADBDataModule(
        data_path=args.data_path,
        dataset=args.dataset,
        batch_size=args.batch_size,
        known_cls_ratio = args.known_cls_ratio,
        labeled_ratio = args.labeled_ratio,
        seed = args.seed,
        worker= args.num_workers
    )

    dm.setup('fit')
    
    model = BERTfeature(args)

    filename =  f'BestModel_{args.dataset}_{args.known_cls_ratio}'

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath=args.model_save_path,
        # filename='{epoch:02d}-{val_acc:.3f}',
        filename = filename,
        verbose=True,
        save_last=False,
        mode='max',
        save_top_k=1,
    )
    # early_stopping = EarlyStopping(
    #     monitor='val_acc', 
    #     mode='max',
    # )

    # tb_logger = pl_loggers.TensorBoardLogger(os.path.join(dir_path, 'tb_logs'))
    # lr_logger = pl.callbacks.LearningRateMonitor()

    trainer = Trainer(
        max_epochs=args.max_epoch,
        accelerator="gpu",
        devices=[1],
        auto_select_gpus=True,
        callbacks=[checkpoint_callback],
    )
    # trainer.fit(model, dm)

    model = load_model_from_experiment(args)
    model.eval()
    trainer.predict(model, dm)

if __name__ == '__main__':
    main()