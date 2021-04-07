import os
import numpy as np
from argparse import ArgumentParser

import tensorflow as tf
import tensorflow_addons as tfa

from model import VisionTransformer

if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation',type=str, default=os.environ['SM_CHANNEL_VALIDATION']) 
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-dim", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
 
    args, _ = parser.parse_known_args()
    
    x_train = np.load(os.path.join(args.train, 'training.npz'))['image']
    y_train = np.load(os.path.join(args.train, 'training.npz'))['label']
    x_test  = np.load(os.path.join(args.validation, 'testing.npz'))['image']
    y_test  = np.load(os.path.join(args.validation, 'testing.npz'))['label']
     
    model = VisionTransformer(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_layers=args.num_layers,
        num_classes=10,
        d_model=args.d_model,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        channels=3,
        dropout=0.1,
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ),
        optimizer=tfa.optimizers.AdamW(
            learning_rate=args.learning_rate, weight_decay=args.weight_decay
        ),
        metrics=["accuracy"],
    )

    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=args.epochs,
        verbose=2
    )
    
    version = '00000000'
    model.save(os.path.join(args.model_dir, version))