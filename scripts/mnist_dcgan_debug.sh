python3 train.py \
  --name mnist_dcgan_debug \
  --verbose \
  --model_name vanilla_gan \
  --dataloader_name simple \
  --logger_name mlflow \
  --out_size 28 \
  --output_nch 1 \
  --n_epochs 10 \
  --n_epochs_decay 10 \
  --discriminator_module_name discriminator_cnn \
  --generator_module_name generator_cnn \
  --discriminator_optimizer_name discriminator_adam \
  --generator_optimizer_name generator_adam \
  --discriminator_scheduler_name linear \
  --generator_scheduler_name linear \
  --init_weight_name xavier \
  --gan_loss_name wgangp_loss \
  --discriminator_norm_module_name batch_norm_2d \
  --discriminator_n_blocks 2 \
  --generator_n_blocks 2 \
  --latent_dim 64 \
  --generator_norm_module_name batch_norm_2d \
  --batch_size 16 \
  --dataset_name mnist \
  --max_dataset_size 1000 \
  --transform_name affine
