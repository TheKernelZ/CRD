#!/usr/bin/env bash
python distill.py --dataroot database/edges2shoes-r \
  --gpu_ids 2 --print_freq 100 \
  --distiller pix2pixbest \
  --log_dir logs/unet_pix2pix/edges2shoes-r/distill \
  --batch_size 4 \
  --real_stat_path real_stat/edges2shoes-r_B.npz \
  --teacher_ngf 64 --student_ngf 16  --norm batch \
  --teacher_netG unet_256 --netD n_layers \
  --nepochs 100 --nepochs_decay 100 --n_dis 1 \
  --AGD_weights 1e1,1e4,1e1,1e-5