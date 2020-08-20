python pruning.py --architecture vae --lr 0.0001 --epochs 50 >> results/out_s2_m4.txt
python weight_share.py saves/model_after_retraining_vae.ptmodel --architecture vae --output vae_after_wt_share.ptmodel >> results/out_s2_m4.txt
python huffman_encode.py vae_after_wt_share.ptmodel >> results/out_s2_m4.txt