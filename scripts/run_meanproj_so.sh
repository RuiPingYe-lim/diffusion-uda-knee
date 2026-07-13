export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH=/root/autodl-tmp/knee/code2:${PYTHONPATH:-}
cd /root/autodl-tmp/knee/code2/idea2_diffusion_baseline/bbdm_strict
M=/root/autodl-tmp/meanproj_stage
echo "########## FORWARD M->K ##########"
python train_eval_meanproj.py --tag "M->K" --src_train $M/mrnet_train.csv --src_val $M/mrnet_valid.csv --tgt $M/kneemri_test.csv --out $M/cls_mrnet_mp.pt --epochs 40
echo "########## REVERSE K->M ##########"
python train_eval_meanproj.py --tag "K->M" --src_train $M/kneemri_train.csv --src_val $M/kneemri_valid.csv --tgt $M/mrnet_test.csv --out $M/cls_kneemri_mp.pt --epochs 40
echo "MEANPROJ_SO_DONE"
