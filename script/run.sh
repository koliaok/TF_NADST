python train.py -save_path=save/nadst -path=temp -d=256 -h_attn=16 -bsz=32 -wu=20000 -dr=0.2 -dv='2.1' -fert_dec_N=3 -state_dec_N=3 -gate=0

python test.py  -save_path=save/nadst -path=temp -d=256 -h_attn=16 -bsz=32 -wu=20000 -dr=0.2 -dv='2.1' -fert_dec_N=3 -state_dec_N=3  -ep=1 -gate=0
