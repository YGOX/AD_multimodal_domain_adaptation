#!/etc/bash


nohup python main_multi.py --gpu 2 --modality mri &> nohup201.out &
nohup python main_multi.py --gpu 3 --modality pet &> nohup301.out &
sleep 9000
nohup python main_multi.py --gpu 2 --modality multi --advers True &> nohup401.out &
nohup python main_multi.py --gpu 3 --modality multi --advers False &> nohup402.out &

