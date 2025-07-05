timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
python main.py --mp_depth > ablation_report_mp_depth_${timestamp}.txt
python main.py --mp_width > ablation_report_mp_width_${timestamp}.txt
python main.py --fc_depth > ablation_report_fc_depth_${timestamp}.txt
python main.py --fc_width > ablation_report_fc_width_${timestamp}.txt

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
python main.py --mp_depth --train_mp > ablation_report_mp_depth_${timestamp}.txt
python main.py --mp_width --train_mp > ablation_report_mp_width_${timestamp}.txt
python main.py --fc_depth --train_mp > ablation_report_fc_depth_${timestamp}.txt
python main.py --fc_width --train_mp > ablation_report_fc_width_${timestamp}.txt
