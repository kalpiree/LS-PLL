#python main.py --smoothing=0.001 > metrics_label_smoothing_1e-3.txt
#python main.py --smoothing=0.01 > metrics_label_smoothing_1e-2.txt
#python main.py --smoothing=0.1 > metrics_label_smoothing_1e-1.txt
#python main.py --smoothing=0.3 > metrics_label_smoothing_3e-1.txt
#python main.py --smoothing=0.5 > metrics_label_smoothing_5e-1.txt
python main.py --ce > metrics_cross_entropy.txt
