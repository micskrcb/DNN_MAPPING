import json
import os

results = [
    # (Workload, Tasks, Chips, BS_file, RS_file, SA_file, RS_label, SA_label)
    ('AlexNet-CONV',  183, 1, 'alexnet_conv_bs.json',  'alexnet_conv_rs_10k.json',  'alexnet_conv_sa_100k.json', 'RS 10k', 'SA 100k'),
    ('AlexNet-FC',    932, 4, 'alexnet_fc_bs.json',    'alexnet_fc_rs_10k.json',    'alexnet_fc_sa_10k.json',     'RS 10k', 'SA 10k'),
    ('VGG16-CONV',   1024, 4, 'vgg16_conv_bs.json',    'vgg16_conv_rs_5k.json',     'vgg16_conv_sa_10k.json',     'RS 5k',  'SA 10k'),
    ('VGG16-FC',     1924, 8, 'vgg16_fc_bs.json',      'vgg16_fc_rs_5k.json',       'vgg16_fc_sa_10k.json',       'RS 5k',  'SA 10k'),
    ('ResNet50-CONV', 512, 2, 'resnet50_conv_bs.json', 'resnet50_conv_rs_5k.json',  'resnet50_conv_sa_100k.json', 'RS 5k',  'SA 100k'),
    ('ResNet50-FC',    37, 1, 'resnet50_fc_bs.json',   'resnet50_fc_rs_10k.json',   'resnet50_fc_sa_100k.json',   'RS 10k', 'SA 100k'),
]

master = []
print("="*105)
print(f"{'Workload':<15} {'Tasks':>5} {'Chips':>5} {'Sequential (BS)':>18} {'Random Search (RS)':>22} {'Simulated Annealing (SA)':>26} {'Gain vs BS':>12}")
print("="*105)

for w, tasks, chips, bs_f, rs_f, sa_f, rs_lbl, sa_lbl in results:
    p_bs = os.path.join('paper/results', bs_f)
    p_rs = os.path.join('paper/results', rs_f)
    p_sa = os.path.join('paper/results', sa_f)
    
    bs_val = round(json.load(open(p_bs))['best_cost']*1e6, 5) if os.path.exists(p_bs) else None
    rs_val = round(json.load(open(p_rs))['best_cost']*1e6, 5) if os.path.exists(p_rs) else None
    sa_val = round(json.load(open(p_sa))['best_cost']*1e6, 5) if os.path.exists(p_sa) else None
    
    gain = f"+{(1 - sa_val/bs_val)*100:.2f}%" if (bs_val and sa_val and bs_val > 0) else "0.00%"
    
    rs_str = f"{rs_val:.3f} us ({rs_lbl})" if rs_val else "N/A"
    sa_str = f"{sa_val:.3f} us ({sa_lbl})" if sa_val else "N/A"
    bs_str = f"{bs_val:.3f} us" if bs_val else "N/A"
    
    print(f"{w:<15} {tasks:>5} {chips:>5} {bs_str:>18} {rs_str:>22} {sa_str:>26} {gain:>12}")
    master.append({
        'workload': w, 'tasks': tasks, 'chips': chips,
        'bs_us': bs_val, 'rs_us': rs_val, 'sa_us': sa_val,
        'rs_label': rs_lbl, 'sa_label': sa_lbl,
        'gain_vs_bs_pct': round((1 - sa_val/bs_val)*100, 2) if bs_val and sa_val else 0.0
    })

print("="*105)
with open('paper/results/complete_all_6_todaes_master.json', 'w') as f:
    json.dump(master, f, indent=2)
