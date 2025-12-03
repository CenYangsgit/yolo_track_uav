import subprocess, re, argparse, json

def list_devices():
    out = subprocess.check_output(["v4l2-ctl","--list-devices"], text=True)
    return out

def list_formats(dev):
    out = subprocess.check_output(["v4l2-ctl","-d",dev,"--list-formats-ext"], text=True)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', type=str, default='my_yolo/configs/cameras.yaml')
    args = ap.parse_args()

    devices_info = list_devices()
    rec = {'devices': []}
    print(devices_info)
    for line in devices_info.splitlines():
        m = re.search(r'(/dev/video\d+)', line)
        if m:
            dev = m.group(1)
            try:
                fm = list_formats(dev)
                rec['devices'].append({'dev': dev, 'formats': fm})
                print(f"=== {dev} ==="); print(fm)
            except Exception as e:
                print(f"[warn] list formats failed for {dev}: {e}")

    # 简单保存文本（你可以手工挑选 IR/TV 的 dev 与格式）
    import os
    os.makedirs('my_yolo/configs', exist_ok=True)
    with open(args.save, 'w') as f:
        f.write("# inspect and edit IR/TV devices\n")
        for d in rec['devices']:
            f.write(f"--- {d['dev']} ---\n{d['formats']}\n")
    print(f"[save] {args.save}")

if __name__ == "__main__":
    main()