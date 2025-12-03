import csv, argparse, statistics, math, os

"""
兼容新版 detect_track_camera.py 的精度评估工具：

支持三种输入：
1) 合并CSV（默认）：ts, frame_idx, ir_ok, tv_ok, ir_bbox, ir_offx, ir_offy, tv_bbox, tv_offx, tv_offy
2) IR单独CSV：ts, frame_idx, ok, bbox, offx, offy
3) TV单独CSV：ts, frame_idx, ok, bbox, offx, offy

参数：
- --csv         合并CSV路径（默认 my_yolo/results/dual_sync_stats.csv）
- --csv_ir      IR单独CSV路径（可选）
- --csv_tv      TV单独CSV路径（可选）
- --ir_thresh   IR允许偏差阈值（像素），默认2.0
- --tv_thresh   TV允许偏差阈值（像素），默认4.0
- --metric      偏差度量：l1 或 l2（默认 l1）
- --pctl        评估用百分位（默认95，即p95）
"""

def _norm(x, y, metric='l1'):
    if x is None or y is None:
        return None
    if metric == 'l2':
        return math.sqrt(x * x + y * y)
    # 默认l1
    return abs(x) + abs(y)

def _read_combined(csv_path):
    """读取合并CSV，返回(ir_offsets, tv_offsets)列表，元素为(tuple offx, offy)。"""
    ir_off = []
    tv_off = []
    with open(csv_path, newline='') as f:
        rd = csv.DictReader(f)
        # 期望字段包含 ir_offx, ir_offy, tv_offx, tv_offy（从首行表头自动解析）
        for row in rd:
            try:
                ix = float(row.get('ir_offx', '')) if row.get('ir_offx', '') != '' else None
                iy = float(row.get('ir_offy', '')) if row.get('ir_offy', '') != '' else None
                tx = float(row.get('tv_offx', '')) if row.get('tv_offx', '') != '' else None
                ty = float(row.get('tv_offy', '')) if row.get('tv_offy', '') != '' else None
            except Exception:
                ix = iy = tx = ty = None
            if ix is not None and iy is not None:
                ir_off.append((ix, iy))
            if tx is not None and ty is not None:
                tv_off.append((tx, ty))
    return ir_off, tv_off

def _read_single(csv_path):
    """读取单路CSV（IR或TV），返回 offsets 列表。"""
    offs = []
    with open(csv_path, newline='') as f:
        rd = csv.DictReader(f)
        # 期望字段：offx, offy
        for row in rd:
            try:
                ox = float(row.get('offx', '')) if row.get('offx', '') != '' else None
                oy = float(row.get('offy', '')) if row.get('offy', '') != '' else None
            except Exception:
                ox = oy = None
            if ox is not None and oy is not None:
                offs.append((ox, oy))
    return offs

def _summarize(arr, name, thresh, metric='l1', pctl=95):
    if not arr:
        print(f"[{name}] no valid samples.")
        return
    mags = []
    xs = []
    ys = []
    for (x, y) in arr:
        xs.append(x); ys.append(y)
        n = _norm(x, y, metric)
        if n is not None:
            mags.append(n)
    mean = statistics.mean(mags)
    stdev = statistics.pstdev(mags)
    # 百分位（若样本不足，用最大值近似）
    if len(mags) >= 2:
        try:
            # statistics.quantiles 默认分位数定义，使用n=100时返回1..99百分位分割
            q = statistics.quantiles(mags, n=100)
            idx = max(1, min(99, pctl)) - 1
            pctv = q[idx]
        except Exception:
            pctv = max(mags)
    else:
        pctv = max(mags)
    ok = (pctv <= thresh)
    print(f"[{name}] N={len(mags)} mean={mean:.2f} stdev={stdev:.2f} p{pctl}={pctv:.2f} | thresh={thresh} => {'OK' if ok else 'FAIL'}")
    # 额外输出x/y方向统计，便于诊断系统性偏移
    mx = statistics.mean(xs); my = statistics.mean(ys)
    sx = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    sy = statistics.pstdev(ys) if len(ys) > 1 else 0.0
    print(f"    x: mean={mx:.2f} stdev={sx:.2f} | y: mean={my:.2f} stdev={sy:.2f}")

def analyze(csv_path=None, csv_ir=None, csv_tv=None,
            ir_thresh=2.0, tv_thresh=4.0, metric='l1', pctl=95):
    ir_off = []
    tv_off = []
    # 读取合并CSV（若提供）
    if csv_path and os.path.isfile(csv_path):
        a, b = _read_combined(csv_path)
        ir_off.extend(a); tv_off.extend(b)
    # 读取单路CSV（若提供）
    if csv_ir and os.path.isfile(csv_ir):
        ir_off.extend(_read_single(csv_ir))
    if csv_tv and os.path.isfile(csv_tv):
        tv_off.extend(_read_single(csv_tv))

    # 汇总输出
    _summarize(ir_off, 'IR', ir_thresh, metric=metric, pctl=pctl)
    _summarize(tv_off, 'TV', tv_thresh, metric=metric, pctl=pctl)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str, default='./results/both.csv',
                    help='合并CSV路径（包含IR与TV）')
    ap.add_argument('--csv_ir', type=str, default=None,
                    help='IR单独CSV路径（可选）')
    ap.add_argument('--csv_tv', type=str, default=None,
                    help='TV单独CSV路径（可选）')
    ap.add_argument('--ir_thresh', type=float, default=2.0)
    ap.add_argument('--tv_thresh', type=float, default=4.0)
    ap.add_argument('--metric', type=str, choices=['l1','l2'], default='l1',
                    help='偏差度量方式：l1或l2')
    ap.add_argument('--pctl', type=int, default=95,
                    help='用于判定的百分位（默认p95）')
    args = ap.parse_args()

    analyze(args.csv, args.csv_ir, args.csv_tv,
            ir_thresh=args.ir_thresh, tv_thresh=args.tv_thresh,
            metric=args.metric, pctl=args.pctl)