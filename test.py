
import os, time
import numpy as np
import cv2
from rknnlite.api import RKNNLite

MODEL = "/home/firefly/yolo_track_uav/weights/IR.rknn"
IMAGE = "/home/firefly/yolo_track_uav/datasets/images/IR/frame_0007.jpg"
SIZE  = 640                   
TO_RGB = True                 
MEAN = [0, 0, 0]              
STD  = [1, 1, 1]             
SAVE_OUTPUTS = False          
# =====================

def letterbox(im, new_shape=(640, 640), color=(114,114,114)):
    h0, w0 = im.shape[:2]
    r = min(new_shape[0]/h0, new_shape[1]/w0)
    new_unpad = (int(round(w0*r)), int(round(h0*r)))
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    dw, dh = new_shape[1]-new_unpad[0], new_shape[0]-new_unpad[1]
    top, bottom = dh//2, dh - dh//2
    left, right = dw//2, dw - dw//2
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im

def prep_image(path, size, to_rgb, mean, std):
    img0 = cv2.imread(path)
    assert img0 is not None, f"Failed to read image: {path}"
    img = letterbox(img0, (size, size))
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = (img - mean) / std
    return img

def try_infer(rk, img):

    # NCHW float32
    try:
        x = np.transpose(img, (2,0,1))  # HWC->CHW
        x = np.expand_dims(x, 0).copy() # NCHW
        x = np.ascontiguousarray(x, dtype=np.float32)
        t0 = time.time()
        outs = rk.inference(inputs=[x])
        t1 = time.time()
        return outs, "NCHW", "float32", (t1 - t0)*1000
    except Exception as e_nchw_f32:
        last_err = e_nchw_f32

    # NCHW uint8
    try:
        x = np.transpose(img, (2,0,1))
        x = np.expand_dims(x, 0).copy()
        x = np.clip(x, 0, 255).astype(np.uint8, copy=False)
        x = np.ascontiguousarray(x)
        t0 = time.time()
        outs = rk.inference(inputs=[x])
        t1 = time.time()
        return outs, "NCHW", "uint8", (t1 - t0)*1000
    except Exception as e_nchw_u8:
        last_err = e_nchw_u8

    # NHWC float32
    try:
        x = np.expand_dims(img, 0).copy()  # NHWC
        x = np.ascontiguousarray(x, dtype=np.float32)
        t0 = time.time()
        outs = rk.inference(inputs=[x])
        t1 = time.time()
        return outs, "NHWC", "float32", (t1 - t0)*1000
    except Exception as e_nhwc_f32:
        last_err = e_nhwc_f32

    # NHWC uint8
    try:
        x = np.expand_dims(img, 0).copy()
        x = np.clip(x, 0, 255).astype(np.uint8, copy=False)
        x = np.ascontiguousarray(x)
        t0 = time.time()
        outs = rk.inference(inputs=[x])
        t1 = time.time()
        return outs, "NHWC", "uint8", (t1 - t0)*1000
    except Exception as e_nhwc_u8:
        last_err = e_nhwc_u8

    raise RuntimeError(f"Inference failed in all layouts/dtypes. Last error: {last_err}")

def main():
  
    img = prep_image(IMAGE, SIZE, TO_RGB, MEAN, STD)
    rk = RKNNLite()
    assert rk.load_rknn(MODEL) == 0, "load_rknn failed"
    assert rk.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2) == 0, "init_runtime failed"
   
    try:
        _ = rk.inference(inputs=[np.expand_dims(np.transpose(img,(2,0,1)),0).astype(np.float32)])
    except Exception:
        pass

    outs, layout, dtype_name, elapsed_ms = try_infer(rk, img)
    print(f"Inference OK. Layout={layout}, dtype={dtype_name}, time={elapsed_ms:.2f} ms")

    for i, o in enumerate(outs):
        a = np.array(o)
        a_np = np.asarray(a)
        print(f"out[{i}] shape={a_np.shape} dtype={a_np.dtype} "
              f"min={a_np.min():.4f} max={a_np.max():.4f}")
        if SAVE_OUTPUTS:
            out_path = f"/tmp/out_{i}.npy"
            np.save(out_path, a_np)
            print(f"  saved -> {out_path}")

    rk.release()

if __name__ == "__main__":

    assert os.path.exists(MODEL), f"Model not found: {MODEL}"
    assert os.path.exists(IMAGE), f"Image not found: {IMAGE}"
    main()
