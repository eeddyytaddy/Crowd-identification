#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YouTube Live HLS → FFmpeg pipe → OpenCV → YOLOv10/ByteTrack
極簡 HUD：左上角只顯示「現在人數」。支援多串流切換、1080/720 一鍵切換、可選計數模式。
Keys: n=下一條, p=上一條, r=重新連線, ESC=離開

【重點】
- --count-mode active|boxes|tid
  active：活躍 CID（t_last + ACTIVE_TTL）→ 最穩（預設）
  boxes ：一個框＝一個人（你要的）
  tid   ：本幀唯一追蹤ID數
- EMA 平滑 + 防抖 + 掉線保護都保留，可套用在任一模式（更穩）
- --res 720|1080 或 --width/--height 調畫質；--profile far 會優先抓 1080(96) 退 720(95)
"""

import os, sys, time, shlex, subprocess, argparse
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# ========= 預設（可被 CLI 覆寫） ========= #
DEFAULT_FPS = 30
MODEL_PATH      = 'runs/detect/train2/weights/best.pt'  # 換成你的權重
IMG_SIZE        = 1280
CONFIDENCE      = 0.3
IOU_THRESHOLD   = 0.5
CLASS_ID        = 0                 # 0 = person
TRACKER_CFG     = 'bytetrack.yaml'  # 預設檔（profiles 會生成專屬檔）

# ========= 線段／區域（邏輯仍運作，但不畫在畫面上） ========= #
LINES = [
    {"name": "Gate A", "p1": (100, 100), "p2": (740, 200), "directional": False, "cooldown": 2.5},
    {"name": "Gate B", "p1": (300, 520), "p2": (1000, 520), "directional": True,  "cooldown": 2.0},
]
REGIONS = [
    {"name": "Zone Left",  "polygon": [(40, 60), (600, 60), (600, 650), (40, 650)], "cooldown": 1.0},
    {"name": "Zone Right", "polygon": [(680, 80), (1230, 80), (1230, 680), (680, 680)], "cooldown": 1.0},
]

# ========= Profiles（偵測/追蹤一鍵切換） ========= #
PROFILES = {
    "default": {
        "imgsz": IMG_SIZE,
        "conf": CONFIDENCE,
        "iou": IOU_THRESHOLD,
        "max_det": 300,
        "tracker_yaml": None,
        "ytfmt": None
    },
    "far": {  # 遠距/小人像
        "imgsz": 1920,
        "conf": 0.25,
        "iou": 0.55,
        "max_det": 1000,
        "tracker_yaml": """\
tracker_type: bytetrack
track_high_thresh: 0.35
track_low_thresh: 0.05
new_track_thresh: 0.28
track_buffer: 45
match_thresh: 0.75
fuse_score: true
""",
        "ytfmt": "96/95"   # 先 1080, 退 720
    },
    "near": {  # 近距/擁擠
        "imgsz": 1920,
        "conf": 0.35,
        "iou": 0.50,
        "max_det": 500,
        "tracker_yaml": """\
tracker_type: bytetrack
track_high_thresh: 0.60
track_low_thresh: 0.10
new_track_thresh: 0.55
track_buffer: 45
match_thresh: 0.85
fuse_score: true
""",
        "ytfmt": None
    }
}

def _ensure_tracker_yaml(profile_name: str, yaml_text: str) -> str:
    if not yaml_text:
        return os.path.abspath(TRACKER_CFG)
    d = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(d, f"bytetrack_{profile_name}.yaml")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(yaml_text.strip() + "\n")
    return os.path.abspath(path)

# ========= Re-ID Profiles（CanonicalID） ========= #
REID_PROFILES = {
    "strong":   dict(iou_w=0.65, sim_w=0.35, iou_thr=0.30, sim_thr=0.72, ttl_sec=3.2),
    "fast":     dict(iou_w=0.50, sim_w=0.50, iou_thr=0.28, sim_thr=0.62, ttl_sec=2.0),
    "balanced": dict(iou_w=0.55, sim_w=0.45, iou_thr=0.30, sim_thr=0.68, ttl_sec=2.8),
}

def _select_reid_profile(args) -> dict:
    if getattr(args, "reid_strong", False) and getattr(args, "reid_fast", False):
        print("[WARN] 同時指定 --reid-strong 與 --reid-fast，優先採用 --reid-strong")
        return REID_PROFILES["strong"]
    if getattr(args, "reid_strong", False):
        return REID_PROFILES["strong"]
    if getattr(args, "reid_fast", False):
        return REID_PROFILES["fast"]
    return REID_PROFILES["balanced"]

# ========= CLI ========= #
def parse_args():
    ap = argparse.ArgumentParser(description="YouTube HLS 人流計數（極簡 HUD）")
    ap.add_argument("--urls", nargs="+", help="一個或多個來源：YouTube 連結/ID 或 m3u8 URL（空白分隔）")
    ap.add_argument("--names", nargs="*", help="對應每個來源的顯示名稱（可少於 urls）")
    ap.add_argument("--ytfmt", default=os.environ.get("YTDLP_FORMAT", "95"),
                    help="yt-dlp 欲抓取的格式，例如 96/95")
    ap.add_argument("--font", default=os.environ.get("FONT_PATH", ""),
                    help="自訂字型檔路徑（.ttf/.ttc）")
    ap.add_argument("--profile", choices=["default", "far", "near"], default="default",
                    help="偵測/追蹤參數預設集")
    ap.add_argument("--reid-strong", action="store_true", help="較保守的 ReID")
    ap.add_argument("--reid-fast", action="store_true", help="較靈敏的 ReID")

    # 畫質 / 幀率
    ap.add_argument("--res", choices=["720", "1080"], default=os.environ.get("RES", "720"),
                    help="輸入與顯示解析度（720/1080）")
    ap.add_argument("--width", type=int, default=int(os.environ.get("WIDTH", "0")),
                    help="自訂寬（需同時指定 --height）")
    ap.add_argument("--height", type=int, default=int(os.environ.get("HEIGHT", "0")),
                    help="自訂高（需同時指定 --width）")
    ap.add_argument("--fps", type=int, default=int(os.environ.get("FPS", DEFAULT_FPS)),
                    help=f"輸入幀率（預設 {DEFAULT_FPS}）")

    # 計數模式
    ap.add_argument("--count-mode",
        choices=["active", "boxes", "tid"],
        default=os.environ.get("COUNT_MODE", "active"),
        help="active=活躍CID(穩定) | boxes=本幀框數(一框一人) | tid=本幀唯一追蹤ID數")

    # 穩定顯示
    ap.add_argument("--active-ttl", type=float, default=float(os.environ.get("ACTIVE_TTL", 1.2)),
                    help="多久內更新過的 CID 視為在場（秒）")
    ap.add_argument("--ema-alpha", type=float, default=float(os.environ.get("EMA_ALPHA", 0.35)),
                    help="EMA 平滑係數（0~1；越小越穩）")
    ap.add_argument("--hold-frames", type=int, default=int(os.environ.get("HOLD_FRAMES", 6)),
                    help="防抖：變更後維持不動的幀數")
    ap.add_argument("--step-thr", type=int, default=int(os.environ.get("STEP_THR", 1)),
                    help="防抖：顯示值最小變化量門檻")
    ap.add_argument("--no-det-hold", type=float, default=float(os.environ.get("NO_DET_HOLD_SEC", 1.8)),
                    help="完全無偵測時，先維持顯示不變的秒數")
    ap.add_argument("--decay-interval", type=float, default=float(os.environ.get("DECAY_INTERVAL", 0.7)),
                    help="維持期過後，每隔多少秒把顯示值遞減 1")
    return ap.parse_args()

# ========= ffmpeg / yt-dlp ========= #
def start_ffmpeg_pipe(url: str, width: int, height: int, fps: int):
    cmd = (
        f"ffmpeg -loglevel error -i {shlex.quote(url)} "
        f"-an -f rawvideo -pix_fmt bgr24 "
        f"-vf scale={width}:{height},fps={fps} -"
    )
    return subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                            bufsize=width * height * 3 * 10)

def is_probably_youtube(s: str) -> bool:
    if not s: return False
    s = s.strip()
    return ("youtube.com" in s) or ("youtu.be" in s) or (len(s) in (11, 12) and s.isalnum())

def resolve_stream(u: str, ytfmt: str) -> str:
    u = u.strip()
    if is_probably_youtube(u):
        try:
            out = subprocess.check_output(["yt-dlp", "-f", ytfmt, "-g", u],
                                          text=True, stderr=subprocess.STDOUT).strip().splitlines()
            if not out:
                raise RuntimeError("yt-dlp 無輸出")
            resolved = out[-1].strip()
            print(f"[yt-dlp] 解析成功 -> {resolved[:120]}{'...' if len(resolved)>120 else ''}")
            return resolved
        except subprocess.CalledProcessError as e:
            print("[ERROR] yt-dlp 解析失敗：")
            print(e.output)
            raise
    return u

# ========= 幾何 ========= #
def side_of_line(p, a, b):
    return np.sign((b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]))
def crossed_line(p0, p1, a, b):
    s0, s1 = side_of_line(p0, a, b), side_of_line(p1, a, b)
    return s0 != 0 and s1 != 0 and s0 != s1
def crossing_direction(p0, p1, a, b):
    s0, s1 = side_of_line(p0, a, b), side_of_line(p1, a, b)
    if s0 != 0 and s1 != 0 and s0 != s1:
        return 1 if s1 > 0 else -1
    return 0
def point_in_polygon(p, poly):
    x, y = p; inside = False; n = len(poly)
    if n < 3: return False
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i+1) % n]
        if min(x1,x2) <= x <= max(x1,x2) and min(y1,y2) <= y <= max(y1,y2):
            if abs((x2-x1)*(y-y1) - (y2-y1)*(x-x1)) < 1e-6: return True
        inter = ((y1 > y) != (y2 > y)) and (x < (x2-x1)*(y-y1)/(y2-y1+1e-12) + x1)
        if inter: inside = not inside
    return inside

# ========= Canonical ID ========= #
def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0: return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter + 1e-6)

def crop_hist_feat(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)
    if x2 <= x1 or y2 <= y1: return np.zeros((512,), dtype=np.float32)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return np.zeros((512,), dtype=np.float32)
    if roi.dtype != np.uint8: roi = np.clip(roi, 0, 255).astype(np.uint8)
    hist = cv2.calcHist([roi], [0,1,2], None, [8,8,8], [0,256, 0,256, 0,256]).flatten()
    s = float(hist.sum())
    if s <= 1e-6: return np.zeros((512,), dtype=np.float32)
    return (hist / s).astype(np.float32)

def cosine_sim(a, b):
    denom = (np.linalg.norm(a)*np.linalg.norm(b) + 1e-6)
    return float(np.dot(a, b) / denom)

class CanonicalID:
    def __init__(self, iou_w=0.5, sim_w=0.5, iou_thr=0.3, sim_thr=0.65, ttl_sec=2.5):
        self.iou_w   = iou_w; self.sim_w = sim_w
        self.iou_thr = iou_thr; self.sim_thr = sim_thr
        self.ttl_sec = ttl_sec
        self.next_cid = 0
        self.tid2cid = {}
        self.cid_db  = {}  # cid -> {'feat','box','t_last'}

    def update(self, frame, tid, box, now_ts):
        if tid in self.tid2cid:
            cid = self.tid2cid[tid]
            feat = crop_hist_feat(frame, box)
            self.cid_db[cid] = {'feat': feat, 'box': box, 't_last': now_ts}
            return cid

        # 清過期
        for cid, rec in list(self.cid_db.items()):
            if now_ts - rec['t_last'] > self.ttl_sec:
                del self.cid_db[cid]
                for k, v in list(self.tid2cid.items()):
                    if v == cid:
                        del self.tid2cid[k]

        # 匹配
        feat = crop_hist_feat(frame, box)
        best_cid, best_score = None, -1.0
        for cid, rec in self.cid_db.items():
            iou = bbox_iou(box, rec['box']); sim = cosine_sim(feat, rec['feat'])
            score = self.iou_w * iou + self.sim_w * sim
            if score > best_score:
                best_score, best_cid = score, cid
        if best_cid is not None:
            rec = self.cid_db[best_cid]
            iou = bbox_iou(box, rec['box']); sim = cosine_sim(feat, rec['feat'])
            if (iou >= self.iou_thr) or (sim >= self.sim_thr) or (best_score >= 0.7):
                cid = best_cid
                self.cid_db[cid] = {'feat': feat, 'box': box, 't_last': now_ts}
                self.tid2cid[tid] = cid
                return cid

        # 新建
        cid = self.next_cid; self.next_cid += 1
        self.cid_db[cid] = {'feat': feat, 'box': box, 't_last': now_ts}
        self.tid2cid[tid] = cid
        return cid

# ========= 計數器（只做邏輯，不畫圖） ========= #
class LineCounter:
    def __init__(self, name, p1, p2, directional=False, cooldown=2.5):
        self.name = name; self.p1, self.p2 = p1, p2
        self.directional = directional; self.cooldown = cooldown
        self.crossed = 0; self.in_cnt = 0; self.out_cnt = 0
        self.unique_cids = set(); self.last_cross_t = {}
    def process(self, cid, hist, now_ts):
        if len(hist) < 2: return False
        if self.directional:
            dir_flag = crossing_direction(hist[-2], hist[-1], self.p1, self.p2); crossed = dir_flag != 0
        else:
            dir_flag = 0; crossed = crossed_line(hist[-2], hist[-1], self.p1, self.p2)
        if not crossed: return False
        last_t = self.last_cross_t.get(cid, 0.0)
        if (now_ts - last_t) < self.cooldown: return False
        self.crossed += 1
        if self.directional:
            if dir_flag > 0: self.in_cnt += 1
            else: self.out_cnt += 1
        self.last_cross_t[cid] = now_ts
        self.unique_cids.add(cid); return True

class RegionCounter:
    def __init__(self, name, polygon, cooldown=1.0):
        self.name = name; self.poly = polygon; self.cooldown = cooldown
        self.inside_cids = set(); self.last_event_t = {}
        self.entries = 0; self.exits = 0; self.unique_entries = set()
    def update(self, cid, point, now_ts):
        prev_inside = (cid in self.inside_cids)
        now_inside = point_in_polygon(point, self.poly)
        if prev_inside == now_inside: return
        last_t = self.last_event_t.get(cid, 0.0)
        if (now_ts - last_t) < self.cooldown: return
        if not prev_inside and now_inside:
            self.entries += 1; self.inside_cids.add(cid); self.unique_entries.add(cid)
        elif prev_inside and not now_inside:
            self.exits += 1; self.inside_cids.discard(cid)
        self.last_event_t[cid] = now_ts

# ========= Pillow 中文繪字 ========= #
def _find_font_path(custom_path: str = "") -> str:
    if custom_path and os.path.exists(custom_path): return custom_path
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/Library/Fonts/Heiti.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "C:\\Windows\\Fonts\\msjh.ttc","C:\\Windows\\Fonts\\msjh.ttf",
        "C:\\Windows\\Fonts\\mingliu.ttc","C:\\Windows\\Fonts\\simhei.ttf",
    ]
    for p in candidates:
        if os.path.exists(p): return p
    return ""

def draw_text(img_bgr, text, xy, font_path="", font_size=24, color=(255,255,0), stroke=2, stroke_color=(0,0,0)):
    img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    fp = _find_font_path(font_path)
    try:    font = ImageFont.truetype(fp, font_size) if fp else ImageFont.load_default()
    except: font = ImageFont.load_default()
    if stroke > 0:
        draw.text(xy, text, font=font, fill=tuple(color), stroke_width=stroke, stroke_fill=stroke_color)
    else:
        draw.text(xy, text, font=font, fill=tuple(color))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def measure_text(text, font_path="", font_size=24, stroke=0):
    fp = _find_font_path(font_path)
    try:    font = ImageFont.truetype(fp, font_size) if fp else ImageFont.load_default()
    except: font = ImageFont.load_default()
    dummy = Image.new('RGB', (10, 10)); d = ImageDraw.Draw(dummy)
    x0,y0,x1,y1 = d.textbbox((0, 0), text, font=font, stroke_width=stroke)
    return (x1 - x0, y1 - y0)

# ========= 主程式 ========= #
def main():
    args = parse_args()

    # 解析輸入解析度
    if args.width > 0 and args.height > 0:
        W, H = args.width, args.height
    else:
        W, H = (1920, 1080) if args.res == "1080" else (1280, 720)
    F = args.fps

    # Profile & yt-dlp 格式
    prof = PROFILES.get(args.profile, PROFILES["default"])
    if args.profile == "far" and "--ytfmt" not in sys.argv and prof.get("ytfmt"):
        args.ytfmt = prof["ytfmt"]

    # URL 清單
    urls = args.urls
    if not urls:
        env_multi = os.environ.get("STREAM_URLS")
        if env_multi:
            urls = [u.strip() for u in env_multi.split(",") if u.strip()]
    if not urls:
        one = os.environ.get("STREAM_URL")
        if one: urls = [one]
    if not urls:
        print("請用 --urls <src...> 或 export STREAM_URLS/STREAM_URL（YouTube 連結/ID 或 m3u8）")
        sys.exit(1)

    names = args.names or []
    def get_name(i): return names[i] if i < len(names) else f"Stream {i+1}"

    # 模型
    model = YOLO(MODEL_PATH, task='track')
    try:    model.to('cuda:0')
    except: 
        print("[WARN] 未使用 GPU，切換到 CPU（性能會下降）")
        model.to('cpu')

    # ReID profile
    reid_cfg = _select_reid_profile(args)

    def new_state(reid_cfg):
        return {
            "track_hist": {},
            "id_linker": CanonicalID(
                iou_w=reid_cfg["iou_w"], sim_w=reid_cfg["sim_w"],
                iou_thr=reid_cfg["iou_thr"], sim_thr=reid_cfg["sim_thr"],
                ttl_sec=reid_cfg["ttl_sec"]
            ),
            "line_counters": [LineCounter(**cfg) for cfg in LINES],
            "region_counters": [RegionCounter(**cfg) for cfg in REGIONS],
        }

    idx = 0
    state = new_state(reid_cfg)

    # ---- 人數平滑器（EMA + 防抖 + 掉線保護）----
    ema_people = None
    shown_people = 0
    hold_left = 0
    last_detection_ts = time.time()
    last_decay_tick = time.time()

    window_name = f"YouTube Live Counter ({W}x{H}@{F})"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def open_pipe(i):
        src = urls[i]
        resolved = resolve_stream(src, args.ytfmt)
        print(f"[INFO] 開啟串流 #{i+1}: {get_name(i)} | {src}")
        return start_ffmpeg_pipe(resolved, W, H, F)

    pipe = open_pipe(idx)
    frame_bytes = W * H * 3

    while True:
        raw = pipe.stdout.read(frame_bytes)
        if len(raw) != frame_bytes:
            print("[WARN] FFmpeg pipe 中斷，嘗試重新解析並重啟…")
            pipe.kill()
            try:
                pipe = open_pipe(idx)
            except Exception:
                time.sleep(1.0); continue
            time.sleep(0.3); continue

        # ---- 讀幀（BGR 彩色） ----
        frame = np.frombuffer(raw, np.uint8).reshape((H, W, 3)).copy()

        # ---- YOLOv8 + ByteTrack ----
        tracker_path = _ensure_tracker_yaml(args.profile, prof["tracker_yaml"])
        results = model.track(
            source=frame, imgsz=prof["imgsz"], conf=prof["conf"], iou=prof["iou"],
            classes=[CLASS_ID], tracker=tracker_path, persist=True, max_det=prof["max_det"]
        )

        now_ts = time.time()
        detected_this_frame = 0
        frame_tids = set()  # 用於 count-mode=tid

        for r in results:
            if r.boxes is None: continue
            for box in r.boxes:
                if box.id is None: continue
                try:    tid = int(box.id.item())
                except:  tid = int(box.id)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2)//2, (y1 + y2)//2

                cid = state["id_linker"].update(frame, tid, (x1, y1, x2, y2), now_ts)
                detected_this_frame += 1
                frame_tids.add(tid)

                # 保存歷史軌跡 + 計邏輯
                hist = state["track_hist"].setdefault(cid, deque(maxlen=6))
                hist.append((cx, cy))
                for lc in state["line_counters"]: lc.process(cid, hist, now_ts)
                for rc in state["region_counters"]: rc.update(cid, (cx, cy), now_ts)

                # 可視化（想純數字可註解掉）
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.circle(frame, (cx,cy), 4, (255,0,0), -1)
                cv2.putText(frame, f'CID:{cid} (tid:{tid})', (cx-10, cy-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        if detected_this_frame > 0:
            last_detection_ts = now_ts

        # ---- 目標人數：依 count-mode 選擇 ----
        if args.count_mode == "boxes":
            target_people = detected_this_frame                 # 一框一人
        elif args.count_mode == "tid":
            target_people = len(frame_tids)                     # 本幀唯一追蹤ID
        else:  # active
            active_cids = sum(1 for _, rec in state["id_linker"].cid_db.items()
                              if (now_ts - rec['t_last']) <= args.active_ttl)
            target_people = active_cids

        # ---- EMA 平滑與防抖 ----
        if ema_people is None:
            ema_people = float(target_people)
        else:
            ema_people = args.ema_alpha * float(target_people) + (1.0 - args.ema_alpha) * ema_people
        cand = int(round(ema_people))

        # 掉線保護
        no_det_elapsed = now_ts - last_detection_ts
        if no_det_elapsed > args.no_det_hold:
            if (now_ts - last_decay_tick) >= args.decay_interval and shown_people > 0:
                shown_people = max(0, shown_people - 1); last_decay_tick = now_ts
            display_people = shown_people
        else:
            if hold_left > 0:
                hold_left -= 1; display_people = shown_people
            else:
                if abs(cand - shown_people) >= args.step_thr:
                    shown_people = cand; display_people = shown_people
                    hold_left = args.hold_frames
                else:
                    display_people = shown_people

        # ---- HUD ----
        hud_y, pad, alpha, fs, stroke = 72, 8, 0.35, 32, 2
        label = f'現在人數: {display_people}'
        tw, th = measure_text(label, font_path=args.font, font_size=fs, stroke=stroke)
        box_w, box_h = tw + pad * 2, th + pad * 2
        overlay = frame.copy()
        cv2.rectangle(overlay, (6, hud_y - (pad + 4)), (6 + box_w, hud_y - (pad + 4) + box_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        frame = draw_text(frame, label, (14, hud_y),
                          font_path=args.font, font_size=fs, color=(0, 255, 255),
                          stroke=stroke, stroke_color=(0, 0, 0))

        cv2.putText(frame, "Keys: [n] next  [p] prev  [r] reconnect  [ESC] quit",
                    (10, H-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord('n'):
            pipe.kill(); idx = (idx + 1) % len(urls)
            try:
                if hasattr(model, "tracker") and model.tracker is not None: model.tracker.reset()
            except: pass
            state = new_state(reid_cfg)
            ema_people = None; shown_people = 0; hold_left = 0
            last_detection_ts = time.time(); last_decay_tick = time.time()
            pipe = open_pipe(idx); time.sleep(0.2)
        elif key == ord('p'):
            pipe.kill(); idx = (idx - 1 + len(urls)) % len(urls)
            try:
                if hasattr(model, "tracker") and model.tracker is not None: model.tracker.reset()
            except: pass
            state = new_state(reid_cfg)
            ema_people = None; shown_people = 0; hold_left = 0
            last_detection_ts = time.time(); last_decay_tick = time.time()
            pipe = open_pipe(idx); time.sleep(0.2)
        elif key == ord('r'):
            pipe.kill()
            try:
                if hasattr(model, "tracker") and model.tracker is not None: model.tracker.reset()
            except: pass
            state = new_state(reid_cfg)
            ema_people = None; shown_people = 0; hold_left = 0
            last_detection_ts = time.time(); last_decay_tick = time.time()
            pipe = open_pipe(idx); time.sleep(0.2)

    pipe.kill()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
