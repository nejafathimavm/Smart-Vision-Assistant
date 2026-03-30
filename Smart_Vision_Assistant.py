# ════════════════════════════════════════════════════════════════
# 🔥 SMART VISION ASSISTANT - FULLY FIXED VERSION
# YOLO + MediaPipe Tasks + Voice Input + Voice Output + Groq LLM
#
# FIXES:
#   ✅ Updated Groq model (llama3-70b was decommissioned)
#   ✅ Person is NEVER detected, shown, or selectable
#   ✅ Smallest box wins when overlapping (select the object not person)
#   ✅ Confidence threshold 65% to reduce noise
#   ✅ Max 5 objects only
#   ✅ No pyaudio - works on Python 3.14
#
# Install:
#   pip install opencv-python mediapipe ultralytics groq pyttsx3
#   pip install sounddevice soundfile SpeechRecognition numpy
# ════════════════════════════════════════════════════════════════

GROQ_API_KEY = "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"

import cv2
import numpy as np
import mediapipe as mp
import os, time, threading, urllib.request, tempfile, wave
from collections import deque

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Download hand model ───────────────────────────────────────────────────────
MODEL_URL  = ("https://storage.googleapis.com/mediapipe-models/" 
              "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("[INFO] Downloading hand landmark model ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[INFO] Model downloaded ✓")

# ── YOLO ──────────────────────────────────────────────────────────────────────
from ultralytics import YOLO
yolo = YOLO("yolov8n.pt")

# ── sounddevice ───────────────────────────────────────────────────────────────
try:
    import sounddevice as sd
    SD_OK = True
    print("[INFO] sounddevice ready ✓")
except ImportError:
    SD_OK = False
    print("[WARN] pip install sounddevice soundfile")

# ── SpeechRecognition ─────────────────────────────────────────────────────────
try:
    import speech_recognition as sr
    SR_OK = True
    print("[INFO] SpeechRecognition ready ✓")
except ImportError:
    SR_OK = False
    print("[WARN] pip install SpeechRecognition")

# ── pyttsx3 ───────────────────────────────────────────────────────────────────
try:
    import pyttsx3
    TTS_OK = True
    print("[INFO] pyttsx3 TTS ready ✓")
except ImportError:
    TTS_OK = False
    print("[WARN] pip install pyttsx3")

# ── Groq ──────────────────────────────────────────────────────────────────────
try:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY)
    GROQ_OK = True
    print("[INFO] Groq LLM ready ✓")
except Exception as e:
    GROQ_OK = False
    print(f"[WARN] Groq not available: {e}")

# ════════════════════════════════════════════════════════════════
# ⚙️  SETTINGS
# ════════════════════════════════════════════════════════════════

# ✅ FIX 1: Updated model — llama3-70b-8192 is decommissioned
GROQ_MODEL   = "llama-3.3-70b-versatile"

# ✅ FIX 2: Person and body parts are ALWAYS blocked
BLOCKED_LABELS = {
    "person", "man", "woman", "human", "face",
    "hand", "arm", "head", "body", "people"
}

CONF_THRESH  = 0.65   # higher = less false detections
MAX_OBJECTS  = 5      # max objects on screen at once
SELECT_HOLD  = 1.0    # seconds to hold 2-finger gesture
SAMPLE_RATE  = 16000
RECORD_SECS  = 6

# Colours (BGR)
C_GREEN  = (  0, 210, 100)
C_CYAN   = (255, 210,   0)
C_YELLOW = (  0, 220, 255)
C_WHITE  = (240, 240, 240)
C_PANEL  = ( 25,  25,  45)
C_SEL    = (255, 120,   0)
C_MIC    = (  0, 200,  80)
C_GRAY   = (130, 130, 160)


# ══════════════════════════════════════════════════════════════════════════════
# 🎯 Gesture Tracker
# ══════════════════════════════════════════════════════════════════════════════
class GestureTracker:
    def __init__(self):
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self.detector  = mp_vision.HandLandmarker.create_from_options(opts)
        self._frame_ms = 0
        self._tips     = [4, 8, 12, 16, 20]

    def process(self, frame):
        self._frame_ms += 1
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res    = self.detector.detect_for_video(mp_img, self._frame_ms)

        count, ptr = 0, None
        if res.hand_landmarks:
            pts = res.hand_landmarks[0]
            H, W = frame.shape[:2]
            if pts[4].x < pts[3].x: count += 1
            for tip in self._tips[1:]:
                if pts[tip].y < pts[tip-2].y: count += 1
            ptr = (pts[8].x, pts[8].y)
            for lm in pts:
                cv2.circle(frame, (int(lm.x*W), int(lm.y*H)), 4, (0,255,100), -1)
        return count, ptr, frame


# ══════════════════════════════════════════════════════════════════════════════
# 🎤 Voice Input  (sounddevice, no pyaudio, works on Python 3.14)
# ══════════════════════════════════════════════════════════════════════════════
class VoiceInput:
    def __init__(self):
        self.recognizer = sr.Recognizer() if SR_OK else None

    def listen(self) -> str:
        if not SD_OK or not SR_OK:
            print("[MIC] Voice unavailable")
            return ""
        try:
            print(f"[MIC] Recording {RECORD_SECS}s — speak your question now ...")
            audio_data = sd.rec(
                int(RECORD_SECS * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
            )
            sd.wait()

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(audio_data.tobytes())

            with sr.AudioFile(tmp_path) as source:
                audio = self.recognizer.record(source)
            os.unlink(tmp_path)

            text = self.recognizer.recognize_google(audio)
            print(f"[MIC] Heard: \"{text}\"")
            return text.strip()

        except sr.UnknownValueError:
            print("[MIC] Could not understand — please speak clearly")
            return ""
        except sr.RequestError as e:
            print(f"[MIC] Google STT error (need internet): {e}")
            return ""
        except Exception as e:
            print(f"[MIC] Error: {e}")
            return ""


# ══════════════════════════════════════════════════════════════════════════════
# 🔊 Voice Output  (pyttsx3, no pyaudio needed)
# ══════════════════════════════════════════════════════════════════════════════
class VoiceOutput:
    def speak(self, text: str):
        if not TTS_OK or not text:
            return
        def _run():
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", 155)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as e:
                print(f"[TTS] Error: {e}")
        threading.Thread(target=_run, daemon=True).start()


# ══════════════════════════════════════════════════════════════════════════════
# 🧠 Groq LLM  (updated model + better prompt)
# ══════════════════════════════════════════════════════════════════════════════
def ask_groq(label: str, question: str) -> str:
    if GROQ_OK:
        try:
            r = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                max_tokens=180,
                temperature=0.4,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise vision assistant. "
                            "The user is pointing a camera at a physical object. "
                            "Answer ONLY about that specific object. "
                            "Be factual, direct, and short — 2 sentences maximum. "
                            "Never mention people or the camera."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"The object in front of the camera is: {label}\n"
                            f"Question about this {label}: {question}"
                        )
                    },
                ],
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            return f"LLM Error: {e}"
    return f"A {label} is a common everyday object."


# ══════════════════════════════════════════════════════════════════════════════
# 🎨 Drawing helpers
# ══════════════════════════════════════════════════════════════════════════════
def alpha_rect(img, x1, y1, x2, y2, color, alpha=0.75):
    x1,y1 = max(0,x1), max(0,y1)
    x2,y2 = min(img.shape[1],x2), min(img.shape[0],y2)
    sub = img[y1:y2, x1:x2]
    if sub.size == 0: return
    overlay = np.full(sub.shape, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, sub, 1-alpha, 0, sub)
    img[y1:y2, x1:x2] = sub

def put(img, text, x, y, scale=0.48, color=C_WHITE, thick=1):
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick, cv2.LINE_AA)

def wrap(img, text, x, y, max_w, scale=0.40, color=C_WHITE, gap=16):
    words, line, lines = text.split(), "", []
    for w in words:
        t = (line+" "+w).strip()
        if cv2.getTextSize(t,cv2.FONT_HERSHEY_SIMPLEX,scale,1)[0][0]>max_w and line:
            lines.append(line); line=w
        else: line=t
    if line: lines.append(line)
    for i,l in enumerate(lines):
        put(img, l, x, y+i*gap, scale, color)
    return len(lines)*gap
 

# ══════════════════════════════════════════════════════════════════════════════
# 🚀 Main Application
# ══════════════════════════════════════════════════════════════════════════════
class App:
    IDLE      = "idle"
    LISTENING = "listening"
    THINKING  = "thinking"
    SPEAKING  = "speaking"

    def __init__(self):
        self.gesture   = GestureTracker()
        self.mic       = VoiceInput()
        self.tts       = VoiceOutput()

        self.selected  = None
        self.cursor    = (0, 0)
        self.fingers   = 0
        self.sel_start = None
        self.sel_prog  = 0.0
        self.sel_flash = 0.0

        self.qa_log    = deque(maxlen=5)
        self.last_q    = ""
        self.v_state   = self.IDLE

        self._running    = True
        self._dets       = []
        self._frame_lock = threading.Lock()
        self._raw_frame  = None

        self._det_thread   = threading.Thread(target=self._yolo_loop,  daemon=True)
        self._voice_thread = threading.Thread(target=self._voice_loop, daemon=True)

    # ── YOLO background thread ────────────────────────────────────────────────
    def _yolo_loop(self):
        while self._running:
            with self._frame_lock:
                f = self._raw_frame
            if f is not None:
                try:
                    res  = yolo(f, conf=CONF_THRESH, verbose=False)[0]
                    dets = []
                    for b in res.boxes:
                        label = res.names[int(b.cls[0])]

                        # ✅ FIX: Skip person and all body-related labels
                        if label.lower() in BLOCKED_LABELS:
                            continue

                        dets.append({
                            "label": label,
                            "conf":  float(b.conf[0]),
                            "box":   tuple(map(int, b.xyxy[0].tolist()))
                        })

                    # Keep only top MAX_OBJECTS by confidence
                    dets.sort(key=lambda d: d["conf"], reverse=True)
                    self._dets = dets[:MAX_OBJECTS]

                except Exception as e:
                    print(f"[YOLO] {e}")
            time.sleep(0.08)

    # ── Voice pipeline thread ─────────────────────────────────────────────────
    def _voice_loop(self):
        while self._running:
            if not self.selected or self.v_state != self.IDLE:
                time.sleep(0.2)
                continue

            label = self.selected
            print(f"\n[VOICE] '{label}' selected — speak your question ...")

            self.v_state = self.LISTENING
            question = self.mic.listen()

            if not question or not self.selected:
                self.v_state = self.IDLE
                time.sleep(0.3)
                continue

            self.last_q  = question
            self.v_state = self.THINKING
            print(f"[LLM]  Q: \"{question}\"  →  object: '{label}'")
            answer = ask_groq(label, question)
            print(f"[LLM]  A: {answer}")

            self.qa_log.append((question, answer))
            self.v_state = self.SPEAKING
            self.tts.speak(answer)

            time.sleep(1.5)
            self.v_state = self.IDLE
            print("[VOICE] Ready for next question.")

    # ── ✅ FIX: Hovered returns SMALLEST box under cursor (not person) ─────────
    def _hovered(self):
        cx, cy = self.cursor
        candidates = []
        for d in self._dets:
            x1, y1, x2, y2 = d["box"]
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                area = (x2-x1) * (y2-y1)   # smaller box = more specific object
                candidates.append((area, d["label"]))

        if not candidates:
            return None

        # Return the label of the SMALLEST overlapping box
        # (person box is huge; object boxes are small → object wins)
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    # ── Draw bounding boxes ───────────────────────────────────────────────────
    def _draw_boxes(self, frame):
        for d in self._dets:
            x1, y1, x2, y2 = d["box"]
            is_sel = (self.selected == d["label"])
            col    = C_SEL if is_sel else C_GREEN
            thick  = 3     if is_sel else 2
            cv2.rectangle(frame, (x1,y1), (x2,y2), col, thick)

            tag = f"{d['label']}  {d['conf']:.0%}"
            (tw,th),_ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
            alpha_rect(frame, x1, max(0,y1-th-10), x1+tw+10, y1, col, 0.88)
            put(frame, tag, x1+5, y1-3, 0.50, (10,10,10), 1)

            if is_sel:
                s = 20
                for (ox,oy),(dx,dy) in [
                    ((x1,y1),(1,1)),((x2,y1),(-1,1)),
                    ((x1,y2),(1,-1)),((x2,y2),(-1,-1))]:
                    cv2.line(frame,(ox,oy),(ox+dx*s,oy),C_CYAN,3)
                    cv2.line(frame,(ox,oy),(ox,oy+dy*s),C_CYAN,3)

    # ── Draw HUD ──────────────────────────────────────────────────────────────
    def _draw_hud(self, frame):
        H, W  = frame.shape[:2]
        pulse = (np.sin(time.time()*5)+1)/2

        # Top bar
        alpha_rect(frame,0,0,W,40,C_PANEL,0.88)
        g_map = {0:"No hand",1:"1 finger - Move",
                 2:"2 fingers - Hold to select",5:"5 fingers - Reset"}
        g_txt   = g_map.get(self.fingers, f"{self.fingers} fingers")
        sel_txt = f"SELECTED: {self.selected.upper()}" if self.selected else "Nothing selected"
        put(frame, f"Smart Vision  |  {g_txt}  |  {sel_txt}", 10,26,0.52,C_CYAN)

        # Object count
        n = len(self._dets)
        put(frame, f"Objects: {n}", W-120, 26, 0.42, C_GRAY)

        # Selection progress bar
        if self.sel_prog > 0:
            cv2.rectangle(frame,(0,38),(int(W*self.sel_prog),42),C_SEL,-1)

        # Flash on select
        if self.sel_flash > 0:
            ov = frame.copy()
            cv2.rectangle(ov,(0,0),(W,H),(0,120,255),-1)
            cv2.addWeighted(ov,self.sel_flash*0.18,frame,1-self.sel_flash*0.18,0,frame)
            self.sel_flash = max(0.0, self.sel_flash-0.10)

        # Q&A panel
        PW = 370; PX = W-PW-8; PY = 50; PH = H-PY-32
        alpha_rect(frame,PX-6,PY-6,W-4,H-30,C_PANEL,0.88)
        cv2.rectangle(frame,(PX-6,PY-6),(W-4,H-30),(60,60,90),1)
        put(frame,"Q & A Panel",PX,PY+18,0.62,C_CYAN,2)
        cv2.line(frame,(PX,PY+26),(PX+PW,PY+26),C_CYAN,1)

        if not self.selected:
            put(frame,"How to use:",                    PX,PY+52, 0.48,C_YELLOW)
            put(frame,"1. Hold 2 fingers over object",  PX,PY+76, 0.42,C_GRAY)
            put(frame,"   until bar fills",             PX,PY+93, 0.42,C_GRAY)
            put(frame,"2. Object gets SELECTED",        PX,PY+114,0.42,C_GRAY)
            put(frame,"3. Speak your question aloud",   PX,PY+135,0.42,C_WHITE)
            put(frame,"4. Answer spoken + shown here",  PX,PY+156,0.42,C_GRAY)
            put(frame,"Person is always ignored",       PX,PY+186,0.40,(80,80,200))
            put(frame,f"Model: {GROQ_MODEL}",           PX,PY+206,0.38,C_GRAY)
        else:
            put(frame,f"Object: {self.selected}", PX,PY+46,0.56,C_YELLOW,1)
            put(frame,f"Model:  {GROQ_MODEL}",    PX,PY+64,0.36,C_GRAY)

            vs = self.v_state
            if vs == self.LISTENING:
                bw = int(50+35*pulse)
                cv2.rectangle(frame,(PX,PY+76),(PX+bw,PY+90),C_MIC,-1)
                put(frame,"Listening — speak now!",PX+bw+6,PY+88,0.43,C_MIC)
            elif vs == self.THINKING:
                dots = "."*(int(time.time()*4)%6)
                put(frame,f"Thinking{dots}",PX,PY+82,0.50,C_YELLOW)
                if self.last_q:
                    q_d=(self.last_q[:40]+"...") if len(self.last_q)>41 else self.last_q
                    put(frame,f'Heard: "{q_d}"',PX,PY+102,0.40,C_GRAY)
            elif vs == self.SPEAKING:
                put(frame,"Speaking answer...",PX,PY+82,0.50,(255,180,0))
            else:
                put(frame,"Ready — speak your question",PX,PY+82,0.44,C_GRAY)

            y = PY+108
            for q,a in reversed(self.qa_log):
                if y > PY+PH-10: break
                q_d=(q[:40]+"...") if len(q)>41 else q
                put(frame,f"Q: {q_d}",PX,y,0.41,C_YELLOW)
                y+=17
                used=wrap(frame,f"A: {a}",PX,y,PW-4,scale=0.39,
                          color=(160,220,160),gap=15)
                y+=used+8
                cv2.line(frame,(PX,y-2),(PX+PW-4,y-2),(55,55,80),1)

        # Bottom bar
        alpha_rect(frame,0,H-26,W,H,C_PANEL,0.88)
        put(frame,"ESC=Quit | 1finger=Move | 2fingers=Hold Select | 5fingers=Reset | Speak question after selecting",
            8,H-8,0.36,(110,110,150))

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Camera not found"); return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)

        self._det_thread.start()
        self._voice_thread.start()

        print("\n" + "="*58)
        print("  Smart Vision Assistant  —  Fully Fixed Version")
        print("="*58)
        print(f"  Groq model : {GROQ_MODEL}")
        print(f"  Confidence : {int(CONF_THRESH*100)}%  |  Max objects: {MAX_OBJECTS}")
        print(f"  Blocked    : {', '.join(BLOCKED_LABELS)}")
        print()
        print("  ☝  1 finger  = move pointer over objects")
        print("  ✌  2 fingers = hold still over object to select it")
        print("  🖐  5 fingers = reset / deselect")
        print("  Speak your question → hear answer + see it on screen")
        print("  Press ESC to quit")
        print("="*58+"\n")

        while True:
            ok, frame = cap.read()
            if not ok: time.sleep(0.03); continue

            frame = cv2.flip(frame, 1)
            H, W  = frame.shape[:2]

            with self._frame_lock:
                self._raw_frame = frame.copy()

            self.fingers, ptr, frame = self.gesture.process(frame)

            if ptr:
                cx = int(ptr[0]*W); cy = int(ptr[1]*H)
                self.cursor = (cx, cy)
                col = {1:C_YELLOW,2:C_SEL,5:C_CYAN}.get(self.fingers,(200,200,200))
                cv2.circle(frame,(cx,cy),14,col,2)
                cv2.circle(frame,(cx,cy), 4,col,-1)
                cv2.line(frame,(cx-18,cy),(cx+18,cy),col,1)
                cv2.line(frame,(cx,cy-18),(cx,cy+18),col,1)

            # 2-finger hold → select smallest object under cursor
            hov = self._hovered()
            if self.fingers == 2 and hov:
                if self.sel_start is None: self.sel_start = time.time()
                elapsed       = time.time()-self.sel_start
                self.sel_prog = min(elapsed/SELECT_HOLD, 1.0)
                if elapsed >= SELECT_HOLD:
                    self.selected  = hov
                    self.sel_start = None
                    self.sel_prog  = 0.0
                    self.sel_flash = 1.0
                    self.qa_log.clear()
                    self.last_q    = ""
                    self.v_state   = self.IDLE
                    print(f"\n[SELECT] {hov.upper()}")
            else:
                self.sel_start = None
                self.sel_prog  = 0.0

            # 5-finger → reset
            if self.fingers == 5:
                if self.selected: print("\n[RESET]")
                self.selected  = None
                self.v_state   = self.IDLE
                self.qa_log.clear()
                self.last_q    = ""

            self._draw_boxes(frame)
            self._draw_hud(frame)
            cv2.imshow("Smart Vision Assistant", frame)

            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

        self._running = False
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Closed.")


if __name__ == "__main__":
    App().run()