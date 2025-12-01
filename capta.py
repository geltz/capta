import sys
import os
import time
import math
import ctypes
import collections
import numpy as np
import soundfile as sf
import multiprocessing
import queue

from PyQt6.QtCore import (Qt, QThread, pyqtSignal, QTimer, QRectF, QPointF, 
                          QObject)
from PyQt6.QtGui import (QColor, QPainter, QLinearGradient, QPen, QPainterPath, 
                         QBrush, QFont, QRadialGradient, QIcon, QPolygonF, 
                         QImage, qRgba)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QSlider)
from scipy.fft import fft

# --- Numpy Compatibility ---

if hasattr(np, 'fromstring'):
    _orig_fromstring = np.fromstring
    def _fromstring_compat(string, dtype=float, count=-1, sep=''):
        if sep == '':
            return np.frombuffer(string, dtype=dtype, count=count)
        return _orig_fromstring(string, dtype=dtype, count=count, sep=sep)
    np.fromstring = _fromstring_compat

# --- Constants ---

SR = 48000
CHUNK_SIZE = 1024

# --- Audio Capture ---

def audio_worker(cmd_queue, vis_queue, result_queue):
    import pyaudiowpatch as pyaudio
    
    p = pyaudio.PyAudio()
    
    try:
        try:
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError:
            print("DEBUG: WASAPI not found. This requires Windows.")
            return

        default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        
        target_device = None
        for loopback in p.get_loopback_device_info_generator():
            if default_speakers["name"] in loopback["name"]:
                target_device = loopback
                break
        
        if target_device is None:
            target_device = next(p.get_loopback_device_info_generator())

        RATE = int(target_device["defaultSampleRate"])
        CHANNELS = int(target_device["maxInputChannels"])
        CHUNK = 1024 
        FORMAT = pyaudio.paInt16

        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            input_device_index=target_device["index"],
            frames_per_buffer=CHUNK,
            start=False
        )

    except Exception as e:
        print(f"DEBUG: Init failed: {e}")
        return

    captured_chunks = []
    recording = False
    running = True
    
    try:
        stream.start_stream()
        
        while running:
            # --- Command Handling ---
            try:
                if not cmd_queue.empty():
                    cmd, _ = cmd_queue.get_nowait()
                    if cmd == 'START':
                        captured_chunks = []
                        recording = True
                    elif cmd == 'STOP':
                        recording = False
                        if captured_chunks:
                            full_audio = b''.join(captured_chunks)
                            audio_array = np.frombuffer(full_audio, dtype=np.int16)
                            audio_array = audio_array.reshape(-1, CHANNELS).astype(np.float32) / 32767.0
                            if CHANNELS > 2:
                                audio_array = audio_array[:, :2]
                            result_queue.put((audio_array, RATE))
                        else:
                            result_queue.put(None)
                        captured_chunks = []
                    elif cmd == 'EXIT':
                        running = False
            except queue.Empty:
                pass
            
            if not running:
                break
            
            # --- Read Audio ---
            data = None
            try:
                # FIX: Check if data is available before reading.
                # WASAPI Loopback stops sending frames during silence, causing 
                # stream.read() to block indefinitely if we don't check this.
                if stream.get_read_available() >= CHUNK:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                else:
                    # Silence detected (no audio playing).
                    # Sleep briefly to prevent CPU spin and allow the loop 
                    # to check for the 'STOP' command.
                    time.sleep(0.01)
            except Exception:
                continue
            
            # Only process if we actually received data
            if data is not None:
                if recording:
                    captured_chunks.append(data)
                
                # --- Visualization ---
                try:
                    if vis_queue.qsize() < 3:
                        audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0
                        audio_np = audio_np.reshape(-1, CHANNELS)
                        if CHANNELS > 2:
                            audio_np = audio_np[:, :2]
                        vis_queue.put_nowait(audio_np)
                except:
                    pass
                
    except Exception as e:
        print(f"DEBUG: Loopback worker crashed: {e}")
    finally:
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
        p.terminate()

class SaveWorker(QThread):
    finished_saving = pyqtSignal(bool, str)

    def __init__(self, full_audio_array, sr):
        super().__init__()
        self.full_audio = full_audio_array
        self.sr = sr

    def run(self):
        try:
            if self.full_audio is None or len(self.full_audio) == 0:
                self.finished_saving.emit(False, "buffer empty")
                return

            # Make a copy to process
            proc_audio = np.array(self.full_audio, copy=True)
            
            # 1. DC Offset Removal
            proc_audio = proc_audio - np.mean(proc_audio, axis=0)
            
            # 2. Enforce Zero-Crossing via Micro-Fades (15ms)
            # This guarantees no clicks at start/end
            fade_duration = 0.015 
            fade_samples = int(self.sr * fade_duration)
            
            if len(proc_audio) > fade_samples * 2:
                # Create fade curves
                fade_in = np.linspace(0.0, 1.0, fade_samples).astype(np.float32)
                fade_out = np.linspace(1.0, 0.0, fade_samples).astype(np.float32)
                
                # Apply to channels (Handle Stereo/Mono)
                if proc_audio.ndim == 2:
                    # Broadcast shapes for stereo
                    fade_in = fade_in[:, np.newaxis]
                    fade_out = fade_out[:, np.newaxis]
                
                proc_audio[:fade_samples] *= fade_in
                proc_audio[-fade_samples:] *= fade_out

            # 3. Normalization (Conservative)
            peak = np.max(np.abs(proc_audio))
            if peak > 0.001:
                target_peak = 0.95
                gain = min(target_peak / peak, 3.0)
                proc_audio = proc_audio * gain
            
            proc_audio = np.clip(proc_audio, -1.0, 1.0)
            audio_int16 = (proc_audio * 32767).astype(np.int16)
                
            home_dir = os.path.expanduser("~")
            save_dir = os.path.join(home_dir, "Music", "capta")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
            timestamp = int(time.time())
            filename = f"capta_record_{timestamp}.wav"
            full_path = os.path.join(save_dir, filename)
            
            sf.write(full_path, audio_int16, self.sr, subtype='PCM_16')
            
            self.finished_saving.emit(True, f"saved to \nMusic/capta")
            
        except Exception as e:
            print(f"Save Error: {e}")
            import traceback
            traceback.print_exc()
            self.finished_saving.emit(False, f"error: {str(e)}")

# --- UI Elements ---

class CaptaLogo(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(80, 50)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.phase = 0.0
        self.base_hue = 0.0
        self.speed = 0.02
        self.target_speed = 0.02
        
        # Smooth hover logic
        self.target_opacity = 0.6
        self.current_opacity = 0.6
        
        # Geometry constants
        self.radius = 14
        self.offset_x = 16 
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(16)
        
    def mousePressEvent(self, event):
        self.speed = 0.8
        self.update()
        
    def enterEvent(self, event):
        self.target_opacity = 1.0 # Fully opaque/saturated on hover
        super().enterEvent(event)
        
    def leaveEvent(self, event):
        self.target_opacity = 0.6 # Faded when idle
        super().leaveEvent(event)
        
    def animate(self):
        # Physics for rotation speed
        if self.speed > self.target_speed:
            self.speed = self.speed * 0.95 + self.target_speed * 0.05
            
        self.phase = (self.phase + self.speed) % (2 * math.pi)
        self.base_hue = (self.base_hue + 0.002) % 1.0
        
        # Physics for opacity
        diff = self.target_opacity - self.current_opacity
        if abs(diff) > 0.001:
            self.current_opacity += diff * 0.1
            
        self.update()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.setOpacity(self.current_opacity)
        
        rect = self.rect()
        cy, cx = rect.height() / 2, rect.width() / 2
        
        grad = QLinearGradient(0, 0, rect.width(), 0)
        grad.setColorAt(0.0, QColor.fromHslF(self.base_hue, 0.6, 0.75))
        grad.setColorAt(1.0, QColor.fromHslF((self.base_hue + 0.15) % 1.0, 0.6, 0.75))
        
        pen = QPen(QBrush(grad), 2.5) 
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        
        painter.drawLine(QPointF(cx - self.offset_x, cy - self.radius), 
                         QPointF(cx + self.offset_x, cy - self.radius))
                         
        def draw_reel(center, offset):
            painter.save()
            painter.translate(center)
            painter.rotate(math.degrees(self.phase + offset))
            painter.drawEllipse(QPointF(0, 0), self.radius, self.radius)
            for i in range(3):
                theta = (i / 3.0) * 2 * math.pi
                painter.drawLine(QPointF(0,0), QPointF(math.cos(theta)*self.radius, math.sin(theta)*self.radius))
            painter.restore()
            
        draw_reel(QPointF(cx - self.offset_x, cy), 0)
        draw_reel(QPointF(cx + self.offset_x, cy), 1.0)

class CaptaButton(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(84, 84)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.is_recording = False
        self.phase = 0.0
        self.hover_val = 0.0 
        self.record_anim_val = 0.0
        self.setMouseTracking(True)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(16)
    def animate(self):
        target_hover = 1.0 if self.underMouse() else 0.0
        self.hover_val += (target_hover - self.hover_val) * 0.1
        target_rec = 1.0 if self.is_recording else 0.0
        if abs(self.record_anim_val - target_rec) > 0.001:
            self.record_anim_val += (target_rec - self.record_anim_val) * 0.08
        if self.record_anim_val > 0.01:
            self.phase = (self.phase + 0.05) % (2 * math.pi)
        else:
            self.phase = 0.0
        self.update()
    def set_recording(self, state):
        self.is_recording = state
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        cx, cy = rect.center().x(), rect.center().y()
        idle_radius = 26
        max_rec_radius = 34
        base_r = idle_radius + ((max_rec_radius - idle_radius) * self.record_anim_val)
        pulse = 0
        if self.record_anim_val > 0.01:
            breath = (math.sin(self.phase) + 1) / 2 
            pulse = breath * 2.0 * self.record_anim_val
        current_r = base_r + pulse
        h_idle, s_idle, l_idle = 210/360, 0.40, 0.90 
        c_idle = QColor.fromHslF(h_idle, s_idle, l_idle)
        c_rec = QColor(255, 60, 60)
        r = c_idle.red() + (c_rec.red() - c_idle.red()) * self.record_anim_val
        g = c_idle.green() + (c_rec.green() - c_idle.green()) * self.record_anim_val
        b = c_idle.blue() + (c_rec.blue() - c_idle.blue()) * self.record_anim_val
        main_color = QColor(int(r), int(g), int(b))
        if self.record_anim_val > 0.01:
            glow_alpha = int(40 * self.record_anim_val)
            painter.setBrush(QColor(255, 50, 50, glow_alpha))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(cx, cy), current_r + 4, current_r + 4)
        grad = QLinearGradient(0, 0, rect.width(), rect.height())
        grad.setColorAt(0, main_color)
        grad.setColorAt(1, main_color.darker(104))
        painter.setBrush(grad)
        painter.setPen(Qt.PenStyle.NoPen)
        hover_exp = self.hover_val * 2.0 * (1.0 - self.record_anim_val)
        painter.drawEllipse(QPointF(cx, cy), current_r + hover_exp, current_r + hover_exp)
        painter.setBrush(QColor("white"))
        icon_size = 14
        corner_radius = 7 - (5 * self.record_anim_val)
        painter.drawRoundedRect(
            QRectF(cx - icon_size/2, cy - icon_size/2, icon_size, icon_size), 
            corner_radius, corner_radius
        )

class WaveformStrip(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(60)
        # Increased buffer slightly for a denser "bar" look
        self.buffer = collections.deque([0.0]*160, maxlen=160)
        self.state_buffer = collections.deque([False]*160, maxlen=160) 
        self.is_rec = False
        self.hue = 0.0
        self.tape_anim_val = 0.0

    def push_data(self, chunk, is_rec):
        self.is_rec = is_rec
        if len(chunk) == 0: val = 0.0
        else: 
            # CHANGED: Reduced multiplier from 3.5 to 1.5 
            # This makes the input sensitivity lower
            val = np.sqrt(np.mean(chunk**2)) * 1.5
        
        last = self.buffer[-1] if self.buffer else 0.0
        val = last * 0.6 + val * 0.4 
        self.buffer.append(val)
        self.state_buffer.append(is_rec)
        
        target = 1.0 if is_rec else 0.0
        self.tape_anim_val += (target - self.tape_anim_val) * 0.1
        self.update()

    def set_hue(self, h):
        self.hue = h
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        w, h = self.width(), self.height()
        cy = h / 2.0
        count = len(self.buffer)
        
        if count < 2: return

        # Geometry
        # Calculate spacing so it fills the width
        step_x = w / (count - 1)
        # Bar width is slightly less than step to leave a tiny gap
        pen_width = max(1.5, step_x * 0.7)
        
        # Colors
        # Idle: Very subtle pastel
        c_idle = QColor.fromHslF(self.hue, 0.35, 0.75)
        # Recording: Saturated Salmon
        c_rec = QColor.fromHslF(0.0, 0.85, 0.65)
        
        # Pen setup
        pen = QPen()
        pen.setWidthF(pen_width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)

        for i, val in enumerate(self.buffer):
            # 1. Logic: Position
            x = i * step_x
            
            # 2. Logic: Height (Amplitude)
            # Base height
            amp = max(2.0, val * h * 0.6)
            
            # 3. Logic: Fading (Left/Right Edges)
            # Create a "Hanning Window" curve (0 -> 1 -> 0)
            norm_x = i / count
            edge_factor = np.sin(norm_x * np.pi)
            
            # Apply fade to height (Bars get shorter at edges)
            draw_h = amp * pow(edge_factor, 0.3) 
            
            # 4. Logic: Color & Opacity
            is_recording_slice = self.state_buffer[i]
            
            if is_recording_slice:
                base_c = c_rec
                # Recording bars are fully opaque in center, fade at edges
                alpha = 255 * edge_factor
            else:
                base_c = c_idle
                # Idle bars are naturally more transparent + edge fade
                alpha = 180 * edge_factor

            # Apply Alpha
            final_c = QColor(base_c)
            final_c.setAlpha(int(alpha))
            
            # Draw
            pen.setColor(final_c)
            painter.setPen(pen)
            
            # Draw vertical line centered at cy
            painter.drawLine(QPointF(x, cy - draw_h/2), QPointF(x, cy + draw_h/2))

class VisualizerPanel(QWidget):
    MODE_BARS = 0
    MODE_SCOPE = 1
    MODE_SPECTRO = 2
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mode = self.MODE_BARS
        self.hue = 0.0
        
        # --- Data Buffers ---
        self.target_bars = np.zeros(64)
        self.displayed_bars = np.zeros(64)
        self.raw_audio_buffer = np.zeros(1024)
        
        # Spectrogram: 256 bins height, 400 pixels history width
        self.spec_height = 256 
        self.spec_width = 400
        # Buffer stores 0.0-1.0 float values
        self.spec_buffer = np.zeros((self.spec_height, self.spec_width), dtype=np.float32)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(7) 

    def set_hue(self, h): 
        self.hue = h

    def update_audio(self, data):
        # Audio Processing / FFT
        if data.ndim > 1: data = np.mean(data, axis=1)
        target_len = 1024
        if len(data) != target_len:
            if len(data) > target_len: data = data[:target_len]
            else: 
                padded = np.zeros(target_len); padded[:len(data)] = data; data = padded
        self.raw_audio_buffer = data

        # Windowing
        windowed = data * np.hanning(len(data))
        fft_d = np.abs(fft(windowed)[:len(windowed)//2])
        
        num_bars = len(self.target_bars)
        
        if len(fft_d) > 0:
            # --- FIX: Smooth Interpolation vs. Peak Binning ---
            # We use float boundaries to prevent "stepping" at low frequencies.
            
            # Create logarithmic edges (indices) as floats
            # Start at index 1 to skip DC offset, go to end of array
            edges = np.logspace(0, np.log10(len(fft_d)-1), num_bars + 1)
            
            new_targets = []
            for i in range(num_bars):
                low_f = edges[i]
                high_f = edges[i+1]
                
                width = high_f - low_f
                
                # Logic: If the bin is narrower than 1 index (Low Freqs),
                # we must interpolate to get a unique value for this bar.
                if width < 1.0:
                    mid = (low_f + high_f) / 2.0
                    idx_0 = int(mid)
                    # Safety clamp
                    idx_1 = min(idx_0 + 1, len(fft_d) - 1)
                    
                    # Linear Interpolation
                    t = mid - idx_0
                    val = fft_d[idx_0] * (1 - t) + fft_d[idx_1] * t
                else:
                    # Logic: If the bin is wide (High Freqs), 
                    # we take the MAX to ensure we catch loud treble sounds.
                    start_idx = int(low_f)
                    end_idx = int(np.ceil(high_f))
                    chunk = fft_d[start_idx:end_idx]
                    val = np.max(chunk) if len(chunk) > 0 else 0.0
                
                new_targets.append(val)
            
            new_targets = np.array(new_targets)
            new_targets = np.log10(new_targets + 1) * 0.3
            self.target_bars = new_targets

        # Spectro buffer logic unchanged...
        self.spec_buffer = np.roll(self.spec_buffer, -1, axis=1)
        if len(fft_d) > 0:
            fft_log = np.log10(fft_d + 1e-9)
            fft_norm = np.clip((fft_log + 2.5) / 5.5, 0, 1)
            x_old = np.linspace(0, 1, len(fft_norm))
            x_new = np.linspace(0, 1, self.spec_height)
            col_data = np.interp(x_new, x_old, fft_norm)
            self.spec_buffer[:, -1] = np.flip(col_data)

    def mousePressEvent(self, event):
        self.mode = (self.mode + 1) % 3
        self.update()

    def animate(self):
        if self.mode == self.MODE_BARS:
            lerp = 0.15
            self.displayed_bars += (self.target_bars - self.displayed_bars) * lerp
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        
        w, h = self.width(), self.height()
        rect = self.rect()

        p.setBrush(QColor(245, 247, 250))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(rect, 12, 12)
        
        # Clip to rounded corners
        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), 12, 12)
        p.setClipPath(path)
        
        if self.mode == self.MODE_BARS: 
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            self.draw_bars(p, w, h)
        elif self.mode == self.MODE_SCOPE:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            self.draw_scope(p, w, h)
        elif self.mode == self.MODE_SPECTRO: 
            p.setRenderHint(QPainter.RenderHint.Antialiasing, False) 
            self.draw_spectro(p, w, h)
            
        # Label
        labels = ["bars", "scope", "spectral"]
        p.setPen(QColor(160, 174, 192))
        p.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        p.drawText(rect.adjusted(10, 10, -15, -10), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop, labels[self.mode])

    def draw_bars(self, p, w, h):
        num_bars = len(self.displayed_bars)
        total_w = (w - 20) / num_bars
        bar_w = max(1, total_w - 2)
        
        for i, val in enumerate(self.displayed_bars):
            bar_h = min(h - 15, val * h * 2.2)
            x = 10 + (i * total_w)
            y = h - bar_h - 10
            
            c = QColor.fromHslF(self.hue, 0.65, 0.7)
            
            # --- FIX: Dynamic Opacity ---
            # Map the value (approx 0.0 to 1.0) to an Alpha range (60 to 255)
            # 60 = very transparent (quiet), 255 = solid (loud)
            alpha = int(min(255, max(60, 60 + (val * 220))))
            c.setAlpha(alpha)
            
            p.setBrush(c)
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRoundedRect(QRectF(x, y, bar_w, bar_h), 2, 2)

    def draw_scope(self, p, w, h):
        cy = h / 2
        data = self.raw_audio_buffer

        # Find the first point where the wave crosses from negative to positive.
        # This locks the visual in place.
        trigger_offset = 0
        for i in range(len(data) - 100): # Scan first portion
            if data[i] <= 0 and data[i+1] > 0:
                trigger_offset = i
                break
        
        # Start the view from the trigger point
        view = data[trigger_offset:]

        step = 2 
        view = view[::step]
        
        if len(view) < 2: return
        
        path = QPainterPath()

        amp_scale = 0.25
        
        path.moveTo(0, cy + (view[0] * h * amp_scale))
        x_step = w / len(view)
        
        for i, val in enumerate(view):
            path.lineTo(i * x_step, cy + (val * h * amp_scale))
            
        p.setPen(QPen(QColor.fromHslF(self.hue, 0.7, 0.6), 1.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)

    def draw_spectro(self, p, w, h):
        # Convert float buffer (0..1) to uint8 (0..255)
        # We perform a slight power curve here to darken silence (gamma correction)
        # Power of 1.5 makes low values lower (cleaner background)
        display_data = np.power(self.spec_buffer, 1.5)
        buf = (display_data * 255).astype(np.uint8)
        
        h_buf, w_buf = buf.shape
        
        # Create Indexed8 Image (Palette based)
        qimg = QImage(buf.data, w_buf, h_buf, w_buf, QImage.Format.Format_Indexed8)
        
        # Generate Color Table based on current Hue
        # 0 = Transparent, 255 = Full Color
        base_c = QColor.fromHslF(self.hue, 0.8, 0.5)
        r, g, b, _ = base_c.getRgb()
        
        # List comprehension is fast enough for 256 items
        # i is alpha
        color_table = [qRgba(r, g, b, int(i * 0.9)) for i in range(256)]
        qimg.setColorTable(color_table)
        
        # Draw image stretched to widget size
        p.drawImage(QRectF(0, 0, w, h), qimg)

class HueSlider(QSlider):
    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setRange(0, 1000)
        self.setFixedHeight(30) # Slightly taller to accommodate shadow
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        self.target_hover = 0.0
        self.hover_val = 0.0
        
        # Animation Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(16)

    def enterEvent(self, event):
        self.target_hover = 1.0
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.target_hover = 0.0
        super().leaveEvent(event)
        
    def animate(self):
        # Smoothly interpolate hover state
        diff = self.target_hover - self.hover_val
        if abs(diff) > 0.001:
            self.hover_val += diff * 0.1
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._update_val_from_pos(event.pos())
            event.accept()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._update_val_from_pos(event.pos())
            event.accept()
        super().mouseMoveEvent(event)

    def _update_val_from_pos(self, pos):
        rect = self.rect()
        margin = 10
        avail_w = rect.width() - (margin * 2)
        if avail_w <= 0: return
        click_x = max(margin, min(pos.x(), rect.width() - margin))
        norm = (click_x - margin) / avail_w
        self.setValue(int(norm * 1000))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        
        # Geometry
        margin = 10
        track_h = 4
        cy = rect.height() / 2
        track_rect = QRectF(margin, cy - track_h/2, rect.width() - (margin*2), track_h)
        
        # 1. Draw Track
        # Logic: Becomes slightly more colorful/opaque on hover
        opacity = 100 + (80 * self.hover_val)
        
        grad = QLinearGradient(track_rect.left(), 0, track_rect.right(), 0)
        for i in range(10):
            grad.setColorAt(i/9.0, QColor.fromHslF(i/9.0, 0.6, 0.8, opacity/255.0))
            
        painter.setBrush(grad)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(track_rect, 2, 2)
        
        # 2. Draw Handle
        val_norm = self.value() / 1000.0
        avail_w = rect.width() - (margin * 2)
        cx = margin + (avail_w * val_norm)
        
        radius = 6 + (1.0 * self.hover_val) # Slight size increase on hover
        
        # -- Neumorphic Shadow --
        # Draw a soft shadow below the handle
        shadow_r = radius + 4
        shadow_grad = QRadialGradient(cx, cy + 2, shadow_r)
        shadow_grad.setColorAt(0.0, QColor(0, 0, 0, 40))  # Center darkness
        shadow_grad.setColorAt(0.5, QColor(0, 0, 0, 20))  # Mid fade
        shadow_grad.setColorAt(1.0, QColor(0, 0, 0, 0))   # Edge transparent
        
        painter.setBrush(shadow_grad)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx, cy + 2), shadow_r, shadow_r)
        
        # -- Handle Body --
        painter.setBrush(QColor("white"))
        # Very subtle grey border for definition against white backgrounds
        painter.setPen(QPen(QColor(200, 200, 200), 1.0)) 
        painter.drawEllipse(QPointF(cx, cy), radius, radius)

class TapeOverlay(QWidget):
    def __init__(self, parent=None, wave_widget=None, reel_widget=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.wave = wave_widget
        self.reel = reel_widget
        self.anim_val = 0.0
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(16)
        
    def animate(self):
        target = 1.0 if self.wave.is_rec else 0.0
        diff = target - self.anim_val
        if abs(diff) > 0.001:
            self.anim_val += diff * 0.15 
            self.update()
            
    def paintEvent(self, event):
        # Stop drawing if practically invisible
        if self.anim_val < 0.01: return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 1. Coordinate Logic (Same as before)
        p1_global = self.wave.mapToGlobal(QPointF(self.wave.width(), self.wave.height()/2).toPoint())
        p1 = QPointF(self.mapFromGlobal(p1_global))
        
        reel_rect = self.reel.rect()
        cx, cy = reel_rect.width() / 2, reel_rect.height() / 2
        lx = cx - self.reel.offset_x
        ly = cy - self.reel.radius
        
        p2_global = self.reel.mapToGlobal(QPointF(lx, ly).toPoint())
        p2 = QPointF(self.mapFromGlobal(p2_global))
        
        current_end_x = p2.x() + (p1.x() - p2.x()) * self.anim_val
        current_end_y = p2.y() + (p1.y() - p2.y()) * self.anim_val
        current_end = QPointF(current_end_x, current_end_y)
        
        # 2. Colors & Morphing Logic
        reel_hue = getattr(self.reel, 'base_hue', 0.6)
        
        # Define the two states
        c_reel_base = QColor.fromHslF(reel_hue, 0.6, 0.75) # Green/Teal
        c_red_base = QColor(255, 80, 80)                   # Recording Red
        
        # INTERPOLATION: Mix Red and Green based on anim_val
        # anim_val 1.0 = Red, anim_val 0.0 = Green
        r = c_reel_base.red() + (c_red_base.red() - c_reel_base.red()) * self.anim_val
        g = c_reel_base.green() + (c_red_base.green() - c_reel_base.green()) * self.anim_val
        b = c_reel_base.blue() + (c_red_base.blue() - c_reel_base.blue()) * self.anim_val
        
        # This is the color of the "Tip" (Waveform side)
        c_tip_dynamic = QColor(int(r), int(g), int(b))
        
        # 3. Master Opacity (The "Fade Away")
        # Multiplier ensures it stays visible for the first 60% of retraction, 
        # then fades out rapidly in the last 40%.
        # Clamp between 0.0 and 1.0
        fade_factor = max(0.0, min(1.0, self.anim_val * 2.5))
        
        # Apply the fade factor to our Alpha channels
        c_start_transparent = QColor(c_reel_base)
        c_start_transparent.setAlphaF(0.0) # Always transparent start
        
        c_start_visible = QColor(c_reel_base)
        c_start_visible.setAlphaF(0.8 * fade_factor) # Reel side color
        
        c_end_visible = QColor(c_tip_dynamic)
        c_end_visible.setAlphaF(0.9 * fade_factor)   # Tip side color
        
        # 4. Gradient Construction
        grad = QLinearGradient(p2, current_end)
        grad.setColorAt(0.0, c_start_transparent)
        grad.setColorAt(0.3, c_start_visible) 
        grad.setColorAt(1.0, c_end_visible)
        
        # 5. Draw Path (Bezier)
        path = QPainterPath()
        path.moveTo(p2)
        
        slack_y = 60 * (1.0 - self.anim_val)
        c1 = QPointF(p2.x() - 30, p2.y() + slack_y)
        c2 = QPointF(current_end_x + 30, current_end_y + slack_y)
        path.cubicTo(c1, c2, current_end)
        
        pen = QPen(QBrush(grad), 2.8) 
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
        
        # 6. Draw Tip Dot
        # Use the dynamic color so the dot also turns green/fades
        painter.setBrush(c_end_visible)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(current_end, 3.0, 3.0)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("capta") 
        self.resize(600, 200) 
        
        # --- MULTIPROCESSING SETUP ---
        self.cmd_queue = multiprocessing.Queue()
        self.vis_queue = multiprocessing.Queue(maxsize=10)
        self.result_queue = multiprocessing.Queue()
        
        self.audio_process = multiprocessing.Process(
            target=audio_worker, 
            args=(self.cmd_queue, self.vis_queue, self.result_queue),
            daemon=True
        )
        self.audio_process.start()
        
        self.setup_ui()
        self.hue = 0.0
        
        # State
        self.is_armed = False # New state for "Wait..."
        
        # --- TAPE OVERLAY INTEGRATION ---
        self.tape_overlay = TapeOverlay(self.centralWidget(), self.wave_strip, self.logo)
        self.tape_overlay.resize(self.size())
        
        # UI Update Timer (Logic)
        self.viz_timer = QTimer(self)
        self.viz_timer.timeout.connect(self.poll_audio_data)
        self.viz_timer.start(5) 
        
        # Result Polling Timer
        self.res_timer = QTimer(self)
        self.res_timer.timeout.connect(self.check_results)
        self.res_timer.start(100)

    # ... (setup_ui remains the same) ...
    # Ensure setup_ui is inside the class (omitted here for brevity if not changed)
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        central.setStyleSheet("background-color: #f7fafc;")
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(15, 10, 15, 10)
        
        viz_container = QWidget()
        viz_layout = QVBoxLayout(viz_container)
        viz_layout.setContentsMargins(0, 0, 15, 0)
        viz_layout.setSpacing(8)
        
        self.wave_strip = WaveformStrip()
        self.wave_strip.setMaximumHeight(60)
        viz_layout.addWidget(self.wave_strip)
        
        self.vis_panel = VisualizerPanel()
        viz_layout.addWidget(self.vis_panel, 1)
        
        self.hue_slider = HueSlider()
        self.hue_slider.valueChanged.connect(self.update_hue)
        viz_layout.addWidget(self.hue_slider)
        main_layout.addWidget(viz_container, 1) 
        
        ctrl_container = QWidget()
        ctrl_container.setFixedWidth(90)
        ctrl_layout = QVBoxLayout(ctrl_container)
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(5)
        
        self.logo = CaptaLogo()
        h_logo = QHBoxLayout()
        h_logo.addStretch(); h_logo.addWidget(self.logo); h_logo.addStretch()
        ctrl_layout.addLayout(h_logo)
        ctrl_layout.addStretch()
        
        self.btn_record = CaptaButton()
        self.btn_record.clicked.connect(self.toggle_recording)
        
        h_center = QHBoxLayout()
        h_center.addStretch(); h_center.addWidget(self.btn_record); h_center.addStretch()
        ctrl_layout.addLayout(h_center)
        ctrl_layout.addStretch()
        
        self.lbl_status = QLabel("ready")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet("color: #cbd5e0; font-family: 'Segoe UI'; font-size: 11px; font-weight: bold;")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setFixedHeight(32) 
        ctrl_layout.addWidget(self.lbl_status)
        main_layout.addWidget(ctrl_container, 0)

    def poll_audio_data(self):
        try:
            last_data = None
            while not self.vis_queue.empty():
                last_data = self.vis_queue.get_nowait()
            
            if last_data is not None:
                # 1. Update Visuals
                self.vis_panel.update_audio(last_data)
                
                mono_data = np.mean(last_data, axis=1) if last_data.ndim > 1 else last_data
                
                # Update Waveform (pass armed state for visualization if needed, or just rec)
                self.wave_strip.push_data(mono_data, self.btn_record.is_recording)

                # 2. Logic: Audio Trigger
                if self.is_armed:
                    # Check peak amplitude
                    peak = np.max(np.abs(mono_data))
                    # Threshold: 1.5% volume triggers recording
                    if peak > 0.015: 
                        self.start_recording_now()

        except queue.Empty:
            pass
            
    def update_hue(self, val):
        self.hue = val / 1000.0
        self.wave_strip.set_hue(self.hue)
        self.vis_panel.set_hue(self.hue)
        # Force logo update to propagate hue to tape overlay
        self.logo.base_hue = self.hue 

    def toggle_recording(self):
        if self.btn_record.is_recording:
            # STOP Recording
            self.stop_recording()
        elif self.is_armed:
            # CANCEL Arming
            self.is_armed = False
            self.lbl_status.setText("ready")
            self.lbl_status.setStyleSheet("color: #cbd5e0; font-family: 'Segoe UI'; font-size: 11px; font-weight: bold;")
            self.btn_record.set_recording(False) # Ensure visual state is off
        else:
            # START Arming (Wait for audio)
            self.is_armed = True
            self.lbl_status.setText("wait...")
            # Yellow for wait state
            self.lbl_status.setStyleSheet("color: #ecc94b; font-family: 'Segoe UI'; font-size: 11px; font-weight: bold;")

    def start_recording_now(self):
        self.is_armed = False
        self.cmd_queue.put(('START', None))
        self.btn_record.set_recording(True)
        self.lbl_status.setText("recording")
        self.lbl_status.setStyleSheet("color: #fc8181; font-family: 'Segoe UI'; font-size: 11px; font-weight: bold;")

    def stop_recording(self):
        self.is_armed = False
        self.cmd_queue.put(('STOP', None))
        self.btn_record.set_recording(False)
        self.lbl_status.setText("processing...")
        self.lbl_status.setStyleSheet("color: #63b3ed; font-family: 'Segoe UI'; font-size: 11px; font-weight: bold;")

    def check_results(self):
        try:
            while not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                if result is None: 
                    # Capture was cancelled or empty
                    self.lbl_status.setText("ready")
                    self.lbl_status.setStyleSheet("color: #cbd5e0; font-family: 'Segoe UI'; font-size: 11px; font-weight: bold;")
                    return
                data, sr = result 
                self.handle_recording_finished(data, sr)
        except queue.Empty:
            pass

    def handle_recording_finished(self, full_buffer, sr):
        self.lbl_status.setText("saving...")
        self.saver = SaveWorker(full_buffer, sr)
        self.saver.finished_saving.connect(self.on_save_finished)
        self.saver.start()

    def on_save_finished(self, success, msg):
        if success:
            self.lbl_status.setText(msg)
            self.lbl_status.setStyleSheet("color: #68d391; font-family: 'Segoe UI'; font-size: 11px; font-weight: bold;")
            QTimer.singleShot(2000, lambda: self.lbl_status.setText("ready"))
            QTimer.singleShot(2000, lambda: self.lbl_status.setStyleSheet("color: #cbd5e0; font-family: 'Segoe UI'; font-size: 11px; font-weight: bold;"))
        else:
            self.lbl_status.setText(msg)
            self.lbl_status.setStyleSheet("color: #f56565; font-family: 'Segoe UI'; font-size: 11px; font-weight: bold;")

    def closeEvent(self, event):
        self.cmd_queue.put(('EXIT', None))
        time.sleep(0.1)
        self.audio_process.terminate()
        event.accept()
    
    def resizeEvent(self, event):
        if hasattr(self, 'tape_overlay'):
            self.tape_overlay.resize(self.size())
        super().resizeEvent(event)

if __name__ == "__main__":
    # REQUIRED for Multiprocessing on Windows
    multiprocessing.freeze_support() 
    
    try:
        myappid = 'capta.audio.tool.v1'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass

    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "capta.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
        
    window = MainWindow()
    window.show()
    sys.exit(app.exec())