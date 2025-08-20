import numpy as np
import sounddevice as sd
import mido
import threading
import string
from dearpygui.dearpygui import *
import json
from tkinter import Tk
from tkinter.filedialog import askdirectory
from tkinter.filedialog import askopenfilename
import os

# Import key mappings dari keyboard
from dearpygui.dearpygui import (
    mvKey_Z, mvKey_X, mvKey_C, mvKey_V, mvKey_B, mvKey_N, mvKey_M,
    mvKey_S, mvKey_D, mvKey_G, mvKey_H, mvKey_J, mvKey_L, mvKey_K,
    mvKey_Q, mvKey_W, mvKey_E, mvKey_R, mvKey_T, mvKey_Y, mvKey_U,
    mvKey_I, mvKey_O, mvKey_P, mvKey_Comma, mvKey_Period, mvKey_Slash,
    mvKey_2, mvKey_3, mvKey_5, mvKey_6, mvKey_7, mvKey_9, mvKey_0
)

# ------------------ Global Configuration ------------------
sample_rate = 48000
volume = 0.5
unison_voices = 1
detune_cents = 10
blend_strength = 0.5
slide_enabled = False
always_slide = False
slide_speed = 5.0
current_freq = None
target_freq = None
last_release_time = None
current_octave_osc1 = 0
current_octave_osc2 = 0
volume_osc1 = 1
volume_osc2 = 1
fine_mod_osc1 = 1
fine_mod_osc2 = 1


# Envelope times (seconds)
attack_time = 0.00
hold_time = 0.0
decay_time = 0.1
sustain_level = 0.7
release_time = 0.2

# State Management
note_current_gain = {}
note_start_time = {}
note_release_time = {}
note_phase = {}
pressed_keys = set()
pressed_midi_notes = set()
active_freqs = set()
current_phase = 0.0
is_mono = False
global_t = None

presets = {}
folder_path = None
root_presets_path = os.path.join(os.getcwd(), "Presets")


# Tambahkan global ID
# Tambahkan di global configuration section
input_id = None



# Key mapping (keyboard -> frekuensi)
key_to_freq = {
    'z': 261.63, 's': 277.18, 'x': 293.66, 'd': 311.13, 'c': 329.63,
    'v': 349.23, 'g': 369.99, 'b': 392.00, 'h': 415.30, 'n': 440.00,
    'j': 466.16, 'm': 493.88, ',': 523.25, 'l': 554.37, '.': 587.33,
    ';': 622.25, '/': 659.25, 'q': 523.25, '2': 554.37, 'w': 587.33,
    '3': 622.25, 'e': 659.25, 'r': 698.46, '5': 739.99, 't': 783.99,
    '6': 830.61, 'y': 880.00, '7': 932.33, 'u': 987.77, 'i': 1046.50,
    '9': 1108.73, 'o': 1174.66, '0': 1244.51, 'p': 1318.51
}

key_lookup = {
    mvKey_Z: 'z', mvKey_X: 'x', mvKey_C: 'c', mvKey_V: 'v', mvKey_B: 'b', mvKey_N: 'n', mvKey_M: 'm',
    mvKey_S: 's', mvKey_D: 'd', mvKey_G: 'g', mvKey_H: 'h', mvKey_J: 'j', mvKey_L: 'l',
    mvKey_Q: 'q', mvKey_W: 'w', mvKey_E: 'e', mvKey_R: 'r', mvKey_T: 't', mvKey_Y: 'y', mvKey_U: 'u',
    mvKey_I: 'i', mvKey_O: 'o', mvKey_P: 'p',
    mvKey_Comma: ',', mvKey_Period: '.', mvKey_Slash: '/',
    mvKey_2: '2', mvKey_3: '3', mvKey_5: '5', mvKey_6: '6', mvKey_7: '7', mvKey_9: '9', mvKey_0: '0'
}

def midi_to_freq(note):
    return 440.0 * (2 ** ((note - 69) / 12.0))

def sine_wave(freq, t): return np.sin(2 * np.pi * freq * t)
def square_wave(freq, t): return np.sign(np.sin(2 * np.pi * freq * t))
def triangle_wave(freq, t): return 2 * np.arcsin(np.sin(2 * np.pi * freq * t)) / np.pi
def sawtooth_wave(freq, t): return 2 * (t * freq - np.floor(0.5 + t * freq))

wave_types = {
    'Sine': sine_wave,
    'Square': square_wave,
    'Triangle': triangle_wave,
    'Sawtooth': sawtooth_wave
}

osc2_enabled = False
osc1_wave = 'Sine'
osc2_wave = 'Sine'

osc1_unison = {'voices': 1, 'detune': 10.0, 'blend': 0.5}
osc2_unison = {'voices': 1, 'detune': 10.0, 'blend': 0.5}

def envelope(time, phase, attack, decay, sustain, release, starting_gain=1.0):
    if phase == 'attack': return min(1.0, time / attack)
    elif phase == 'hold': return 1.0
    elif phase == 'decay': return max(sustain, 1.0 - (time / decay) * (1.0 - sustain))
    elif phase == 'sustain': return sustain
    elif phase == 'release': return max(0.0, starting_gain * (1.0 - (time / release)))
    return 0.0

def generate_detuned_frequencies(base_freq, num_voices, detune_cents):
    center = (num_voices - 1) / 2
    return [base_freq * 2 ** (((i - center) * detune_cents) / 1200) for i in range(num_voices)]

def audio_callback(outdata, frames, time, status):
    global current_freq, current_phase, global_t

    waveform = np.zeros(frames)

    # Mono glide (slide)
    if is_mono and (slide_enabled or always_slide) and current_freq is not None and target_freq is not None:
        diff = target_freq - current_freq
        max_step = (frames / sample_rate) / slide_speed * abs(diff)
        current_freq += np.sign(diff) * min(abs(diff), max_step)

    for freq in list(active_freqs):
        if freq not in note_start_time:
            note_start_time[freq] = current_phase / sample_rate

        t_start = current_phase / sample_rate
        elapsed = t_start - note_start_time[freq]

        if freq in note_release_time:
            phase = 'release'
            time_in_phase = t_start - note_release_time[freq]
            if time_in_phase >= release_time:
                active_freqs.remove(freq)
                note_start_time.pop(freq, None)
                note_release_time.pop(freq, None)
                note_current_gain.pop(freq, None)
                continue
            starting_gain = note_current_gain.get(freq, sustain_level)
        else:
            if elapsed < attack_time:
                phase, time_in_phase = 'attack', elapsed
            elif elapsed < attack_time + hold_time:
                phase, time_in_phase = 'hold', elapsed - attack_time
            elif elapsed < attack_time + hold_time + decay_time:
                phase, time_in_phase = 'decay', elapsed - attack_time - hold_time
            else:
                phase, time_in_phase = 'sustain', elapsed - attack_time - hold_time - decay_time
            starting_gain = 1.0

        gain = envelope(time_in_phase, phase, attack_time, decay_time, sustain_level, release_time, starting_gain)

        # ===== OSCILLATOR 1 =====
        osc1_freq = freq * (2 ** current_octave_osc1) * fine_mod_osc1
        detuned1 = generate_detuned_frequencies(osc1_freq, unison_voices, detune_cents)
        center1 = (unison_voices - 1) / 2

        for i, dfreq in enumerate(detuned1):
            weight1 = 1.0 - (blend_strength * abs(i - center1) / center1) if center1 != 0 else 1.0
            weight1 = max(weight1, 0.0)
            
            # Perhitungan frekuensi efektif dengan memperhatikan octave shift dalam mode mono
            if is_mono and (slide_enabled or always_slide) and current_freq is not None:
                # Pertahankan rasio detuning relatif terhadap frekuensi dasar yang sudah di-shift octave
                effective_freq = current_freq * (dfreq / osc1_freq)
            else:
                effective_freq = dfreq
                
            phase_val = note_phase.get((freq, dfreq, 'osc1'), 0.0)
            local_t = (np.arange(frames) / sample_rate) + (phase_val / (2 * np.pi * effective_freq))
            global_t = local_t
            waveform += (gain * volume_osc1 * 0.4 * weight1 / (np.sqrt(unison_voices) / 1.5)) * wave_types[osc1_wave](effective_freq, local_t)
            note_phase[(freq, dfreq, 'osc1')] = (phase_val + 2 * np.pi * effective_freq * frames / sample_rate) % (2 * np.pi)

        # ===== OSCILLATOR 2 =====
        if osc2_enabled:
            osc2_freq = freq * (2 ** current_octave_osc2) * fine_mod_osc2
            detuned2 = generate_detuned_frequencies(osc2_freq, osc2_unison['voices'], osc2_unison['detune'])
            center2 = (osc2_unison['voices'] - 1) / 2

            for i, dfreq in enumerate(detuned2):
                weight2 = 1.0 - (osc2_unison['blend'] * abs(i - center2) / center2) if center2 != 0 else 1.0
                weight2 = max(weight2, 0.0)
                
                # Perhitungan serupa untuk OSC2
                if is_mono and (slide_enabled or always_slide) and current_freq is not None:
                    effective_freq = current_freq * (dfreq / osc2_freq)
                else:
                    effective_freq = dfreq
                    
                phase_val = note_phase.get((freq, dfreq, 'osc2'), 0.0)
                local_t = (np.arange(frames) / sample_rate) + (phase_val / (2 * np.pi * effective_freq))
                waveform += (gain * volume_osc2 * 0.4 * weight2 / (np.sqrt(osc2_unison['voices']) / 1.5)) * wave_types[osc2_wave](effective_freq, local_t)
                note_phase[(freq, dfreq, 'osc2')] = (phase_val + 2 * np.pi * effective_freq * frames / sample_rate) % (2 * np.pi)

    outdata[:, 0] = volume * waveform
    current_phase += frames



def start_note(freq):
    global current_freq, target_freq, last_release_time

    if is_mono:
        is_overlapping = len(pressed_keys | pressed_midi_notes) > 1
        if (slide_enabled and is_overlapping) or always_slide:
            if current_freq is None:
                current_freq = freq * (2 ** current_octave_osc1)  # Terapkan octave shift
            target_freq = freq * (2 ** current_octave_osc1)  # Terapkan octave shift
        else:
            current_freq = freq * (2 ** current_octave_osc1)  # Terapkan octave shift
            target_freq = freq * (2 ** current_octave_osc1)  # Terapkan octave shift
        active_freqs.clear()
    else:
        if len(active_freqs) >= 6: return

    active_freqs.add(freq)
    note_start_time[freq] = current_phase / sample_rate
    note_phase[freq] = 0.0
    note_release_time.pop(freq, None)

def stop_note(freq):
    global last_release_time
    if freq in active_freqs:
        if release_time <= 0.0:
            active_freqs.remove(freq)
            note_start_time.pop(freq, None)
            note_release_time.pop(freq, None)
            note_current_gain.pop(freq, None)
        else:
            elapsed = (current_phase / sample_rate) - note_start_time[freq]
            if elapsed < attack_time:
                phase, time_in_phase = 'attack', elapsed
            elif elapsed < attack_time + hold_time:
                phase, time_in_phase = 'hold', elapsed - attack_time
            elif elapsed < attack_time + hold_time + decay_time:
                phase, time_in_phase = 'decay', elapsed - attack_time - hold_time
            else:
                phase, time_in_phase = 'sustain', elapsed - attack_time - hold_time - decay_time

            note_current_gain[freq] = envelope(time_in_phase, phase, attack_time, decay_time, sustain_level, release_time)
            note_release_time[freq] = current_phase / sample_rate
        last_release_time = current_phase / sample_rate




def generate_wave_from_text(sender, text):
    global wave_types, osc1_wave, current_octave_osc1

    if not text:
        print("Input teks kosong, tidak dapat menghasilkan gelombang.")
        return

    raw_vals = [ord(c) for c in text]
    wave_cycle = np.array(raw_vals, dtype=np.float32)
    wave_cycle -= np.min(wave_cycle)
    wave_cycle /= np.max(wave_cycle) if np.max(wave_cycle) > 0 else 1
    wave_cycle = (wave_cycle * 2.0) - 1.0

    resolution = 512
    interpolated_wave = np.interp(
        np.linspace(0, len(wave_cycle), resolution, endpoint=False),
        np.arange(len(wave_cycle)),
        wave_cycle
    )

    def custom_wave(freq, t):
        # Terapkan octave shift untuk OSC1
        adjusted_freq = freq * (2 ** current_octave_osc1)
        phase = (adjusted_freq * t) % 1.0
        idx = (phase * resolution).astype(int)
        return interpolated_wave[idx % resolution]

    wave_types['Custom'] = custom_wave
    osc1_wave = 'Custom'
    plot_waveform('Custom', "wave_canvas_osc1")
    print(f"Generated Custom wave from text: '{text}' with octave shift {current_octave_osc1}")


def generate_wave_from_text_osc2(sender, text):
    global wave_types, osc2_wave

    if not text:
        print("Teks kosong.")
        return

    raw_vals = [ord(c) for c in text]
    wave_cycle = np.array(raw_vals, dtype=np.float32)
    wave_cycle -= np.min(wave_cycle)
    wave_cycle /= np.max(wave_cycle) if np.max(wave_cycle) > 0 else 1
    wave_cycle = (wave_cycle * 2.0) - 1.0

    resolution = 512
    interpolated_wave = np.interp(
        np.linspace(0, len(wave_cycle), resolution, endpoint=False),
        np.arange(len(wave_cycle)),
        wave_cycle
    )

    def custom_wave(freq, t):
        adjusted_freq = freq * (2 ** current_octave_osc2)
        phase = (adjusted_freq * t) % 1.0
        idx = (phase * resolution).astype(int)
        return interpolated_wave[idx % resolution]

    wave_types['CustomOsc2'] = custom_wave
    osc2_wave = 'CustomOsc2'
    plot_waveform('CustomOsc2', "wave_canvas_osc2")
    print(f"[OSC2] Generated custom wave from text: '{text}' with octave shift {current_octave_osc2}")




def midi_listener():
    try:
        inputs = mido.get_input_names()
        if not inputs:
            print("Tidak ada perangkat MIDI ditemukan.")
            return
        print(f"Menggunakan MIDI input: {inputs[0]}")
        with mido.open_input(inputs[0]) as port:
            for msg in port:
                if msg.type == 'note_on' and msg.velocity > 0:
                    pressed_midi_notes.add(msg.note)
                    start_note(midi_to_freq(msg.note))
                elif msg.type in ('note_off', 'note_on') and msg.velocity == 0:
                    pressed_midi_notes.discard(msg.note)
                    stop_note(midi_to_freq(msg.note))
    except Exception as e:
        print(f"MIDI error: {e}")


# ------------------ Parameter Callbacks ------------------
def update_attack(s, a):
    global attack_time
    attack_time = a

def update_decay(s, a):
    global decay_time
    decay_time = a

def update_sustain(s, a):
    global sustain_level
    sustain_level = a

def update_release(s, a):
    global release_time
    release_time = a

def update_hold(s, a):
    global hold_time
    hold_time = a

def update_wave(s, a):
    global osc1_wave
    osc1_wave = a
    plot_waveform(a, "wave_canvas_osc1")

def update_osc2_wave(s, a):
    global osc2_wave
    osc2_wave = a
    plot_waveform(a, "wave_canvas_osc2")

def toggle_mono_mode(s, a):
    global is_mono
    is_mono = a

def set_unison_voices(a):
    global unison_voices
    unison_voices = a

def set_detune_cents(a):
    global detune_cents
    detune_cents = a

def set_blend_strength(a):
    global blend_strength
    blend_strength = a

def set_osc2_unison_voices(a):
    osc2_unison['voices'] = a

def set_osc2_detune_cents(a):
    osc2_unison['detune'] = a

def set_osc2_blend_strength(a):
    osc2_unison['blend'] = a


def set_slide_enabled(a):
    global slide_enabled, always_slide
    slide_enabled = a
    if not a:
        always_slide = False

def set_always_slide(a):
    global always_slide
    always_slide = a if slide_enabled else False

def set_slide_speed(a):
    global slide_speed
    slide_speed = a / 1000.0

def toggle_osc2(s, a):
    global osc2_enabled
    osc2_enabled = a

def plot_waveform(wave_name, canvas_tag):
    if wave_name not in wave_types:
        print(f"Waveform {wave_name} tidak ditemukan.")
        return

    t = np.linspace(0, 1 / 440.0, 512)
    y = wave_types[wave_name](440.0, t)

    delete_item(canvas_tag, children_only=True)

    width, height = 250, 80
    margin = 5
    for i in range(len(t)-1):
        x1 = int((i / len(t)) * width)
        y1 = int((0.5 - y[i]/2) * (height - 2 * margin)) + margin
        x2 = int(((i+1) / len(t)) * width)
        y2 = int((0.5 - y[i+1]/2) * (height - 2 * margin)) + margin
        draw_line(parent=canvas_tag, p1=[x1, y1], p2=[x2, y2], color=(0, 255, 0, 255), thickness=1)

def on_octave_change(sender, value):
    print(f"Octave: {value}")

def update_octave_osc1(s, a):
    global current_octave_osc1
    current_octave_osc1 = a
    print(current_octave_osc1)

def update_octave_osc2(s, a):
    global current_octave_osc2
    current_octave_osc2 = a
    print(f"[OSC2] Octave set to {a}")

def on_key_press(sender, app_data):
    if is_item_focused(input_id) or is_item_focused(input_id2):  # Jangan trigger note kalau sedang ngetik
        return

    key = key_lookup.get(app_data)
    if key and key not in pressed_keys and key in key_to_freq:
        pressed_keys.add(key)
        start_note(key_to_freq[key])


def on_key_release(sender, app_data):
    if is_item_focused(input_id) or is_item_focused(input_id2):
        return

    key = key_lookup.get(app_data)
    if key and key in pressed_keys and key in key_to_freq:
        pressed_keys.remove(key)
        stop_note(key_to_freq[key])

def update_volume_osc1(s, a):
    global volume_osc1
    volume_osc1 = a / 100

def update_volume_osc2(s, a):
    global volume_osc2
    volume_osc2 = a / 100

def update_fine_osc1(s, a):
    global fine_mod_osc1
    # Convert range 0-200 -> -100 to +100 cent (1 semitone up/down)
    fine_mod_osc1 = 2 ** ((a - 100) / 1200)  # 1200 cent per octave

def update_fine_osc2(s, a):
    global fine_mod_osc2
    # Convert range 0-200 -> -100 to +100 cent (1 semitone up/down)
    fine_mod_osc2 = 2 ** ((a - 100) / 1200)  # 1200 cent per octave

def save_preset(s, preset_name):
    global presets, folder_path
    presets = {
        "wave_osc1": get_value("wave_osc1"),
        "octave_osc1": get_value("octave_osc1"),
        "volume_osc1": get_value("volume_osc1"),
        "fine_osc1": get_value("fine_osc1"),
        "osc2_enable": get_value("osc2_enable"),
        "wave_osc2": get_value("wave_osc2"),
        "octave_osc2": get_value("octave_osc2"),
        "volume_osc2": get_value("volume_osc2"),
        "fine_osc2": get_value("fine_osc2"),
        "unison_osc1": get_value("unison_osc1"),
        "detune_osc1": get_value("detune_osc1"),
        "blend_osc1": get_value("blend_osc1"),
        "unison_osc2": get_value("unison_osc2"),
        "detune_osc2": get_value("detune_osc2"),
        "blend_osc2": get_value("blend_osc2"),
        "attack": get_value("attack"),
        "hold": get_value("hold"),
        "decay": get_value("decay"),
        "sustain": get_value("sustain"),
        "release": get_value("release"),
        "mono": get_value("mono"),
        "porta": get_value("porta"),
        "always": get_value("always"),
        "time": get_value("time"),
    }

    saving_preset = presets

    folder_path = select_folder()

    preset_name_with_extension = preset_name + ".lnp"
    preset_path = os.path.join(folder_path, preset_name_with_extension)
    with open(preset_path, "w") as f:
        json.dump(saving_preset, f, indent=4)
    
    print("Preset saved")

def load_preset(input_path):
    try:
        if input_path is None:
            input_path = select_file()
            if not input_path:  # User membatalkan dialog
                return
                
        with open(input_path, "r") as f:
            presets = json.load(f)

        set_value("wave_osc1", presets["wave_osc1"])
        set_value("octave_osc1", presets["octave_osc1"])
        set_value("volume_osc1", presets["volume_osc1"])
        set_value("fine_osc1", presets["fine_osc1"])
        set_value("osc2_enable", presets["osc2_enable"])
        set_value("wave_osc2", presets["wave_osc2"])
        set_value("octave_osc2", presets["octave_osc2"])
        set_value("volume_osc2", presets["volume_osc2"])
        set_value("fine_osc2", presets["fine_osc2"])
        set_value("unison_osc1", presets["unison_osc1"])
        set_value("detune_osc1", presets["detune_osc1"])
        set_value("blend_osc1", presets["blend_osc1"])
        set_value("unison_osc2", presets["unison_osc2"])
        set_value("detune_osc2", presets["detune_osc2"])
        set_value("blend_osc2", presets["blend_osc2"])
        set_value("attack", presets["attack"])
        set_value("hold", presets["hold"])
        set_value("decay", presets["decay"])
        set_value("sustain", presets["sustain"])
        set_value("release", presets["release"])
        set_value("mono", presets["mono"])
        set_value("porta", presets["porta"])
        set_value("always", presets["always"])
        set_value("time", presets["time"])

        update_wave(None, presets["wave_osc1"])
        update_octave_osc1(None, presets["octave_osc1"])
        update_volume_osc1(None, presets["volume_osc1"])
        update_fine_osc1(None, presets["fine_osc1"])
        toggle_osc2(None, presets["osc2_enable"])
        update_osc2_wave(None, presets["wave_osc2"])
        update_octave_osc2(None, presets["octave_osc2"])
        update_volume_osc2(None, presets["volume_osc2"])
        update_fine_osc2(None, presets["fine_osc2"])
        set_unison_voices(presets["unison_osc1"])
        set_detune_cents(presets["detune_osc1"])
        set_blend_strength(presets["blend_osc1"])
        set_osc2_unison_voices(presets["unison_osc2"])
        set_osc2_detune_cents(presets["detune_osc2"])
        set_osc2_blend_strength(presets["blend_osc2"])
        update_attack(None, presets["attack"])
        update_hold(None, presets["hold"])
        update_decay(None, presets["decay"])
        update_sustain(None, presets["sustain"])
        update_release(None, presets["release"])
        toggle_mono_mode(None, presets["mono"])
        set_slide_enabled(presets["porta"])
        set_always_slide(presets["always"])
        set_slide_speed(presets["time"])

        print("Preset loaded")
            
    except Exception as e:
        print(f"Error loading preset: {e}")
    



def select_folder():
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    folder_path = askdirectory(title="Choose folder to save the preset")
    root.attributes('-topmost', False)
    root.destroy()
    return folder_path

def select_file():
    root = Tk()
    root.withdraw()  # Menyembunyikan jendela utama Tkinter
    root.attributes('-topmost', True)  # Menjadikan dialog tetap di atas
    file_path = askopenfilename(title="Choose a preset")
    root.attributes('-topmost', False)
    root.destroy()
    return file_path

def get_all_presets(folder_path):
    presets_dict = {}
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.join(root, file)
            display_name = get_display_name(full_path)
            presets_dict[full_path] = display_name
    return presets_dict



def get_display_name(filepath):
    # Ambil nama file dengan ekstensi
    filename_with_ext = os.path.basename(filepath)
    # Hilangkan ekstensi
    filename, _ = os.path.splitext(filename_with_ext)
    return filename

list_of_presets = get_all_presets(root_presets_path)

# ------------------ GUI ------------------
create_context()
with window(label="", width=1000, height=800, no_title_bar=True, no_collapse=True, no_move=True, no_resize=True):
    add_text("Leemonn.", wrap=300)
    with group (horizontal=True):
        add_combo(
            width= 200,
            items=list(list_of_presets.values()),
            label="Presets",
            callback=lambda s, a: load_preset(list(list_of_presets.keys())[list(list_of_presets.values()).index(a)])
        )
        add_button(label="Browse", callback=lambda: load_preset(None))
        preset_name = add_input_text(width=200, label="Preset's name")
        add_button(label="Save Preset", callback=lambda: save_preset(None, get_value(preset_name)))

    add_separator()
    with group(horizontal=True):
        with group(width= 360):
            add_text("Oscilator 1")
            add_combo(list(wave_types.keys()), label="Wave", default_value="Sine", callback=update_wave, tag="wave_osc1")
            add_separator()
            input_id = add_input_text(label="Text 2 Wave", callback=generate_wave_from_text, on_enter=True)
            add_button(label="Generate", callback=lambda: generate_wave_from_text(None, get_value(input_id)))
            add_separator()
            with child_window(border=True, height=140, width=310):
                add_slider_int(label="Octave", min_value=-2, max_value=2, default_value=0, width=100, callback=lambda s, a: update_octave_osc1(s, a), tag="octave_osc1")
                add_drawlist(tag="wave_canvas_osc1", width=300, height=100)
            with group(horizontal=True):
                add_knob_float(label="Volume", min_value=0, max_value=100, default_value=100, callback=lambda s, a: update_volume_osc1(s, a), tag="volume_osc1")
                add_knob_float(label="Fine Tune (Cent)", min_value=0, max_value=200, default_value=100, callback=lambda s, a: update_fine_osc1(s, a), tag="fine_osc1")

        add_spacer(width=10)
        with group(width=360):
            add_checkbox(label="Oscilator 2", callback=toggle_osc2, tag="osc2_enable")
            add_combo(list(wave_types.keys()), label="Wave", default_value="Sine", callback=update_osc2_wave, tag="wave_osc2")
            add_separator()
            input_id2 = add_input_text(label="Text 2 Wave", callback=None, on_enter=True)
            add_button(label="Generate", callback=lambda: generate_wave_from_text_osc2(None, get_value(input_id2)))
            add_separator()
            with child_window(border=True, height=140):
                add_slider_int(label="Octave", min_value=-2, max_value=2, default_value=0, width=100, callback=lambda s, a: update_octave_osc2(s, a), tag="octave_osc2")
                add_drawlist(tag="wave_canvas_osc2", width=300, height=100)
            with group(horizontal=True):
                add_knob_float(label="Volume", min_value=0, max_value=100, default_value=100, callback=lambda s, a: update_volume_osc2(s, a), tag="volume_osc2")
                add_knob_float(label="Fine Tune", min_value=0, max_value=200, default_value=100, callback=lambda s, a: update_fine_osc2(s, a), tag="fine_osc2")

    add_separator()
    with group(horizontal=True):
        with child_window(width=400, height=150, border=False):
            with group(horizontal=True):
                with group(width=400):
                    add_text("Unison")
                    add_slider_int(tag="unison_osc1", label="", min_value=1, max_value=12, default_value=1, width=150, callback=lambda s, a: set_unison_voices(a))
                    add_text("Detune")
                    add_slider_float(tag="detune_osc1", label="", min_value=0.1, max_value=50.0, default_value=10.0, width=150, callback=lambda s, a: set_detune_cents(a))
                    add_text("Blend")
                    add_slider_float(tag="blend_osc1", label="", min_value=0.0, max_value=1.0, default_value=0.5, width=150, callback=lambda s, a: set_blend_strength(a)) 
        add_spacer(width=65)
        with child_window(width=400, height=150,border=False):
            with group(horizontal=True):
                with group(width=400):
                    add_text("Unison")
                    add_slider_int(tag="unison_osc2", label="", min_value=1, max_value=12, default_value=1, width=150, callback=lambda s, a: set_osc2_unison_voices(a))
                    add_text("Detune")
                    add_slider_float(tag="detune_osc2", label="", min_value=0.1, max_value=50.0, default_value=10.0, width=150, callback=lambda s, a: set_osc2_detune_cents(a))
                    add_text("Blend")
                    add_slider_float(tag="blend_osc2", label="", min_value=0.0, max_value=1.0, default_value=0.5, width=150, callback=lambda s, a: set_osc2_blend_strength(a))     
    with group(horizontal=True):
        with child_window(width=600, height=100):
            add_text("Envelope")
            with group(horizontal=True):
                add_knob_float(tag="attack", label="Attack", min_value=0.00, max_value=2.0, default_value=attack_time, callback=update_attack)
                add_spacer(width=20)
                add_knob_float(tag="hold", label="Hold", min_value=0.0, max_value=2.0, default_value=hold_time, callback=update_hold)
                add_spacer(width=20)
                add_knob_float(tag="decay", label="Decay", min_value=0.01, max_value=2.0, default_value=decay_time, callback=update_decay)
                add_spacer(width=20)
                add_knob_float(tag="sustain", label="Sustain", min_value=0.0, max_value=1.0, default_value=sustain_level, callback=update_sustain)
                add_spacer(width=20)
                add_knob_float(tag="release", label="Release", min_value=0.01, max_value=2.0, default_value=release_time, callback=update_release)
        with child_window(width=350, height=180):
            with group(horizontal=False):
                add_checkbox(tag="mono", label="Mono", callback=toggle_mono_mode)
                add_separator()
                add_text("Slide Control")
                add_checkbox(tag="porta", label="Porta", callback=lambda s, a: set_slide_enabled(a))
                add_checkbox(tag="always", label="Always", callback=lambda s, a: set_always_slide(a))
                add_knob_float(tag="time", label="Time (ms)", min_value=5.0, max_value=200.0, default_value=5.0, callback=lambda s, a: set_slide_speed(a))



create_viewport(title="Leemonn Wavetable Synth", width=1000, height=800, resizable=False)
setup_dearpygui()
show_viewport()

plot_waveform(osc1_wave, "wave_canvas_osc1")
plot_waveform(osc2_wave, "wave_canvas_osc2")


with handler_registry():
    add_key_press_handler(callback=on_key_press)
    add_key_release_handler(callback=on_key_release)

threading.Thread(target=midi_listener, daemon=True).start()

with sd.OutputStream(
    samplerate=sample_rate,
    channels=1,
    callback=audio_callback,
    blocksize=128,
    latency='low'
):
    start_dearpygui()

destroy_context()