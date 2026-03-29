import sys, os, random
from dotenv import load_dotenv
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSlider, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui  import QPainter, QColor, QPen, QBrush, QFont, QRadialGradient

load_dotenv()

INPUT_SIZE  = int(os.getenv("INPUT_SIZE",  3))
HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE", 5))
OUTPUT_SIZE = int(os.getenv("OUTPUT_SIZE", 2))
THRESHOLD   = float(os.getenv("THRESHOLD", 0.5))
LEAK        = float(os.getenv("LEAK",      0.9))

ANIM_STEPS  = 40    # frames per layer-to-layer hop
TICK_MS     = 16    # ~60 fps

# ─── LIF Neuron ────────────────────────────────────────────────────────────
#
#  Each timestep:
#    membrane = membrane * LEAK  +  weighted_input
#  If membrane >= THRESHOLD  →  spike fires, membrane resets to 0
#  If membrane <  THRESHOLD  →  no spike, membrane retains (leaky) value
#
#  Input neurons are also LIF — the voltage you set is the raw current
#  injected. If that current pushes the membrane past threshold, the
#  input neuron fires a spike. Otherwise nothing propagates.

class LIFNeuron:
    def __init__(self):
        self.membrane = 0.0

    def inject(self, current):
        """Inject raw current. Returns True if neuron fires."""
        self.membrane = self.membrane * LEAK + current
        if self.membrane >= THRESHOLD:
            self.membrane = 0.0
            return True
        return False

    def receive(self, weighted_sum):
        """Receive weighted synaptic input. Returns True if neuron fires."""
        self.membrane = self.membrane * LEAK + weighted_sum
        if self.membrane >= THRESHOLD:
            self.membrane = 0.0
            return True
        return False


class LIFLayer:
    def __init__(self, n_in, n_out, seed=1):
        rng = random.Random(seed)
        self.neurons = [LIFNeuron() for _ in range(n_out)]
        # Synaptic weights: shape [n_out][n_in]
        self.weights = [
            [rng.uniform(0.15, 0.85) for _ in range(n_in)]
            for _ in range(n_out)
        ]

    def forward(self, spike_train):
        """
        spike_train: list of floats (0.0 or voltage) from upstream.
        Returns list of bools — which neurons fired.
        Membrane state is PRESERVED between calls (true LIF accumulation).
        """
        fired = []
        for i, neuron in enumerate(self.neurons):
            weighted = sum(
                self.weights[i][j] * spike_train[j]
                for j in range(len(spike_train))
            )
            fired.append(neuron.receive(weighted))
        return fired

    @property
    def membranes(self):
        return [n.membrane for n in self.neurons]


class SNN:
    def __init__(self):
        # Input neurons are LIF too — voltage is injected current
        self.input_neurons = [LIFNeuron() for _ in range(INPUT_SIZE)]
        self.hidden = LIFLayer(INPUT_SIZE,  HIDDEN_SIZE, seed=1)
        self.output = LIFLayer(HIDDEN_SIZE, OUTPUT_SIZE, seed=2)

    def inject_inputs(self, currents):
        """
        Inject current into each input neuron.
        Only neurons that cross threshold emit a spike (1.0).
        Returns list of spike values (voltage if fired, else 0.0),
        and which input neurons actually fired.
        """
        spikes  = []
        fired   = []
        for i, neuron in enumerate(self.input_neurons):
            did_fire = neuron.inject(currents[i])
            spikes.append(currents[i] if did_fire else 0.0)
            fired.append(did_fire)
        return spikes, fired

    def propagate_to_hidden(self, input_spikes):
        """Feed input spikes into hidden layer. Returns which hidden neurons fired."""
        return self.hidden.forward(input_spikes)

    def propagate_to_output(self, hidden_spikes):
        """Feed hidden spikes (as float 1.0/0.0) into output layer."""
        return self.output.forward([1.0 if s else 0.0 for s in hidden_spikes])

    def reset(self):
        for n in self.input_neurons:
            n.membrane = 0.0
        for n in self.hidden.neurons:
            n.membrane = 0.0
        for n in self.output.neurons:
            n.membrane = 0.0


# ─── Visual constants ──────────────────────────────────────────────────────

NEURON_R   = 20
BG_COL     = QColor(15, 15, 22)
WIRE_COL   = QColor(255, 255, 255, 15)
WIRE_FIRE  = QColor(255, 160, 80, 120)
LAYER_COLS = [QColor(83, 74, 183), QColor(29, 158, 117), QColor(186, 117, 23)]
THRESH_COL = QColor(232, 89, 60)    # red-orange: fired
ACCUM_COL  = QColor(120, 200, 255)  # blue-white: accumulating below threshold


# ─── Canvas ────────────────────────────────────────────────────────────────

class NetworkCanvas(QWidget):
    def __init__(self, snn, parent=None):
        super().__init__(parent)
        self.snn    = snn
        self.layers = [INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE]

        # Membrane fractions for display (0..1 relative to THRESHOLD)
        self.memb   = [[0.0]*n for n in self.layers]
        # Which neurons are currently flashing (just fired)
        self.flash  = {}   # (li, ni) -> frames remaining
        # Animated pulses travelling along wires
        # Each: {lf, nf, lt, nt, t 0..1, v, real_spike bool}
        self.pulses = []
        # Pending: hidden spikes waiting to launch output pulses
        self._pending_hidden_spikes = None

        self.setMinimumSize(700, 440)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#0f0f16; border-radius:10px;")

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(TICK_MS)

    # ── layout ──────────────────────────────────────────────────────────────

    def neuron_pos(self, li, ni):
        W, H    = self.width(), self.height()
        x       = int(W * (li + 1) / (len(self.layers) + 1))
        n       = self.layers[li]
        spacing = min(72, (H - 90) // max(n, 1))
        total   = (n - 1) * spacing
        y0      = (H - total) // 2
        return x, y0 + ni * spacing

    # ── public: fire ────────────────────────────────────────────────────────

    def fire_input(self, currents):
        """
        Called by UI. Injects currents into input LIF neurons.
        Only those that spike emit pulses toward hidden layer.
        """
        input_spikes, input_fired = self.snn.inject_inputs(currents)

        # Update input membrane display
        for ni, neuron in enumerate(self.snn.input_neurons):
            # After inject, membrane was reset if fired — show 0 for fired,
            # accumulated fraction for non-fired
            if input_fired[ni]:
                self.memb[0][ni] = 1.0   # brief full flash
                self.flash[(0, ni)] = 14
            else:
                self.memb[0][ni] = min(
                    neuron.membrane / THRESHOLD, 1.0
                )

        # Pre-compute what hidden layer WILL do when spikes arrive,
        # but don't apply yet — wait until pulses land
        self._pending_input_spikes  = input_spikes
        self._pending_hidden_spikes = None

        any_fired = any(input_fired)

        if any_fired:
            # Spawn pulses for each fired input neuron → all hidden neurons
            for ni, fired in enumerate(input_fired):
                if fired:
                    for hi in range(HIDDEN_SIZE):
                        self.pulses.append(dict(
                            lf=0, nf=ni, lt=1, nt=hi,
                            t=0.0, v=input_spikes[ni],
                            real=True
                        ))
        # If no input fired, nothing travels — membranes just updated above

    # ── animation tick ──────────────────────────────────────────────────────

    def _tick(self):
        done = []
        for p in self.pulses:
            p['t'] += 1.0 / ANIM_STEPS
            if p['t'] >= 1.0:
                done.append(p)

        for p in done:
            self.pulses.remove(p)

        # Pulses that landed at hidden layer
        inp_landed = [p for p in done if p['lt'] == 1]
        if inp_landed:
            self._resolve_hidden()

        # Pulses that landed at output layer
        out_landed = [p for p in done if p['lt'] == 2]
        if out_landed:
            self._resolve_output()

        # Decay flash timers
        for k in list(self.flash.keys()):
            self.flash[k] -= 1
            if self.flash[k] <= 0:
                del self.flash[k]
                li, ni = k
                if self.memb[li][ni] == 1.0:
                    self.memb[li][ni] = 0.0

        self.update()

    def _resolve_hidden(self):
        """
        Pulses have landed at hidden layer.
        Now run the hidden LIF neurons with the input spikes.
        Neurons that fire spawn pulses to output; others just update membrane bar.
        """
        input_spikes = self._pending_input_spikes or ([0.0] * INPUT_SIZE)
        hidden_fired = self.snn.propagate_to_hidden(input_spikes)

        for ni, fired in enumerate(hidden_fired):
            if fired:
                self.memb[1][ni] = 1.0
                self.flash[(1, ni)] = 14
                # Spawn pulses hidden → output
                for oi in range(OUTPUT_SIZE):
                    self.pulses.append(dict(
                        lf=1, nf=ni, lt=2, nt=oi,
                        t=0.0, v=1.0,
                        real=True
                    ))
            else:
                # Show accumulated membrane (does NOT reset — true LIF)
                self.memb[1][ni] = min(
                    self.snn.hidden.neurons[ni].membrane / THRESHOLD, 1.0
                )

        self._pending_hidden_spikes = hidden_fired

    def _resolve_output(self):
        """
        Pulses landed at output layer.
        Run output LIF neurons. Flash those that fire.
        """
        hidden_spikes = self._pending_hidden_spikes or ([False] * HIDDEN_SIZE)
        output_fired  = self.snn.propagate_to_output(hidden_spikes)

        for ni, fired in enumerate(output_fired):
            if fired:
                self.memb[2][ni] = 1.0
                self.flash[(2, ni)] = 18
            else:
                self.memb[2][ni] = min(
                    self.snn.output.neurons[ni].membrane / THRESHOLD, 1.0
                )

    # ── paint ────────────────────────────────────────────────────────────────

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), BG_COL)

        # ── wires ──
        for li in range(len(self.layers) - 1):
            for a in range(self.layers[li]):
                for b in range(self.layers[li+1]):
                    x1,y1 = self.neuron_pos(li,   a)
                    x2,y2 = self.neuron_pos(li+1, b)
                    p.setPen(QPen(WIRE_COL, 0.7))
                    p.drawLine(x1, y1, x2, y2)

        # ── travelling pulses ──
        for pulse in self.pulses:
            if not pulse['real']:
                continue
            x1,y1 = self.neuron_pos(pulse['lf'], pulse['nf'])
            x2,y2 = self.neuron_pos(pulse['lt'], pulse['nt'])
            t = pulse['t']
            t_e   = t*t*(3 - 2*t)   # ease in-out
            cx    = x1 + (x2-x1)*t_e
            cy    = y1 + (y2-y1)*t_e

            # fade in / fade out around midpoint
            alpha = 1.0 - abs(t_e - 0.5) * 1.6
            alpha = max(0.0, min(1.0, alpha))

            r_glow = 12
            grad = QRadialGradient(cx, cy, r_glow)
            grad.setColorAt(0.0, QColor(255, 180, 80,  int(220*alpha)))
            grad.setColorAt(0.4, QColor(232,  89, 60,  int(140*alpha)))
            grad.setColorAt(1.0, QColor(232,  89, 60,  0))
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(grad))
            p.drawEllipse(QRectF(cx-r_glow, cy-r_glow, r_glow*2, r_glow*2))
            # bright core
            p.setBrush(QBrush(QColor(255, 240, 200, int(255*alpha))))
            p.drawEllipse(QRectF(cx-3, cy-3, 6, 6))

        # ── neurons ──
        for li, n in enumerate(self.layers):
            base = LAYER_COLS[li]
            for ni in range(n):
                x, y   = self.neuron_pos(li, ni)
                frac   = self.memb[li][ni]          # 0..1
                firing = (li, ni) in self.flash

                if firing:
                    # glow halo
                    r_glow = NEURON_R + 16
                    grad = QRadialGradient(x, y, r_glow)
                    grad.setColorAt(0.0, QColor(255, 160,  80, 170))
                    grad.setColorAt(0.6, QColor(232,  89,  60,  50))
                    grad.setColorAt(1.0, QColor(232,  89,  60,   0))
                    p.setPen(Qt.NoPen)
                    p.setBrush(QBrush(grad))
                    p.drawEllipse(QRectF(x-r_glow, y-r_glow, r_glow*2, r_glow*2))
                    fill = QColor(255, 165, 80)
                else:
                    # blue-tint accumulation colour — gets brighter as membrane fills
                    r = int(base.red()   * (1-frac) + 120 * frac)
                    g = int(base.green() * (1-frac) + 200 * frac)
                    b2= int(base.blue()  * (1-frac) + 255 * frac)
                    fill = QColor(r, g, b2, int(50 + frac * 205))

                p.setBrush(QBrush(fill))
                border = QColor(255,255,255, 90 if firing else 20)
                p.setPen(QPen(border, 1.5))
                p.drawEllipse(QRectF(x-NEURON_R, y-NEURON_R, NEURON_R*2, NEURON_R*2))

                # ── membrane bar ──
                bw, bh = 42, 5
                bx, by = x - bw//2, y + NEURON_R + 6
                # background track
                p.setBrush(QBrush(QColor(50, 50, 60)))
                p.setPen(Qt.NoPen)
                p.drawRoundedRect(bx, by, bw, bh, 2, 2)
                # threshold tick mark
                p.setPen(QPen(QColor(255,255,255,40), 1))
                p.drawLine(bx + bw - 1, by, bx + bw - 1, by + bh)
                # fill
                fill_w = int(bw * min(frac, 1.0))
                if fill_w > 0:
                    bar_col = THRESH_COL if firing else ACCUM_COL
                    p.setPen(Qt.NoPen)
                    p.setBrush(QBrush(bar_col))
                    p.drawRoundedRect(bx, by, fill_w, bh, 2, 2)

        # ── layer labels ──
        font = QFont("Arial", 10)
        p.setFont(font)
        labels = [
            f"Input  ({INPUT_SIZE})",
            f"Hidden ({HIDDEN_SIZE})",
            f"Output ({OUTPUT_SIZE})",
        ]
        for li, lbl in enumerate(labels):
            x, _ = self.neuron_pos(li, 0)
            p.setPen(QPen(QColor(140, 140, 160)))
            p.drawText(x-52, 10, 104, 20, Qt.AlignCenter, lbl)

        # ── threshold legend ──
        font2 = QFont("Arial", 9)
        p.setFont(font2)
        p.setPen(QPen(ACCUM_COL))
        p.drawText(10, self.height()-28, 200, 20, Qt.AlignLeft,
                   f"■ accumulating  (θ = {THRESHOLD})")
        p.setPen(QPen(THRESH_COL))
        p.drawText(10, self.height()-14, 200, 20, Qt.AlignLeft,
                   "■ fired spike")

        p.end()


# ─── Main Window ───────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SNN — Leaky Integrate-and-Fire Simulator")
        self.setMinimumSize(780, 660)

        self.snn    = SNN()
        self.canvas = NetworkCanvas(self.snn)

        # Input panel
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        pl = QVBoxLayout(panel)

        title = QLabel(
            f"Input neurons  —  "
            f"voltage injected as current  |  "
            f"threshold θ = {THRESHOLD} V"
        )
        pl.addWidget(title)

        self.sliders = []
        self.vlabels = []

        for i in range(INPUT_SIZE):
            row  = QHBoxLayout()
            lbl  = QLabel(f"N{i+1}")
            lbl.setFixedWidth(22)
            s    = QSlider(Qt.Horizontal)
            s.setRange(0, 100)
            s.setValue(40 + i*10)
            s.setFixedWidth(200)
            vlbl = QLabel(f"{s.value()/100:.2f} V")
            vlbl.setFixedWidth(52)
            btn  = QPushButton(f"⚡ N{i+1}")
            btn.setFixedWidth(72)
            btn.setToolTip(
                "Injects this voltage as current.\n"
                "Neuron only spikes if membrane crosses threshold."
            )
            btn.clicked.connect(lambda _, idx=i: self._fire_one(idx))
            s.valueChanged.connect(lambda v, l=vlbl: l.setText(f"{v/100:.2f} V"))
            row.addWidget(lbl)
            row.addWidget(s)
            row.addWidget(vlbl)
            row.addWidget(btn)
            pl.addLayout(row)
            self.sliders.append(s)
            self.vlabels.append(vlbl)

        btn_row = QHBoxLayout()
        fire_all  = QPushButton("⚡ Fire all inputs")
        reset_btn = QPushButton("↺ Reset all membranes")
        fire_all.clicked.connect(self._fire_all)
        reset_btn.clicked.connect(self._reset)
        btn_row.addWidget(fire_all)
        btn_row.addWidget(reset_btn)
        pl.addLayout(btn_row)

        self.info = QLabel(
            "Set voltage and click ⚡.  "
            "Input neuron only fires if voltage pushes it past θ.  "
            "Hidden neurons accumulate across events."
        )
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color:#888;font-size:11px;padding:4px 0;")

        central = QWidget()
        layout  = QVBoxLayout(central)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(panel)
        layout.addWidget(self.info)
        self.setCentralWidget(central)

    def _fire_one(self, idx):
        currents      = [0.0] * INPUT_SIZE
        currents[idx] = self.sliders[idx].value() / 100.0
        self.canvas.fire_input(currents)
        v = currents[idx]
        m = self.snn.input_neurons[idx].membrane
        if v + m >= THRESHOLD:
            self.info.setText(
                f"N{idx+1} fired at {v:.2f} V — spike propagating to hidden layer."
            )
        else:
            self.info.setText(
                f"N{idx+1} injected {v:.2f} V — membrane now "
                f"{self.snn.input_neurons[idx].membrane:.3f} V "
                f"(threshold {THRESHOLD} V). No spike yet."
            )

    def _fire_all(self):
        currents = [s.value() / 100.0 for s in self.sliders]
        self.canvas.fire_input(currents)
        v_str = ", ".join(f"{v:.2f}" for v in currents)
        self.info.setText(f"Injected [{v_str}] V into all input neurons.")

    def _reset(self):
        self.snn.reset()
        self.canvas.memb   = [[0.0]*n for n in self.canvas.layers]
        self.canvas.flash  = {}
        self.canvas.pulses = []
        self.canvas._pending_input_spikes  = None
        self.canvas._pending_hidden_spikes = None
        self.canvas.update()
        self.info.setText(
            "All membrane potentials reset to 0. "
            "LIF accumulators cleared."
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())