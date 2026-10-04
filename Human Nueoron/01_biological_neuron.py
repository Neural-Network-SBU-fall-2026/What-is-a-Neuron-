"""
02_neural_network_pro.py
========================

Interactive biophysical neuron + neural network simulator
---------------------------------------------------------

ویژگی‌ها:
    * مدل Hodgkin-Huxley کامل
    * ظاهر حرفه‌ای نورون (سوما، دندریت درختی، آکسون میلین‌دار، پایانه)
    * امکان ساخت شبکه‌ی چندنورونی با اتصالات سیناپسی
    * نمایش زنده‌ی پتانسیل غشاء، گیت‌ها و جریان‌ها
    * تحریک دستی، سیناپسی و دوره‌ای
    * refractory period
    * event-based spike propagation

نصب:
    pip install pygame numpy

اجرا:
    python 02_neural_network_pro.py

کنترل‌ها:
    SPACE : پالس جریان قوی
    E     : سیناپس تحریکی
    I     : سیناپس مهاری
    T     : تحریک دوره‌ای
    P     : توقف / ادامه
    R     : بازنشانی
    N     : جابه‌جایی بین حالت تک‌نورون و شبکه
    ESC   : خروج
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from collections import deque
from typing import Deque, List, Tuple, Optional

import pygame


# ============================================================
# 1. CONSTANTS
# ============================================================

WIDTH = 1500
HEIGHT = 920
FPS = 60

DT_MS = 0.02
STEPS_PER_FRAME = 5

CM = 1.0
G_NA_MAX = 120.0
G_K_MAX = 36.0
G_L = 0.3

E_NA = 50.0
E_K = -77.0
E_L = -54.387

RESTING_VOLTAGE = -65.0

E_EXCITATORY = 0.0
E_INHIBITORY = -75.0

TAU_EXCITATORY = 3.0
TAU_INHIBITORY = 8.0

GRAPH_X = 30
GRAPH_Y = 650
GRAPH_W = 1440
GRAPH_H = 230

SINGLE_NEURON_CENTER = (760, 320)


# ============================================================
# 2. UTILITY
# ============================================================

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def alpha_n(v: float) -> float:
    x = v + 55.0
    if abs(x) < 1e-9:
        return 0.1
    return 0.01 * x / (1.0 - math.exp(-x / 10.0))


def beta_n(v: float) -> float:
    return 0.125 * math.exp(-(v + 65.0) / 80.0)


def alpha_m(v: float) -> float:
    x = v + 40.0
    if abs(x) < 1e-9:
        return 1.0
    return 0.1 * x / (1.0 - math.exp(-x / 10.0))


def beta_m(v: float) -> float:
    return 4.0 * math.exp(-(v + 65.0) / 18.0)


def alpha_h(v: float) -> float:
    return 0.07 * math.exp(-(v + 65.0) / 20.0)


def beta_h(v: float) -> float:
    return 1.0 / (1.0 + math.exp(-(v + 35.0) / 10.0))


def steady_state_gate(v: float, a_fn, b_fn) -> float:
    a = a_fn(v)
    b = b_fn(v)
    return a / (a + b)


def lerp_color(c1, c2, t: float):
    t = clamp(t, 0.0, 1.0)
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


# ============================================================
# 3. ION STATE
# ============================================================

@dataclass
class IonState:
    m: float = field(default_factory=lambda: steady_state_gate(
        RESTING_VOLTAGE, alpha_m, beta_m))
    h: float = field(default_factory=lambda: steady_state_gate(
        RESTING_VOLTAGE, alpha_h, beta_h))
    n: float = field(default_factory=lambda: steady_state_gate(
        RESTING_VOLTAGE, alpha_n, beta_n))


# ============================================================
# 4. SYNAPSE (biological, used inside a neuron)
# ============================================================

@dataclass
class Synapse:
    kind: str
    tau_ms: float
    reversal_potential: float
    conductance: float = 0.0
    peak_conductance: float = 0.15

    def trigger(self, strength: float = 1.0) -> None:
        self.conductance += self.peak_conductance * strength

    def update(self, dt_ms: float) -> None:
        self.conductance *= math.exp(-dt_ms / self.tau_ms)

    def current(self, v: float) -> float:
        return self.conductance * (v - self.reversal_potential)


# ============================================================
# 5. ACTION POTENTIAL EVENT
# ============================================================

@dataclass
class ActionPotential:
    created_at_ms: float
    amplitude: float = 100.0
    progress: float = 0.0
    speed: float = 0.025

    def update(self) -> None:
        self.progress += self.speed

    @property
    def finished(self) -> bool:
        return self.progress >= 1.0


# ============================================================
# 6. BIOLOGICAL NEURON
# ============================================================

class BiologicalNeuron:
    def __init__(self, name: str = "N1") -> None:
        self.name = name
        self.time_ms = 0.0

        self.voltage = RESTING_VOLTAGE
        self.previous_voltage = RESTING_VOLTAGE

        self.ions = IonState()

        self.excitatory_synapse = Synapse(
            kind="excitatory",
            tau_ms=TAU_EXCITATORY,
            reversal_potential=E_EXCITATORY,
            peak_conductance=0.08,
        )

        self.inhibitory_synapse = Synapse(
            kind="inhibitory",
            tau_ms=TAU_INHIBITORY,
            reversal_potential=E_INHIBITORY,
            peak_conductance=0.12,
        )

        self.external_current = 0.0

        self.absolute_refractory_remaining = 0.0
        self.relative_refractory_remaining = 0.0

        self.spike_count = 0
        self.last_spike_time = -math.inf
        self.spike_events: List[float] = []

        self.action_potentials: List[ActionPotential] = []
        self.recently_fired = False

        self.current_na = 0.0
        self.current_k = 0.0
        self.current_l = 0.0
        self.current_exc = 0.0
        self.current_inh = 0.0

    # ---------------------------
    # Gating
    # ---------------------------
    def update_gates(self, dt_ms: float) -> None:
        v = self.voltage
        dm = alpha_m(v) * (1.0 - self.ions.m) - beta_m(v) * self.ions.m
        dh = alpha_h(v) * (1.0 - self.ions.h) - beta_h(v) * self.ions.h
        dn = alpha_n(v) * (1.0 - self.ions.n) - beta_n(v) * self.ions.n

        self.ions.m = clamp(self.ions.m + dt_ms * dm, 0.0, 1.0)
        self.ions.h = clamp(self.ions.h + dt_ms * dh, 0.0, 1.0)
        self.ions.n = clamp(self.ions.n + dt_ms * dn, 0.0, 1.0)

    # ---------------------------
    # Currents
    # ---------------------------
    def calculate_ionic_currents(self) -> Tuple[float, float, float]:
        g_na = G_NA_MAX * (self.ions.m ** 3) * self.ions.h
        g_k = G_K_MAX * (self.ions.n ** 4)

        i_na = g_na * (self.voltage - E_NA)
        i_k = g_k * (self.voltage - E_K)
        i_l = G_L * (self.voltage - E_L)
        return i_na, i_k, i_l

    # ---------------------------
    # Spike
    # ---------------------------
    def detect_spike(self) -> bool:
        return (self.previous_voltage < 0.0 <= self.voltage)

    def emit_spike(self) -> None:
        self.spike_count += 1
        self.last_spike_time = self.time_ms
        self.spike_events.append(self.time_ms)
        if len(self.spike_events) > 1000:
            self.spike_events.pop(0)

        self.action_potentials.append(
            ActionPotential(created_at_ms=self.time_ms)
        )
        self.absolute_refractory_remaining = 1.0
        self.relative_refractory_remaining = 3.0
        self.recently_fired = True

    # ---------------------------
    # Step
    # ---------------------------
    def step(self, dt_ms: float) -> None:
        self.previous_voltage = self.voltage

        self.excitatory_synapse.update(dt_ms)
        self.inhibitory_synapse.update(dt_ms)

        self.absolute_refractory_remaining = max(
            0.0, self.absolute_refractory_remaining - dt_ms)
        self.relative_refractory_remaining = max(
            0.0, self.relative_refractory_remaining - dt_ms)

        self.update_gates(dt_ms)

        self.current_na, self.current_k, self.current_l = \
            self.calculate_ionic_currents()

        self.current_exc = self.excitatory_synapse.current(self.voltage)
        self.current_inh = self.inhibitory_synapse.current(self.voltage)

        total_ionic = (
            self.current_na + self.current_k + self.current_l)
        total_syn = self.current_exc + self.current_inh

        net = self.external_current - total_ionic - total_syn
        dV = net / CM

        if self.absolute_refractory_remaining > 0.0:
            dV *= 0.05

        self.voltage = clamp(
            self.voltage + dt_ms * dV, -100.0, 60.0)
        self.time_ms += dt_ms

        if self.detect_spike():
            self.emit_spike()

        for ev in self.action_potentials:
            ev.update()
        self.action_potentials = [
            e for e in self.action_potentials if not e.finished]

        # reset the visual "just fired" flag
        if self.recently_fired and self.time_ms - self.last_spike_time > 4.0:
            self.recently_fired = False

    # ---------------------------
    # Stimulation
    # ---------------------------
    def stimulate_current(self, current: float) -> None:
        self.external_current = current

    def stimulate_excitatory(self, strength: float = 1.0) -> None:
        self.excitatory_synapse.trigger(strength)

    def stimulate_inhibitory(self, strength: float = 1.0) -> None:
        self.inhibitory_synapse.trigger(strength)

    # ---------------------------
    # Reset
    # ---------------------------
    def reset(self) -> None:
        self.time_ms = 0.0
        self.voltage = RESTING_VOLTAGE
        self.previous_voltage = RESTING_VOLTAGE
        self.ions = IonState()
        self.external_current = 0.0
        self.excitatory_synapse.conductance = 0.0
        self.inhibitory_synapse.conductance = 0.0
        self.absolute_refractory_remaining = 0.0
        self.relative_refractory_remaining = 0.0
        self.spike_count = 0
        self.last_spike_time = -math.inf
        self.spike_events.clear()
        self.action_potentials.clear()
        self.recently_fired = False

    def firing_rate_hz(self, window_ms: float = 1000.0) -> float:
        cutoff = self.time_ms - window_ms
        count = sum(t >= cutoff for t in self.spike_events)
        return count / (window_ms / 1000.0)


# ============================================================
# 7. NETWORK
# ============================================================

@dataclass
class SynapticConnection:
    """اتصال سیناپسی بین دو نورون در شبکه."""
    source: int
    target: int
    weight: float = 1.0
    kind: str = "excitatory"   # یا "inhibitory"
    delay_ms: float = 1.0
    # buffer of pending deliveries: (arrival_time, weight)
    pending: List[Tuple[float, float]] = field(default_factory=list)

    def schedule(self, now_ms: float) -> None:
        self.pending.append((now_ms + self.delay_ms, self.weight))


class NeuralNetwork:
    """شبکه‌ی عصبی ساده با اتصالات سیناپسی و تأخیر."""

    def __init__(self, neurons: List[BiologicalNeuron]) -> None:
        self.neurons = neurons
        self.connections: List[SynapticConnection] = []

    def add_connection(
        self,
        source: int,
        target: int,
        weight: float = 1.0,
        kind: str = "excitatory",
        delay_ms: float = 1.0,
    ) -> None:
        self.connections.append(
            SynapticConnection(source, target, weight, kind, delay_ms)
        )

    def step(self, dt_ms: float) -> None:
        # 1) step all neurons
        for n in self.neurons:
            n.step(dt_ms)

        # 2) detect newly fired neurons, schedule deliveries
        now = self.neurons[0].time_ms if self.neurons else 0.0
        for n in self.neurons:
            if n.recently_fired and n.last_spike_time > now - dt_ms * 1.5:
                # only schedule once per spike
                pass

        # Use a cleaner approach: each neuron reports spikes via a queue
        for idx, n in enumerate(self.neurons):
            if n.spike_events and n.spike_events[-1] >= now - dt_ms * 1.5:
                for c in self.connections:
                    if c.source == idx:
                        c.schedule(now)

        # 3) deliver pending
        for c in self.connections:
            remaining = []
            for arrival, w in c.pending:
                if arrival <= now:
                    tgt = self.neurons[c.target]
                    if c.kind == "excitatory":
                        tgt.stimulate_excitatory(w)
                    else:
                        tgt.stimulate_inhibitory(w)
                else:
                    remaining.append((arrival, w))
            c.pending = remaining

    def reset(self) -> None:
        for n in self.neurons:
            n.reset()
        for c in self.connections:
            c.pending.clear()


# ============================================================
# 8. HISTORY
# ============================================================

class SimulationHistory:
    def __init__(self, maxlen: int = 1600) -> None:
        self.time: Deque[float] = deque(maxlen=maxlen)
        self.voltage: Deque[float] = deque(maxlen=maxlen)
        self.na: Deque[float] = deque(maxlen=maxlen)
        self.k: Deque[float] = deque(maxlen=maxlen)
        self.exc: Deque[float] = deque(maxlen=maxlen)
        self.inh: Deque[float] = deque(maxlen=maxlen)

    def append(self, neuron: BiologicalNeuron) -> None:
        self.time.append(neuron.time_ms)
        self.voltage.append(neuron.voltage)
        self.na.append(neuron.current_na)
        self.k.append(neuron.current_k)
        self.exc.append(neuron.current_exc)
        self.inh.append(neuron.current_inh)

    def clear(self) -> None:
        for d in (self.time, self.voltage, self.na,
                  self.k, self.exc, self.inh):
            d.clear()


# ============================================================
# 9. VISUALIZER
# ============================================================

class Visualizer:
    BG = (10, 14, 22)
    PANEL = (20, 27, 39)
    PANEL_2 = (28, 36, 52)
    GRID = (45, 55, 70)
    TEXT = (230, 235, 245)
    MUTED = (150, 160, 175)
    ACCENT = (90, 180, 255)

    DENDRITE = (95, 175, 220)
    DENDRITE_DARK = (55, 110, 150)
    SOMA = (215, 135, 95)
    SOMA_DARK = (150, 80, 55)
    SOMA_HOT = (255, 220, 120)
    NUCLEUS = (60, 70, 95)
    NUCLEUS_LIGHT = (130, 145, 175)
    AXON = (170, 195, 120)
    AXON_DARK = (105, 125, 75)
    MYELIN = (85, 100, 130)
    MYELIN_LIGHT = (110, 130, 165)
    ACTIVE = (255, 210, 80)
    EXC = (100, 225, 130)
    INH = (220, 105, 130)
    CONN_EXC = (100, 225, 130, 120)
    CONN_INH = (220, 105, 130, 120)

    def __init__(self, mode: str = "single") -> None:
        pygame.init()
        pygame.display.set_caption(
            "Biological Neuron & Neural Network Simulator")

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("consolas", 18)
        self.small_font = pygame.font.SysFont("consolas", 15)
        self.tiny_font = pygame.font.SysFont("consolas", 13)
        self.title_font = pygame.font.SysFont("consolas", 24, bold=True)

        self.mode = mode  # "single" or "network"

        # single neuron
        self.neuron = BiologicalNeuron("N1")
        self.history = SimulationHistory()

        # network
        self.network = self._build_demo_network()
        self.network_histories = [
            SimulationHistory() for _ in self.network.neurons
        ]

        self.running = True
        self.paused = False

        self.manual_stimulus = False
        self.periodic_stimulus = False
        self.periodic_timer = 0.0
        self.periodic_interval = 500.0

    # --------------------------------------------------------
    # Network setup
    # --------------------------------------------------------
    def _build_demo_network(self) -> NeuralNetwork:
        """ساخت یک شبکه‌ی نمونه: 3 نورون با اتصالات."""
        n1 = BiologicalNeuron("N1")
        n2 = BiologicalNeuron("N2")
        n3 = BiologicalNeuron("N3")

        net = NeuralNetwork([n1, n2, n3])
        net.add_connection(0, 1, weight=0.9,
                           kind="excitatory", delay_ms=2.0)
        net.add_connection(1, 2, weight=0.9,
                           kind="excitatory", delay_ms=2.0)
        net.add_connection(0, 2, weight=0.5,
                           kind="excitatory", delay_ms=3.0)
        net.add_connection(2, 0, weight=0.7,
                           kind="inhibitory", delay_ms=2.5)
        return net

    # --------------------------------------------------------
    # Text helper
    # --------------------------------------------------------
    def draw_text(self, text, x, y, color=None, font=None):
        if color is None:
            color = self.TEXT
        if font is None:
            font = self.font
        surface = font.render(text, True, color)
        self.screen.blit(surface, (x, y))

    # --------------------------------------------------------
    # PROFESSIONAL NEURON DRAWING
    # --------------------------------------------------------
    def _draw_soma(self, cx, cy, radius, voltage):
        """سوما با گرادیان، هسته و هاله‌ی فعالیت."""
        # Activity halo
        if voltage > -45:
            intensity = clamp((voltage + 45) / 60.0, 0.0, 1.0)
            for i in range(6, 0, -1):
                r = radius + i * 5
                alpha = int(60 * intensity * (i / 6.0))
                surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                pygame.draw.circle(
                    surf, (255, 210, 80, alpha), (r, r), r)
                self.screen.blit(surf, (cx - r, cy - r))

        # Body base
        pygame.draw.circle(self.screen, self.SOMA_DARK,
                           (cx, cy), radius)
        # Gradient highlight
        for i in range(radius, 0, -3):
            t = 1.0 - (i / radius)
            col = lerp_color(self.SOMA_DARK, self.SOMA, t)
            pygame.draw.circle(self.screen, col,
                               (cx - 2, cy - 2), i)

        # Hot overlay when firing
        if voltage > -20:
            t = clamp((voltage + 20) / 60.0, 0.0, 1.0)
            col = lerp_color(self.SOMA, self.SOMA_HOT, t)
            pygame.draw.circle(self.screen, col, (cx, cy), radius - 4)

        # Nucleus
        pygame.draw.circle(self.screen, self.NUCLEUS,
                           (cx, cy), int(radius * 0.42))
        pygame.draw.circle(self.screen, self.NUCLEUS_LIGHT,
                           (cx - 3, cy - 3), int(radius * 0.18))

    def _draw_dendrite_branch(self, start, end, depth, max_depth=2):
        """رسم بازگشتی دندریت درختی."""
        pygame.draw.line(self.screen, self.DENDRITE_DARK,
                         start, end, max(2, 7 - depth * 2))
        pygame.draw.line(self.screen, self.DENDRITE,
                         start, end, max(1, 5 - depth * 2))

        if depth >= max_depth:
            # synaptic bouton
            pygame.draw.circle(self.screen, self.DENDRITE,
                               end, 4)
            return

        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy) + 1e-6

        # perpendicular
        px = -dy / length
        py = dx / length

        spread = length * 0.35
        b1 = (mx + px * spread, my + py * spread)
        b2 = (mx - px * spread, my - py * spread)

        self._draw_dendrite_branch(
            (mx, my), b1, depth + 1, max_depth)
        self._draw_dendrite_branch(
            (mx, my), b2, depth + 1, max_depth)

    def _draw_axon(self, start_x, end_x, cy, progress_points, myelinated=True):
        """آکسون میلین‌دار با گره‌های رانویه و امواج AP."""
        # main axon
        pygame.draw.line(self.screen, self.AXON_DARK,
                         (start_x, cy), (end_x, cy), 14)
        pygame.draw.line(self.screen, self.AXON,
                         (start_x, cy), (end_x, cy), 10)

        node_count = 12
        seg = (end_x - start_x) / node_count

        for i in range(node_count):
            x = int(start_x + i * seg + seg * 0.08)
            w = int(seg * 0.78)

            if myelinated:
                pygame.draw.rect(
                    self.screen, self.MYELIN,
                    (x, cy - 15, w, 30),
                    border_radius=8)
                pygame.draw.rect(
                    self.screen, self.MYELIN_LIGHT,
                    (x + 2, cy - 13, w - 4, 10),
                    border_radius=6)

            # Node of Ranvier
            node_x = int(start_x + (i + 1) * seg)
            pygame.draw.circle(self.screen, self.ACTIVE,
                               (node_x, cy), 4)

        # Progress points (spikes propagating)
        for ev in progress_points:
            x = int(start_x + ev.progress * (end_x - start_x))
            # glow
            for r, a in ((20, 40), (14, 80), (9, 160)):
                surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                pygame.draw.circle(
                    surf, (255, 210, 80, a), (r, r), r)
                self.screen.blit(surf, (x - r, cy - r))
            pygame.draw.circle(self.screen, self.ACTIVE, (x, cy), 9)

    def _draw_terminals(self, x, cy):
        """پایانه‌های آکسونی با دکمه‌های سیناپسی."""
        for i, offset in enumerate((-34, 0, 34)):
            end = (x + 55, cy + offset)
            pygame.draw.line(self.screen, self.AXON_DARK,
                             (x, cy), end, 7)
            pygame.draw.line(self.screen, self.AXON,
                             (x, cy), end, 4)

            # synaptic bouton
            color = self.EXC if i != 1 else self.INH
            pygame.draw.circle(self.screen, color,
                               (end[0] + 6, end[1]), 10)
            pygame.draw.circle(self.screen, (255, 255, 255),
                               (end[0] + 6, end[1]), 10, 2)
            pygame.draw.circle(self.screen, (255, 255, 255),
                               (end[0] + 3, end[1] - 3), 3)

    def draw_neuron_single(self):
        """رسم نورون حرفه‌ای در حالت تک‌نورون."""
        cx, cy = SINGLE_NEURON_CENTER

        # ---- Dendrites (recursive tree) ----
        dendrite_roots = [
            ((cx - 70, cy - 55), (cx - 260, cy - 150)),
            ((cx - 75, cy - 15), (cx - 300, cy - 60)),
            ((cx - 70, cy + 30), (cx - 280, cy + 90)),
            ((cx - 55, cy + 60), (cx - 200, cy + 160)),
            ((cx - 45, cy - 75), (cx - 170, cy - 180)),
        ]
        for start, end in dendrite_roots:
            self._draw_dendrite_branch(start, end, 0, max_depth=2)

        # ---- Soma ----
        self._draw_soma(cx, cy, 72, self.neuron.voltage)

        # ---- Axon hillock ----
        hillock = [
            (cx + 60, cy - 34),
            (cx + 130, cy - 20),
            (cx + 130, cy + 20),
            (cx + 60, cy + 34),
        ]
        pygame.draw.polygon(self.screen, self.AXON_DARK, hillock)
        pygame.draw.polygon(self.screen, self.AXON, hillock, 2)

        # ---- Axon ----
        axon_start = cx + 125
        axon_end = 1330
        self._draw_axon(
            axon_start, axon_end, cy,
            self.neuron.action_potentials, myelinated=True)

        # ---- Terminals ----
        self._draw_terminals(axon_end, cy)

        # ---- Labels ----
        self.draw_text("DENDRITES", cx - 290, cy - 205,
                       self.MUTED, self.small_font)
        self.draw_text("SOMA", cx - 22, cy + 88,
                       self.MUTED, self.small_font)
        self.draw_text("AXON HILLOCK", cx + 70, cy - 70,
                       self.MUTED, self.small_font)
        self.draw_text("MYELINATED AXON", 1010, cy - 50,
                       self.MUTED, self.small_font)
        self.draw_text("NODE OF RANVIER", 1010, cy + 40,
                       self.MUTED, self.small_font)
        self.draw_text("TERMINALS", 1290, cy + 75,
                       self.MUTED, self.small_font)

    # --------------------------------------------------------
    # NETWORK DRAWING
    # --------------------------------------------------------
    def draw_network(self):
        """رسم شبکه‌ی نورون‌ها با اتصالات."""
        positions = [
            (330, 320),
            (760, 200),
            (1150, 360),
        ]

        # ---- Connections ----
        for c in self.network.connections:
            x1, y1 = positions[c.source]
            x2, y2 = positions[c.target]

            color = (self.EXC if c.kind == "excitatory"
                     else self.INH)

            # curved line
            mx = (x1 + x2) // 2
            my = (y1 + y2) // 2 - 70

            points = []
            for t in [i / 20.0 for i in range(21)]:
                px = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * mx + t ** 2 * x2
                py = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * my + t ** 2 * y2
                points.append((px, py))

            # thickness by weight
            w = max(1, int(c.weight * 5))
            pygame.draw.lines(self.screen, color, False,
                              [(int(p[0]), int(p[1])) for p in points], w)

            # arrow head
            if len(points) >= 2:
                p1 = points[-2]
                p2 = points[-1]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                L = math.hypot(dx, dy) + 1e-6
                ux, uy = dx / L, dy / L
                # shrink to stop at neuron boundary
                tip = (p2[0] - ux * 70, p2[1] - uy * 70)
                left = (tip[0] - ux * 12 - uy * 7,
                        tip[1] - uy * 12 + ux * 7)
                right = (tip[0] - ux * 12 + uy * 7,
                         tip[1] - uy * 12 - ux * 7)
                pygame.draw.polygon(self.screen, color,
                                    [tip, left, right])

            # label
            self.draw_text(
                f"{c.kind[0].upper()} w={c.weight:.1f}",
                int(mx) - 30, int(my) - 10,
                self.MUTED, self.tiny_font)

        # ---- Neurons ----
        for i, (nx, ny) in enumerate(positions):
            n = self.network.neurons[i]

            # Dendrites (smaller, radial)
            for k in range(6):
                ang = k * math.pi / 3.0 + 0.3
                ex = nx + math.cos(ang) * 120
                ey = ny + math.sin(ang) * 120
                self._draw_dendrite_branch(
                    (nx, ny), (ex, ey), depth=0, max_depth=1)

            # Soma
            self._draw_soma(nx, ny, 45, n.voltage)

            # Axon stub
            pygame.draw.line(self.screen, self.AXON_DARK,
                             (nx + 40, ny), (nx + 75, ny), 7)
            pygame.draw.line(self.screen, self.AXON,
                             (nx + 40, ny), (nx + 75, ny), 4)

            # Name
            self.draw_text(n.name, nx - 14, ny - 70,
                           self.TEXT, self.font)

            # Voltage label
            self.draw_text(f"{n.voltage:6.1f} mV",
                           nx - 38, ny + 52,
                           self.ACTIVE if n.voltage > -45 else self.MUTED,
                           self.tiny_font)

            # Spikes
            self.draw_text(f"spikes: {n.spike_count}",
                           nx - 40, ny + 68,
                           self.MUTED, self.tiny_font)

        # legend
        self.draw_text("EXCITATORY", 1200, 60,
                       self.EXC, self.small_font)
        pygame.draw.line(self.screen, self.EXC,
                         (1160, 78), (1190, 78), 4)
        self.draw_text("INHIBITORY", 1200, 90,
                       self.INH, self.small_font)
        pygame.draw.line(self.screen, self.INH,
                         (1160, 108), (1190, 108), 4)

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------
    def draw_graph(self):
        x0, y0 = GRAPH_X, GRAPH_Y
        w, h = GRAPH_W, GRAPH_H

        pygame.draw.rect(self.screen, self.PANEL,
                         (x0, y0, w, h), border_radius=10)
        pygame.draw.rect(self.screen, self.PANEL_2,
                         (x0, y0, w, h), 2, border_radius=10)

        for i in range(1, 10):
            x = x0 + i * w // 10
            pygame.draw.line(self.screen, self.GRID,
                             (x, y0), (x, y0 + h), 1)
        for i in range(1, 5):
            y = y0 + i * h // 5
            pygame.draw.line(self.screen, self.GRID,
                             (x0, y), (x0 + w, y), 1)

        # choose which histories to draw
        if self.mode == "single":
            datasets = [(self.history, self.ACTIVE, "V (single)")]
        else:
            colors = [self.ACTIVE, self.EXC, self.INH]
            datasets = []
            for i, hist in enumerate(self.network_histories):
                datasets.append(
                    (hist, colors[i % len(colors)],
                     self.network.neurons[i].name))

        for hist, color, label in datasets:
            values = list(hist.voltage)
            if len(values) < 2:
                continue
            pts = []
            for i, v in enumerate(values):
                px = x0 + int(i / (len(values) - 1) * w)
                norm = (v + 100.0) / 160.0
                py = y0 + h - int(norm * h)
                pts.append((px, py))
            pygame.draw.lines(self.screen, color, False, pts, 2)

        # zero line
        zero_y = y0 + h - int((100.0 / 160.0) * h)
        pygame.draw.line(self.screen, self.GRID,
                         (x0, zero_y), (x0 + w, zero_y), 1)

        self.draw_text("MEMBRANE POTENTIAL", x0 + 15, y0 + 10,
                       self.TEXT, self.small_font)
        self.draw_text("+60 mV", x0 + w - 75, y0 + 10,
                       self.MUTED, self.tiny_font)
        self.draw_text("0 mV", x0 + w - 60, zero_y - 18,
                       self.MUTED, self.tiny_font)
        self.draw_text("-100 mV", x0 + w - 80, y0 + h - 20,
                       self.MUTED, self.tiny_font)

        # legend
        lx = x0 + 180
        for _, color, label in datasets:
            pygame.draw.rect(self.screen, color, (lx, y0 + 12, 12, 12))
            self.draw_text(label, lx + 18, y0 + 10,
                           self.MUTED, self.tiny_font)
            lx += 100

    # --------------------------------------------------------
    # INFO
    # --------------------------------------------------------
    def draw_info(self):
        x, y = 30, 30

        title = ("BIOLOGICAL NEURON — SINGLE"
                 if self.mode == "single"
                 else "NEURAL NETWORK")
        self.draw_text(title, x, y, self.TEXT, self.title_font)

        if self.mode == "single":
            n = self.neuron
            lines = [
                f"time          : {n.time_ms:9.2f} ms",
                f"membrane V    : {n.voltage:9.2f} mV",
                f"Na gate m     : {n.ions.m:9.4f}",
                f"Na gate h     : {n.ions.h:9.4f}",
                f"K  gate n     : {n.ions.n:9.4f}",
                f"I_Na          : {n.current_na:9.3f}",
                f"I_K           : {n.current_k:9.3f}",
                f"I_Leak        : {n.current_l:9.3f}",
                f"I_Excitatory  : {n.current_exc:9.3f}",
                f"I_Inhibitory  : {n.current_inh:9.3f}",
                f"spikes        : {n.spike_count}",
                f"firing rate   : {n.firing_rate_hz():9.2f} Hz",
            ]
        else:
            lines = []
            for nn in self.network.neurons:
                lines.append(
                    f"{nn.name}: V={nn.voltage:7.2f} mV  "
                    f"spk={nn.spike_count:3d}  "
                    f"fr={nn.firing_rate_hz():6.1f} Hz"
                )
            lines.append("")
            lines.append(f"connections: {len(self.network.connections)}")

        for i, line in enumerate(lines):
            self.draw_text(line, x, y + 45 + i * 19,
                           self.MUTED, self.small_font)

        # Controls panel
        cx = WIDTH - 320
        controls = [
            "CONTROLS",
            "SPACE : current pulse",
            "E     : excitatory synapse",
            "I     : inhibitory synapse",
            "T     : toggle periodic",
            "P     : pause / resume",
            "R     : reset",
            "N     : single / network",
            "ESC   : quit",
        ]
        for i, t in enumerate(controls):
            self.draw_text(
                t, cx, 45 + i * 22,
                self.TEXT if i == 0 else self.MUTED,
                self.font if i == 0 else self.small_font)

        # Mode badge
        badge = "MODE: " + ("SINGLE" if self.mode == "single" else "NETWORK")
        self.draw_text(badge, WIDTH - 320, 20, self.ACCENT, self.font)

    # --------------------------------------------------------
    # Periodic stimulation
    # --------------------------------------------------------
    def update_periodic_stimulation(self, dt_ms: float):
        if not self.periodic_stimulus:
            return
        self.periodic_timer += dt_ms
        if self.periodic_timer >= self.periodic_interval:
            self.periodic_timer = 0.0
            if self.mode == "single":
                self.neuron.stimulate_current(10.0)
            else:
                self.network.neurons[0].stimulate_current(10.0)
        else:
            if self.mode == "single":
                self.neuron.stimulate_current(0.0)
            else:
                self.network.neurons[0].stimulate_current(0.0)

    # --------------------------------------------------------
    # Events
    # --------------------------------------------------------
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False

                elif event.key == pygame.K_SPACE:
                    self.manual_stimulus = True
                    if self.mode == "single":
                        self.neuron.stimulate_current(15.0)
                    else:
                        self.network.neurons[0].stimulate_current(20.0)

                elif event.key == pygame.K_e:
                    if self.mode == "single":
                        self.neuron.stimulate_excitatory(1.0)
                    else:
                        self.network.neurons[0].stimulate_excitatory(1.0)

                elif event.key == pygame.K_i:
                    if self.mode == "single":
                        self.neuron.stimulate_inhibitory(1.0)
                    else:
                        self.network.neurons[0].stimulate_inhibitory(1.0)

                elif event.key == pygame.K_t:
                    self.periodic_stimulus = not self.periodic_stimulus
                    self.periodic_timer = 0.0

                elif event.key == pygame.K_p:
                    self.paused = not self.paused

                elif event.key == pygame.K_r:
                    if self.mode == "single":
                        self.neuron.reset()
                        self.history.clear()
                    else:
                        self.network.reset()
                        for h in self.network_histories:
                            h.clear()

                elif event.key == pygame.K_n:
                    self.mode = ("network" if self.mode == "single"
                                 else "single")

            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    self.manual_stimulus = False
                    if not self.periodic_stimulus:
                        if self.mode == "single":
                            self.neuron.stimulate_current(0.0)
                        else:
                            self.network.neurons[0].stimulate_current(0.0)

    # --------------------------------------------------------
    # Run
    # --------------------------------------------------------
    def run(self):
        while self.running:
            self.handle_events()

            if not self.paused:
                self.update_periodic_stimulation(
                    DT_MS * STEPS_PER_FRAME)

                if self.mode == "single":
                    for _ in range(STEPS_PER_FRAME):
                        self.neuron.step(DT_MS)
                    self.history.append(self.neuron)
                else:
                    for _ in range(STEPS_PER_FRAME):
                        self.network.step(DT_MS)
                    for i, n in enumerate(self.network.neurons):
                        self.network_histories[i].append(n)

            self.screen.fill(self.BG)

            if self.mode == "single":
                self.draw_neuron_single()
            else:
                self.draw_network()

            self.draw_graph()
            self.draw_info()

            if self.paused:
                self.draw_text("PAUSED", WIDTH // 2 - 45, 600,
                               self.ACTIVE, self.title_font)

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()


# ============================================================
# 10. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    Visualizer(mode="single").run()