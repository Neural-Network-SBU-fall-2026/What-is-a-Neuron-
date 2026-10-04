"""
01_biological_neuron.py
=======================

Interactive physiological neuron simulator
-------------------------------------------

هدف:
    شبیه‌سازی یک نورون زیستی به‌صورت قابل توسعه، با تمرکز بر:

    * غشای سلولی و ظرفیت خازنی
    * پتانسیل استراحت
    * کانال‌های ولتاژ-وابسته Na+ و K+
    * جریان نشتی
    * مدل Hodgkin-Huxley برای تولید Action Potential
    * دندریت‌ها، سوما، Axon Hillock، آکسون و پایانه آکسونی
    * سیناپس‌های تحریکی و مهاری
    * EPSP / IPSP
    * دوره refractory
    * نمایش زنده در pygame
    * نمودار ولتاژ و جریان‌ها
    * ثبت رویدادهای Action Potential
    * تحریک دستی با کلید SPACE
    * تحریک سیناپسی با کلید E / I
    * تحریک دوره‌ای با کلید T
    * توقف/ادامه با P
    * بازنشانی با R

نکته علمی:
    این برنامه «تمام فیزیولوژی نورون انسانی» را مدل نمی‌کند.
    نورون واقعی بسیار پیچیده‌تر است و به کانال‌های یونی متعدد،
    morphology سه‌بعدی، compartmentهای متعدد، Ca2+ dynamics،
    neurotransmitter kinetics، شبکه‌های مولکولی و غیره نیاز دارد.

    هسته این نسخه بر Hodgkin-Huxley کلاسیک بنا شده است و به‌صورت
    آگاهانه به‌عنوان یک مدل biophysical قابل توسعه نوشته شده است.

نصب:
    pip install pygame numpy

اجرا:
    python 01_biological_neuron.py
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from collections import deque
from typing import Deque, List, Tuple

import numpy as np
import pygame


# ============================================================
# 1. CONSTANTS
# ============================================================

WIDTH = 1500
HEIGHT = 900
FPS = 60

# Simulation time:
# هر فریم pygame چند گام کوچک فیزیولوژیک را اجرا می‌کند.
DT_MS = 0.02
STEPS_PER_FRAME = 5

# Hodgkin-Huxley canonical parameters
CM = 1.0                 # uF/cm^2
G_NA_MAX = 120.0        # mS/cm^2
G_K_MAX = 36.0           # mS/cm^2
G_L = 0.3                # mS/cm^2

E_NA = 50.0              # mV
E_K = -77.0              # mV
E_L = -54.387            # mV

RESTING_VOLTAGE = -65.0  # mV

# Synaptic reversal potentials
E_EXCITATORY = 0.0       # mV, AMPA-like approximation
E_INHIBITORY = -75.0     # mV, GABA-like approximation

# Synaptic time constants
TAU_EXCITATORY = 3.0     # ms
TAU_INHIBITORY = 8.0     # ms

# Visualization
GRAPH_X = 30
GRAPH_Y = 620
GRAPH_W = 1440
GRAPH_H = 230

NEURON_CENTER = (760, 300)


# ============================================================
# 2. UTILITY FUNCTIONS
# ============================================================

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def alpha_n(v: float) -> float:
    """
    Hodgkin-Huxley alpha_n.

    n controls potassium-channel activation.
    """
    x = v + 55.0
    if abs(x) < 1e-9:
        return 0.1
    return 0.01 * x / (1.0 - math.exp(-x / 10.0))


def beta_n(v: float) -> float:
    """Potassium-channel deactivation rate."""
    return 0.125 * math.exp(-(v + 65.0) / 80.0)


def alpha_m(v: float) -> float:
    """
    Sodium-channel activation rate.

    m rises rapidly during depolarization.
    """
    x = v + 40.0
    if abs(x) < 1e-9:
        return 1.0
    return 0.1 * x / (1.0 - math.exp(-x / 10.0))


def beta_m(v: float) -> float:
    """Sodium-channel activation decay."""
    return 4.0 * math.exp(-(v + 65.0) / 18.0)


def alpha_h(v: float) -> float:
    """Sodium-channel availability/inactivation rate."""
    return 0.07 * math.exp(-(v + 65.0) / 20.0)


def beta_h(v: float) -> float:
    """Sodium-channel recovery rate."""
    return 1.0 / (1.0 + math.exp(-(v + 35.0) / 10.0))


def steady_state_gate(v: float, alpha_fn, beta_fn) -> float:
    """Calculate x_inf = alpha / (alpha + beta)."""
    a = alpha_fn(v)
    b = beta_fn(v)
    return a / (a + b)


# ============================================================
# 3. PHYSIOLOGICAL STATE
# ============================================================

@dataclass
class IonState:
    """
    State of the main Hodgkin-Huxley gating variables.

    m:
        Sodium activation.

    h:
        Sodium inactivation.

    n:
        Potassium activation.
    """

    m: float = field(default_factory=lambda: steady_state_gate(
        RESTING_VOLTAGE, alpha_m, beta_m
    ))
    h: float = field(default_factory=lambda: steady_state_gate(
        RESTING_VOLTAGE, alpha_h, beta_h
    ))
    n: float = field(default_factory=lambda: steady_state_gate(
        RESTING_VOLTAGE, alpha_n, beta_n
    ))


@dataclass
class Synapse:
    """
    ساده‌ترین مدل conductance-based synapse.

    نوع:
        excitatory -> E_syn ~= 0 mV
        inhibitory -> E_syn ~= -75 mV

    activation:
        با دریافت spike افزایش می‌یابد.

    سپس activation با یک decay نمایی کاهش پیدا می‌کند.
    """

    kind: str
    tau_ms: float
    reversal_potential: float
    conductance: float = 0.0
    peak_conductance: float = 0.15

    def trigger(self, strength: float = 1.0) -> None:
        self.conductance += self.peak_conductance * strength

    def update(self, dt_ms: float) -> None:
        decay = math.exp(-dt_ms / self.tau_ms)
        self.conductance *= decay

    def current(self, voltage: float) -> float:
        """
        I_syn = g_syn * (V - E_syn)
        """
        return self.conductance * (voltage - self.reversal_potential)


# ============================================================
# 4. BIOLOGICAL COMPARTMENTS
# ============================================================

@dataclass
class Dendrite:
    """
    Dendrite به‌صورت یک compartment ساده.

    در نسخه‌های بعدی می‌توانیم هر dendrite را به چندین
    compartment تقسیم کنیم و morphology واقعی را وارد کنیم.
    """

    length_um: float
    diameter_um: float
    excitatory_synapses: int = 0
    inhibitory_synapses: int = 0

    def surface_area(self) -> float:
        return math.pi * self.diameter_um * self.length_um


@dataclass
class Soma:
    """
    Cell body.

    سوما محل اصلی یکپارچه‌سازی جریان‌های ورودی در این مدل است.
    """

    diameter_um: float = 20.0

    def surface_area(self) -> float:
        return 4.0 * math.pi * (self.diameter_um / 2.0) ** 2


@dataclass
class Axon:
    """
    Axon با در نظر گرفتن myelin و Node of Ranvier.

    propagation در این نسخه به‌صورت visualization/event-based
    نمایش داده می‌شود؛ هنوز یک cable equation چند-compartment
    کامل برای آکسون اجرا نمی‌شود.
    """

    length_um: float = 1000.0
    diameter_um: float = 1.0
    node_count: int = 20
    myelinated: bool = True

    @property
    def conduction_delay_ms(self) -> float:
        if self.myelinated:
            # مقدار تقریبی صرفاً برای visualization.
            velocity_m_per_s = 100.0
        else:
            velocity_m_per_s = 1.0

        length_m = self.length_um * 1e-6
        return 1000.0 * length_m / velocity_m_per_s


# ============================================================
# 5. ACTION POTENTIAL EVENT
# ============================================================

@dataclass
class ActionPotential:
    """
    یک رویداد spike برای visualization و ارتباط نورون‌ها.
    """

    created_at_ms: float
    amplitude: float = 100.0
    progress: float = 0.0
    speed: float = 0.04

    def update(self) -> None:
        self.progress += self.speed

    @property
    def finished(self) -> bool:
        return self.progress >= 1.0


# ============================================================
# 6. MAIN BIOPHYSICAL NEURON
# ============================================================

class BiologicalNeuron:
    """
    نورون biophysical مبتنی بر Hodgkin-Huxley.

    ساختار:

        Dendrites
            |
            v
          Soma
            |
       Axon Hillock
            |
            v
          Axon
            |
            v
      Axon Terminal

    هسته الکتریکی در این نسخه در soma/hillock متمرکز است.
    """

    def __init__(self) -> None:
        self.time_ms = 0.0

        self.voltage = RESTING_VOLTAGE
        self.previous_voltage = RESTING_VOLTAGE

        self.ions = IonState()

        # Morphology
        self.dendrites: List[Dendrite] = [
            Dendrite(160, 3.5, excitatory_synapses=4),
            Dendrite(130, 2.5, excitatory_synapses=3),
            Dendrite(110, 2.0, inhibitory_synapses=2),
            Dendrite(190, 3.0, excitatory_synapses=4),
            Dendrite(100, 2.0, inhibitory_synapses=1),
        ]

        self.soma = Soma()
        self.axon = Axon()

        # Synapses
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

        # External current
        self.external_current = 0.0

        # Refractory state
        self.absolute_refractory_remaining = 0.0
        self.relative_refractory_remaining = 0.0

        # Spike detection
        self.spike_count = 0
        self.last_spike_time = -math.inf
        self.spike_events: List[float] = []

        # Propagating visual events
        self.action_potentials: List[ActionPotential] = []

        # Diagnostics
        self.current_na = 0.0
        self.current_k = 0.0
        self.current_l = 0.0
        self.current_exc = 0.0
        self.current_inh = 0.0

    # --------------------------------------------------------
    # GATING DYNAMICS
    # --------------------------------------------------------

    def update_gates(self, dt_ms: float) -> None:
        """
        dx/dt = alpha_x(V)(1-x) - beta_x(V)x

        Integration:
            Euler method.

        برای پایداری، مقادیر gate بین 0 و 1 محدود می‌شوند.
        """

        v = self.voltage

        dm = alpha_m(v) * (1.0 - self.ions.m) - beta_m(v) * self.ions.m
        dh = alpha_h(v) * (1.0 - self.ions.h) - beta_h(v) * self.ions.h
        dn = alpha_n(v) * (1.0 - self.ions.n) - beta_n(v) * self.ions.n

        self.ions.m += dt_ms * dm
        self.ions.h += dt_ms * dh
        self.ions.n += dt_ms * dn

        self.ions.m = clamp(self.ions.m, 0.0, 1.0)
        self.ions.h = clamp(self.ions.h, 0.0, 1.0)
        self.ions.n = clamp(self.ions.n, 0.0, 1.0)

    # --------------------------------------------------------
    # IONIC CURRENTS
    # --------------------------------------------------------

    def calculate_ionic_currents(self) -> Tuple[float, float, float]:
        """
        Hodgkin-Huxley:

        I_Na = g_Na * m^3 * h * (V - E_Na)
        I_K  = g_K  * n^4     * (V - E_K)
        I_L  = g_L            * (V - E_L)
        """

        g_na = G_NA_MAX * (self.ions.m ** 3) * self.ions.h
        g_k = G_K_MAX * (self.ions.n ** 4)

        i_na = g_na * (self.voltage - E_NA)
        i_k = g_k * (self.voltage - E_K)
        i_l = G_L * (self.voltage - E_L)

        return i_na, i_k, i_l

    # --------------------------------------------------------
    # SPIKE DETECTION
    # --------------------------------------------------------

    def detect_spike(self) -> bool:
        """
        Spike را هنگام عبور صعودی از threshold تشخیص می‌دهیم.

        این threshold برای event detection است، نه اینکه
        تولید action potential را به‌صورت if/else انجام دهد.
        """

        threshold = 0.0

        return (
            self.previous_voltage < threshold
            and self.voltage >= threshold
        )

    def emit_spike(self) -> None:
        self.spike_count += 1
        self.last_spike_time = self.time_ms
        self.spike_events.append(self.time_ms)

        # محدود کردن حافظه eventها
        if len(self.spike_events) > 1000:
            self.spike_events.pop(0)

        self.action_potentials.append(
            ActionPotential(created_at_ms=self.time_ms)
        )

        # Absolute refractory period
        self.absolute_refractory_remaining = 1.0

        # Relative refractory period
        self.relative_refractory_remaining = 3.0

    # --------------------------------------------------------
    # SIMULATION STEP
    # --------------------------------------------------------

    def step(self, dt_ms: float) -> None:
        """
        اجرای یک گام زمانی فیزیولوژیک.

        ترتیب:

            1. synapse decay
            2. refractory timers
            3. ionic gates
            4. ionic currents
            5. synaptic currents
            6. membrane equation
            7. spike detection
            8. event propagation
        """

        self.previous_voltage = self.voltage

        # Synaptic kinetics
        self.excitatory_synapse.update(dt_ms)
        self.inhibitory_synapse.update(dt_ms)

        # Refractory timers
        self.absolute_refractory_remaining = max(
            0.0,
            self.absolute_refractory_remaining - dt_ms
        )

        self.relative_refractory_remaining = max(
            0.0,
            self.relative_refractory_remaining - dt_ms
        )

        # Update gates
        self.update_gates(dt_ms)

        # Currents
        self.current_na, self.current_k, self.current_l = \
            self.calculate_ionic_currents()

        self.current_exc = self.excitatory_synapse.current(self.voltage)
        self.current_inh = self.inhibitory_synapse.current(self.voltage)

        total_ionic_current = (
            self.current_na
            + self.current_k
            + self.current_l
        )

        total_synaptic_current = (
            self.current_exc
            + self.current_inh
        )

        # Membrane equation:
        #
        # C_m dV/dt = I_ext - I_ion - I_syn
        #
        net_current = (
            self.external_current
            - total_ionic_current
            - total_synaptic_current
        )

        dV_dt = net_current / CM

        # Absolute refractory:
        # در مدل ساده از اعمال تحریک خارجی شدید جلوگیری می‌کنیم.
        if self.absolute_refractory_remaining > 0.0:
            dV_dt *= 0.05

        self.voltage += dt_ms * dV_dt

        # Numerical safety
        self.voltage = clamp(self.voltage, -100.0, 60.0)

        self.time_ms += dt_ms

        # Spike detection
        if self.detect_spike():
            self.emit_spike()

        # Update visual AP events
        for event in self.action_potentials:
            event.update()

        self.action_potentials = [
            event for event in self.action_potentials
            if not event.finished
        ]

    # --------------------------------------------------------
    # STIMULATION
    # --------------------------------------------------------

    def stimulate_current(self, current: float) -> None:
        self.external_current = current

    def stimulate_excitatory(self, strength: float = 1.0) -> None:
        self.excitatory_synapse.trigger(strength)

    def stimulate_inhibitory(self, strength: float = 1.0) -> None:
        self.inhibitory_synapse.trigger(strength)

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # INFORMATION
    # --------------------------------------------------------

    def firing_rate_hz(self, window_ms: float = 1000.0) -> float:
        cutoff = self.time_ms - window_ms
        count = sum(t >= cutoff for t in self.spike_events)
        return count / (window_ms / 1000.0)


# ============================================================
# 7. DATA LOGGER
# ============================================================

class SimulationHistory:
    """
    نگهداری داده‌های اخیر برای رسم نمودار.

    برای جلوگیری از رشد نامحدود حافظه از deque استفاده شده است.
    """

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
        self.time.clear()
        self.voltage.clear()
        self.na.clear()
        self.k.clear()
        self.exc.clear()
        self.inh.clear()


# ============================================================
# 8. PYGAME VISUALIZER
# ============================================================

class Visualizer:
    """
    رابط گرافیکی کامل pygame.

    مسئولیت این کلاس فقط visualization است؛
    منطق فیزیولوژی داخل BiologicalNeuron باقی می‌ماند.
    """

    BG = (12, 16, 24)
    PANEL = (20, 27, 39)
    GRID = (45, 55, 70)
    TEXT = (225, 230, 240)
    MUTED = (150, 160, 175)
    DENDRITE = (90, 180, 220)
    SOMA = (210, 130, 90)
    AXON = (150, 180, 110)
    ACTIVE = (255, 210, 80)
    EXC = (100, 220, 130)
    INH = (210, 100, 120)

    def __init__(self, neuron: BiologicalNeuron) -> None:
        pygame.init()
        pygame.display.set_caption(
            "Biological Neuron Simulator — Hodgkin-Huxley"
        )

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("consolas", 18)
        self.small_font = pygame.font.SysFont("consolas", 15)
        self.title_font = pygame.font.SysFont("consolas", 24, bold=True)

        self.neuron = neuron
        self.history = SimulationHistory()

        self.running = True
        self.paused = False

        self.manual_stimulus = False
        self.periodic_stimulus = False

        self.periodic_timer = 0.0
        self.periodic_interval = 500.0

    # --------------------------------------------------------
    # TEXT
    # --------------------------------------------------------

    def draw_text(
        self,
        text: str,
        x: int,
        y: int,
        color=None,
        font=None
    ) -> None:
        if color is None:
            color = self.TEXT
        if font is None:
            font = self.font

        surface = font.render(text, True, color)
        self.screen.blit(surface, (x, y))

    # --------------------------------------------------------
    # NEURON DRAWING
    # --------------------------------------------------------

    def draw_neuron(self) -> None:
        cx, cy = NEURON_CENTER

        # Dendrites
        dendrite_points = [
            ((cx - 60, cy - 25), (cx - 220, cy - 100)),
            ((cx - 60, cy - 5), (cx - 250, cy - 35)),
            ((cx - 55, cy + 20), (cx - 230, cy + 80)),
            ((cx - 45, cy + 35), (cx - 170, cy + 135)),
            ((cx - 75, cy - 45), (cx - 160, cy - 145)),
        ]

        for start, end in dendrite_points:
            pygame.draw.line(
                self.screen,
                self.DENDRITE,
                start,
                end,
                7
            )

            # Branches
            mx = (start[0] + end[0]) // 2
            my = (start[1] + end[1]) // 2

            pygame.draw.line(
                self.screen,
                self.DENDRITE,
                (mx, my),
                (mx - 50, my - 35),
                4
            )

            pygame.draw.line(
                self.screen,
                self.DENDRITE,
                (mx, my),
                (mx - 45, my + 35),
                4
            )

        # Synaptic buttons
        for i in range(8):
            angle = i * math.pi / 4.0
            x = cx - 220 + int(math.cos(angle) * 20)
            y = cy + int(math.sin(angle) * 100)

            pygame.draw.circle(
                self.screen,
                self.EXC if i % 3 else self.INH,
                (x, y),
                6
            )

        # Soma
        soma_color = self.SOMA

        if self.neuron.voltage > -40:
            soma_color = self.ACTIVE

        pygame.draw.ellipse(
            self.screen,
            soma_color,
            pygame.Rect(cx - 70, cy - 70, 140, 140)
        )

        # Nucleus
        pygame.draw.circle(
            self.screen,
            (70, 80, 100),
            (cx, cy),
            28
        )

        pygame.draw.circle(
            self.screen,
            (120, 135, 160),
            (cx, cy),
            10
        )

        # Axon hillock
        hillock = [
            (cx + 60, cy - 30),
            (cx + 120, cy - 18),
            (cx + 120, cy + 18),
            (cx + 60, cy + 30),
        ]

        pygame.draw.polygon(
            self.screen,
            self.AXON,
            hillock
        )

        # Axon
        axon_start = cx + 115
        axon_end = 1320

        pygame.draw.line(
            self.screen,
            self.AXON,
            (axon_start, cy),
            (axon_end, cy),
            12
        )

        # Myelin segments
        node_count = 10
        segment_length = (
            axon_end - axon_start
        ) / node_count

        for i in range(node_count):
            x = int(
                axon_start
                + i * segment_length
                + segment_length * 0.1
            )
            w = int(segment_length * 0.72)

            pygame.draw.rect(
                self.screen,
                (85, 100, 125),
                (x, cy - 14, w, 28),
                border_radius=7
            )

        # Action potential wave
        for event in self.neuron.action_potentials:
            x = int(
                axon_start
                + event.progress
                * (axon_end - axon_start)
            )

            pygame.draw.circle(
                self.screen,
                self.ACTIVE,
                (x, cy),
                17
            )

        # Axon terminals
        terminal_x = axon_end

        for offset in (-30, 0, 30):
            pygame.draw.line(
                self.screen,
                self.AXON,
                (terminal_x, cy),
                (terminal_x + 60, cy + offset),
                6
            )

            pygame.draw.circle(
                self.screen,
                self.EXC if offset != 0 else self.INH,
                (terminal_x + 65, cy + offset),
                9
            )

        # Labels
        self.draw_text(
            "DENDRITES",
            cx - 240,
            cy - 180,
            self.MUTED,
            self.small_font
        )

        self.draw_text(
            "SOMA",
            cx - 25,
            cy + 85,
            self.MUTED,
            self.small_font
        )

        self.draw_text(
            "AXON HILLOCK",
            cx + 70,
            cy - 65,
            self.MUTED,
            self.small_font
        )

        self.draw_text(
            "MYELINATED AXON",
            1000,
            cy - 45,
            self.MUTED,
            self.small_font
        )

        self.draw_text(
            "AXON TERMINAL",
            1270,
            cy + 50,
            self.MUTED,
            self.small_font
        )

    # --------------------------------------------------------
    # GRAPH
    # --------------------------------------------------------

    def draw_graph(self) -> None:
        x0 = GRAPH_X
        y0 = GRAPH_Y
        w = GRAPH_W
        h = GRAPH_H

        pygame.draw.rect(
            self.screen,
            self.PANEL,
            (x0, y0, w, h),
            border_radius=8
        )

        # Grid
        for i in range(1, 10):
            x = x0 + i * w // 10
            pygame.draw.line(
                self.screen,
                self.GRID,
                (x, y0),
                (x, y0 + h),
                1
            )

        for i in range(1, 5):
            y = y0 + i * h // 5
            pygame.draw.line(
                self.screen,
                self.GRID,
                (x0, y),
                (x0 + w, y),
                1
            )

        # Voltage scaling
        values = list(self.history.voltage)

        if len(values) < 2:
            return

        def point(index: int, value: float) -> Tuple[int, int]:
            px = x0 + int(index / (len(values) - 1) * w)

            # -100 mV -> bottom
            # +60 mV -> top
            normalized = (value + 100.0) / 160.0
            py = y0 + h - int(normalized * h)

            return px, py

        points = [
            point(i, v)
            for i, v in enumerate(values)
        ]

        if len(points) >= 2:
            pygame.draw.lines(
                self.screen,
                self.ACTIVE,
                False,
                points,
                2
            )

        # Zero line
        zero_y = y0 + h - int((100.0 / 160.0) * h)

        pygame.draw.line(
            self.screen,
            self.GRID,
            (x0, zero_y),
            (x0 + w, zero_y),
            1
        )

        self.draw_text(
            "MEMBRANE POTENTIAL",
            x0 + 15,
            y0 + 10,
            self.TEXT,
            self.small_font
        )

        self.draw_text(
            "+60 mV",
            x0 + w - 70,
            y0 + 10,
            self.MUTED,
            self.small_font
        )

        self.draw_text(
            "0 mV",
            x0 + w - 60,
            zero_y - 18,
            self.MUTED,
            self.small_font
        )

        self.draw_text(
            "-100 mV",
            x0 + w - 75,
            y0 + h - 20,
            self.MUTED,
            self.small_font
        )

    # --------------------------------------------------------
    # INFORMATION PANEL
    # --------------------------------------------------------

    def draw_info(self) -> None:
        x = 30
        y = 40

        self.draw_text(
            "BIOLOGICAL NEURON SIMULATOR",
            x,
            y,
            self.TEXT,
            self.title_font
        )

        lines = [
            f"Simulation time : {self.neuron.time_ms:8.2f} ms",
            f"Membrane V      : {self.neuron.voltage:8.2f} mV",
            f"Na gate (m)     : {self.neuron.ions.m:8.4f}",
            f"Na gate (h)     : {self.neuron.ions.h:8.4f}",
            f"K gate  (n)     : {self.neuron.ions.n:8.4f}",
            f"I_Na            : {self.neuron.current_na:8.3f}",
            f"I_K             : {self.neuron.current_k:8.3f}",
            f"I_Leak          : {self.neuron.current_l:8.3f}",
            f"I_Excitatory    : {self.neuron.current_exc:8.3f}",
            f"I_Inhibitory    : {self.neuron.current_inh:8.3f}",
            f"Spikes          : {self.neuron.spike_count}",
            f"Firing rate     : {self.neuron.firing_rate_hz():8.2f} Hz",
        ]

        for i, line in enumerate(lines):
            self.draw_text(
                line,
                x,
                y + 42 + i * 19,
                self.MUTED,
                self.small_font
            )

        # Controls
        control_x = 1150

        controls = [
            "CONTROLS",
            "SPACE : strong current pulse",
            "E     : excitatory synapse",
            "I     : inhibitory synapse",
            "T     : toggle periodic stimulation",
            "P     : pause / resume",
            "R     : reset",
            "ESC   : quit",
        ]

        for i, text in enumerate(controls):
            self.draw_text(
                text,
                control_x,
                45 + i * 22,
                self.TEXT if i == 0 else self.MUTED,
                self.font if i == 0 else self.small_font
            )

    # --------------------------------------------------------
    # PERIODIC STIMULATION
    # --------------------------------------------------------

    def update_periodic_stimulation(self, dt_ms: float) -> None:
        if not self.periodic_stimulus:
            return

        self.periodic_timer += dt_ms

        if self.periodic_timer >= self.periodic_interval:
            self.periodic_timer = 0.0
            self.neuron.stimulate_current(10.0)

        else:
            # Current is turned off between pulses.
            self.neuron.stimulate_current(0.0)

    # --------------------------------------------------------
    # EVENT LOOP
    # --------------------------------------------------------

    def handle_events(self) -> None:
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:

                if event.key == pygame.K_ESCAPE:
                    self.running = False

                elif event.key == pygame.K_SPACE:
                    # Briefly inject current.
                    self.manual_stimulus = True
                    self.neuron.stimulate_current(15.0)

                elif event.key == pygame.K_e:
                    self.neuron.stimulate_excitatory(1.0)

                elif event.key == pygame.K_i:
                    self.neuron.stimulate_inhibitory(1.0)

                elif event.key == pygame.K_t:
                    self.periodic_stimulus = not self.periodic_stimulus
                    self.periodic_timer = 0.0

                elif event.key == pygame.K_p:
                    self.paused = not self.paused

                elif event.key == pygame.K_r:
                    self.neuron.reset()
                    self.history.clear()

            elif event.type == pygame.KEYUP:

                if event.key == pygame.K_SPACE:
                    self.manual_stimulus = False
                    if not self.periodic_stimulus:
                        self.neuron.stimulate_current(0.0)

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    def run(self) -> None:
        accumulator = 0.0

        while self.running:
            self.handle_events()

            if not self.paused:
                # Periodic stimulation
                self.update_periodic_stimulation(
                    DT_MS * STEPS_PER_FRAME
                )

                # Multiple tiny physiological steps per frame
                for _ in range(STEPS_PER_FRAME):
                    self.neuron.step(DT_MS)

                self.history.append(self.neuron)

            # Rendering
            self.screen.fill(self.BG)

            self.draw_neuron()
            self.draw_graph()
            self.draw_info()

            if self.paused:
                self.draw_text(
                    "PAUSED",
                    WIDTH // 2 - 50,
                    570,
                    self.ACTIVE,
                    self.title_font
                )

            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()


# ============================================================
# 9. EXPERIMENTS
# ============================================================

def run_single_neuron_experiment() -> None:
    """
    نقطه ورود اصلی.

    اینجا در آینده می‌توانیم experimentهای استاندارد neuroscience
    را اضافه کنیم:

        * Current clamp
        * Voltage clamp
        * Frequency-current curve
        * Refractory period measurement
        * Synaptic integration
        * EPSP/IPSP summation
        * Noise stimulation
        * STDP
    """

    neuron = BiologicalNeuron()
    visualizer = Visualizer(neuron)
    visualizer.run()


# ============================================================
# 10. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_single_neuron_experiment()
