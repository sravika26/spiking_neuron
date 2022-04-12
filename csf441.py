from tkinter import font

import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    """instantiating Neuron object with conductance, eqm potentials for Na,K,Leak channels"""
    def __init__(self, sim):
        self.sim = sim
        self.creating_arrays(0, 0)
        pass

    def creating_arrays(self, counter, deltaT):
        """arrays to store values"""
        self.I_Na = np.empty(counter)
        self.I_K = np.empty(counter)
        self.I_L = np.empty(counter)
        self.V_m = np.empty(counter)

        self.n_state = np.empty(counter)
        self.m_state = np.empty(counter)
        self.h_state = np.empty(counter)

        self.time = np.arange(counter) * deltaT

    def run_sim(self, waveform, step_size):
        assert isinstance(waveform, np.ndarray)
        self.creating_arrays(len(waveform), step_size)

        """loop that runs 20000 times to update values"""
        for i in range(len(waveform)):
            self.sim.iteration(waveform[i], step_size)
            self.I_Na[i] = self.sim.I_Na
            self.I_K[i] = self.sim.I_K
            self.I_L[i] = self.sim.I_L
            self.V_m[i] = self.sim.V_m

            self.m_state[i] = self.sim.m.state
            self.h_state[i] = self.sim.h.state
            self.n_state[i] = self.sim.n.state


class Hodkin_Huxley_model:
    """conductance-based model to calculate membrane potential(Vm) at different timesteps"""

    class gates:
        state, a, b = 0, 0, 0

        def f_infty(self):
            """to calculate const f_infinity for ion channels"""
            self.state = self.a / (self.a + self.b)

        def diff_eqns(self, deltaT):
            a_state = self.a * (1 - self.state)
            b_state = self.b * self.state
            self.state += deltaT * (a_state - b_state)

    """values taken directly from original HH paper(1952)"""
    g_Na, g_K, g_L = 120, 36, 0.3
    E_Na, E_K, E_L = 115, -12, 10.6
    m, n, h = gates(), gates(), gates()
    Cm = 1

    def __init__(self, V_init=0):
        self.V_m = V_init
        self.update_const(V_init)
        self.m.f_infty()
        self.h.f_infty()
        self.n.f_infty()

    def update_const(self, V_m):
        """updating time const of ion channels based on  Vm"""
        self.m.a = 0.1 * ((25 - V_m) / (np.exp((25 - V_m) / 10) - 1))
        self.m.b = 4 * np.exp(-V_m / 18)
        self.h.a = 0.07 * np.exp(-V_m / 20)
        self.h.b = 1 / (np.exp((30 - V_m) / 10) + 1)
        self.n.a = 0.01 * ((10 - V_m) / (np.exp((10 - V_m) / 10) - 1))
        self.n.b = 0.125 * np.exp(-V_m / 80)

    def update_curr(self, I_inj, deltaT):
        """calculate channel currents and Vm using time const"""
        self.I_Na = np.power(self.m.state, 3) * self.g_Na * \
                   self.h.state * (self.V_m - self.E_Na)
        self.I_K = np.power(self.n.state, 4) * self.g_K * (self.V_m - self.E_K)
        self.I_L = self.g_L * (self.V_m - self.E_L)
        Isum = I_inj - self.I_Na - self.I_K - self.I_L
        self.V_m += deltaT * Isum / self.Cm

    def iteration(self, I_inj, deltaT):
        """runs all the 3 methods to update all values on every increment of deltaT"""
        self.update_const(self.V_m)
        self.update_curr(I_inj, deltaT)
        self.m.diff_eqns(deltaT)
        self.h.diff_eqns(deltaT)
        self.n.diff_eqns(deltaT)

sim = Hodkin_Huxley_model()
sim.g_Na = 120
sim.g_K = 36
sim.E_K = -12

"""20000 timesteps with deltaT=0.01mS increment to plot simulation over 200mS. current injected in square pulse between timesteps 5000 and 15000 (period of 100mS). These values adjusted to obtain rheobase and spiking frequency """
inj = np.zeros(20000)
inj[5000:15000] = 2.24

# Plotting
simulate = Neuron(sim)
simulate.run_sim(waveform=inj, step_size=0.01)
plt.figure(figsize=(10, 8))

# V-t plot
ax1 = plt.subplot(411)
ax1.plot(simulate.time, simulate.V_m - 70)
ax1.set_ylabel("Membrane Potential(mV)")
ax1.set_xlabel("Time(in ms)")
ax1.set_title("Hodgkin-Huxley Model")

# injected square pulse
ax2 = plt.subplot(412)
ax2.plot(simulate.time, inj)
ax2.set_ylabel("Current Injected (µA/cm²)")
ax2.set_xlabel("Time(in ms)")

plt.tight_layout()
plt.show()