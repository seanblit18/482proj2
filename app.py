import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Define the ODE system
def hpa_axis_model(y, t, k1_base, k2, k3, k4, eta, Kd, A):
    CRH, ACTH, CORT = y
    
    # Circadian modulation of k1
    k1 = k1_base * (1 + A * np.cos(2 * np.pi * t / 24))
    
    # ODEs
    dCRH_dt = k1 * (1 - eta * CORT / (Kd + CORT)) - k2 * CRH
    dACTH_dt = k2 * CRH - k3 * ACTH
    dCORT_dt = k3 * ACTH - k4 * CORT
    
    return [dCRH_dt, dACTH_dt, dCORT_dt]

# Parameters (tuned to match the figure)
k1_base = 0.2    # Base production rate of CRH
k2 = 0.5         # CRH degradation and ACTH production rate
k3 = 0.4         # ACTH degradation and CORT production rate
k4 = 0.3         # CORT degradation rate
eta = 0.8        # Feedback strength
Kd = 0.2         # Dissociation constant for CORT feedback
A = 0.3          # Amplitude of circadian oscillation in k1

# Time points (0 to 72 hours)
t = np.linspace(0, 72, 1000)

# Initial conditions: [CRH, ACTH, CORT]
y0 = [0.15, 0.1, 0.1]

# Solve the ODEs
solution = odeint(hpa_axis_model, y0, t, args=(k1_base, k2, k3, k4, eta, Kd, A))

# Extract CRH for plotting
CRH = solution[:, 0]

# Plot the CRH timecourse
plt.figure(figsize=(8, 4))
plt.plot(t, CRH, label='CRH', color='black')
plt.xlabel('Time (hours)')
plt.ylabel('Cortisol')
plt.title('Computational simulation of the HPA axis system')
plt.grid(True)
plt.ylim(0.10, 0.25)
plt.xticks(np.arange(0, 73, 24))
plt.show()