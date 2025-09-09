# %% [markdown]
# # Header 

# %% [markdown]
# ## Information
# 
# **Title**: Generate a Zona Plate Using the Holographic Principle
# 
# **Project**: Zone Plate Simulations
# 
# **Version**: v1, January 2025
# 
# **Authors**: 
# - Wei "Francis" He (UCSD/LBNL)
# - Dr. Antoine Islegen-Wojdyla (LBNL)
# 
# **Contact**: wehe@ucsd.edu / francisho@lbl.gov, awojdyla@lbl.gov

# %% [markdown]
# ## Overview
# 
# In this notebook, we want to generate Fresnel zoneplates with different functions (focus, dual beam, triple beam, orbital angular momentum if we're crazy) using Fourier optics and coherent light propagation

# %% [markdown]
# # Set up and Configuration

# %% [markdown]
# ## Import required libraries and functions

# %%
import numpy as np
import matplotlib.pyplot as plt
import time

# %%
from monoplus import gaussfunc, propHF, propTF

# Monoplus is a wavefront propagation package that was implemented by Paola Luna (SULI student at Berkeley Lab at the time.)
# Base repo (private): https://github.com/paola-luna24/Monoplus-project
# Also available here: https://github.com/awojdyla/Monoplus-project

# Note: We modified the original code for our purpose. 

# %% [markdown]
# ## Parameter initialization

# %%
# Grids for computation
x_m  = np.linspace(-20e-3, 20e-3, 40000)
L_m = x_m[-1]-x_m[0]

# Optical parameters
wavelength_m = 632.8e-9     # units: meters (632.8 nm, He-Ne laser)

# Zone plate parameters
diameter_m = 5e-3     # zoneplate diameter


# %% [markdown]
# ## Define intitial beams

# %%
# Define the plane wave

E0_plane = np.sqrt(gaussfunc(x_m, 0, diameter_m/2))   # Gaussian beam

plt.figure(figsize=(4, 4))
plt.plot(x_m*1e3, np.abs(E0_plane)**2)  
plt.xlabel('Position (mm)')
plt.ylabel('Intensity')
plt.show()  

# %%
# Define the point source, Gaussian beam

size_rms_m = 10e-6 
E0_sph = np.sqrt(gaussfunc(x_m, 0, size_rms_m))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

ax1.plot(x_m*1e3, np.abs(E0_sph)**2)
ax1.set_xlabel('Position (mm)')
ax1.set_ylabel('Intensity')

ax2.plot(x_m*1e6, np.abs(E0_sph)**2)
ax2.set_xlabel('Position (um)')
ax2.set_ylabel('Intensity')
ax2.set_xlim(-200, 200)

plt.tight_layout()
plt.show()


# %%
# Define dual beams

size_rms_m = 10e-6      # sigma of the Gaussian beam
E0_dual = np.sqrt(gaussfunc(x_m, -100e-6, size_rms_m)) + np.sqrt(gaussfunc(x_m, +100e-6, size_rms_m))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

ax1.plot(x_m*1e3, np.abs(E0_dual)**2)
ax1.set_xlabel('Position (mm)')
ax1.set_ylabel('Intensity')

ax2.plot(x_m*1e6, np.abs(E0_dual)**2)
ax2.set_xlabel('Position (um)')
ax2.set_ylabel('Intensity')
ax2.set_xlim(-200, 200)

plt.tight_layout()
plt.show()


# %% [markdown]
# # Analysis Codes

# %% [markdown]
# ## Zone plate as a hologram of a plane wave and a spherical wave
# 
# A Fresnel zone can be generated as interference pattern from a plane wave and a spherical wave (from a point source), and the distance between the point source and the plane of interference is the focal length of the zone plate. 

# %%
# Define the propagation distance, which will be the same as the focal length of the zone plate
z_m = 0.5   # in meters

# %% [markdown]
# ### Phase from spherical wave
# 
# To begin with, let's only consider the phase of the wave from the point source, propagated to the interested plane. 

# %%
# Propagate the spheritcal wave
Ezp_sph = propTF(E0_sph, L_m, wavelength_m, z_m)

# Phase and from which the binary pattern generated
ZP_gen = np.angle(Ezp_sph)
ZP = ZP_gen > 0

# Plot the results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

ax1.plot(x_m*1e3, np.abs(Ezp_sph)**2)
ax1.set_xlabel('Position (mm)')
ax1.set_ylabel('Intensity')
ax1.set_xlim(-6, 6)
ax1.set_title('Intensity of the Propagated Spherical Wave')

ax2.plot(x_m*1e3, ZP_gen)
ax2.set_xlabel('Position (mm)')
ax2.set_ylabel('Phase (radians)')
ax2.set_xlim(-6, 6)
ax2.set_title('Phase of the Propagated Spherical Wave')

ax3.plot(x_m*1e3, ZP)
ax3.set_xlabel('Position (mm)')
ax3.set_ylabel('Binary Value')
ax3.set_xlim(-6, 6)
ax3.set_title('ZP Binary Pattern')

plt.tight_layout()
plt.show()


# %%
# Test the zone plate
 
# Exit beam from the zone plate
E_ext = E0_plane * ZP   # the incident beam assumed to be a plane wave

# Propagate the exit beam to the focal plane
E_foc = propTF(E_ext, L_m, wavelength_m, z_m)

# Plot the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

ax1.plot(x_m*1e3, np.abs(E_ext)**2)
ax1.set_xlabel('Position (mm)')
ax1.set_ylabel('Intensity')
ax1.set_xlim(-8, 8)
ax1.set_title('Intensity of the Exit Beam')

ax2.plot(x_m*1e3, np.abs(E_foc)**2)
ax2.set_xlabel('Position (mm)')
ax2.set_ylabel('Intensity')
ax2.set_xlim(-8, 8)
ax2.set_title('Intensity of the Beam at the Focal Plane')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Phase from spherical wave + plane wave
# 
# Should be similar as the previous case. 

# %%
# Propagate the plane wave
Ezp_plane = propTF(E0_plane, L_m, wavelength_m, z_m)

# Phase of the interferennce pattern, and from which the binary pattern generated
ZP_gen = np.angle(Ezp_sph+Ezp_plane)
ZP2 = ZP_gen > 0

# Plot the results

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

ax1.plot(x_m*1e3, np.abs(Ezp_plane)**2)
ax1.set_xlabel('Position (mm)')
ax1.set_ylabel('Intensity')
ax1.set_xlim(-6, 6)
ax1.set_title('Intensity of the Propagated Spherical Wave')

ax2.plot(x_m*1e3, ZP_gen)
ax2.set_xlabel('Position (mm)')
ax2.set_ylabel('Phase (radians)')
# ax2.set_xlim(-6, 6)
ax2.set_title('Phase of the Propagated Spherical Wave')

ax3.plot(x_m*1e3, ZP2)
ax3.set_xlabel('Position (mm)')
ax3.set_ylabel('Binary Value')
ax3.set_xlim(-6, 6)
ax3.set_title('ZP Binary Pattern')

plt.tight_layout()
plt.show()


# %%
# Test the zone plate
 
# Exit beam from the zone plate
# E_ext2 = Ezp_plane * ZP2
E_ext2 = E0_plane * ZP2

# Propagate the exit beam to the focal plane
E_foc2 = propTF(E_ext, L_m, wavelength_m, z_m)

# Plot the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

ax1.plot(x_m*1e3, np.abs(E_ext2)**2)
ax1.set_xlabel('Position (mm)')
ax1.set_ylabel('Intensity')
ax1.set_xlim(-8, 8)
ax1.set_title('Intensity of the Exit Beam')

ax2.plot(x_m*1e3, np.abs(E_foc2)**2)
ax2.set_xlabel('Position (mm)')
ax2.set_ylabel('Intensity')
ax2.set_xlim(-8, 8)
ax2.set_title('Intensity of the Beam at the Focal Plane')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Phase from two point sources
# 
# Now we consider the interference between two point sources, both on the optical axis. The relation between their distances to the ZP plane and the ZP's focal length is: 
# 
# `1/f = 1/z1 - 1/z2`

# %%
# Propagate the spheritcal wave for a longer distance
z2_m = 0.25
Ezp_sph2 = propTF(E0_sph, L_m, wavelength_m, z2_m)

# Phase and from which the binary pattern generated
ZP_gen = np.angle(Ezp_sph+Ezp_sph2)
ZP3 = ZP_gen > 0

# Plot the results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

ax1.plot(x_m*1e3, np.abs(Ezp_sph2)**2)
ax1.set_xlabel('Position (mm)')
ax1.set_ylabel('Intensity')
ax1.set_xlim(-6, 6)
ax1.set_title('Intensity of the Propagated Spherical Wave')

ax2.plot(x_m*1e3, ZP_gen)
ax2.set_xlabel('Position (mm)')
ax2.set_ylabel('Phase (radians)')
ax2.set_xlim(-6, 6)
ax2.set_title('Phase of the Propagated Spherical Wave')

ax3.plot(x_m*1e3, ZP3)
ax3.set_xlabel('Position (mm)')
ax3.set_ylabel('Binary Value')
ax3.set_xlim(-6, 6)
ax3.set_title('ZP Binary Pattern')

plt.tight_layout()
plt.show()

# %%
# Test the zone plate
 
# Exit beam from the zone plate
E_ext3 = E0_plane * ZP3

# Propagate the exit beam to the focal plane
f_m = 1/(-1/z_m+1/z2_m)
print(f"Focal length: {f_m:.2f} m")
E_foc3 = propTF(E_ext, L_m, wavelength_m, f_m)

# Plot the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

ax1.plot(x_m*1e3, np.abs(E_ext3)**2)
ax1.set_xlabel('Position (mm)')
ax1.set_ylabel('Intensity')
ax1.set_xlim(-8, 8)
ax1.set_title('Intensity of the Exit Beam')

ax2.plot(x_m*1e3, np.abs(E_foc3)**2)
ax2.set_xlabel('Position (mm)')
ax2.set_ylabel('Intensity')
ax2.set_xlim(-8, 8)
ax2.set_title('Intensity of the Beam at the Focal Plane')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Phase from a dual-beam source
# 
# So far we have demonstrated how Fresnel zone plates can be generated from the interference pattern from sources. Let's now move to our a more interesting case: what if we want to design a ZP that can focus the incident beam into two focal spots. 

# %%
# Propagate the spheritcal wave
Ezp_dual = propTF(E0_dual, L_m, wavelength_m, z_m)

# Phase and from which the binary pattern generated
ZP_gen = np.angle(Ezp_dual)
ZP4 = ZP_gen > 0

# Plot the results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

ax1.plot(x_m*1e3, np.abs(Ezp_dual)**2)
ax1.set_xlabel('Position (mm)')
ax1.set_ylabel('Intensity')
ax1.set_xlim(-6, 6)
ax1.set_title('Intensity of the Propagated Spherical Wave')

ax2.plot(x_m*1e3, ZP_gen)
ax2.set_xlabel('Position (mm)')
ax2.set_ylabel('Phase (radians)')
ax2.set_xlim(-6, 6)
ax2.set_title('Phase of the Propagated Spherical Wave')

ax3.plot(x_m*1e3, ZP4)
ax3.set_xlabel('Position (mm)')
ax3.set_ylabel('Binary Value')
ax3.set_xlim(-6, 6)
ax3.set_title('ZP Binary Pattern')

plt.tight_layout()
plt.show()


# %%
# Test the zone plate
 
# Exit beam from the zone plate
E_ext4 = E0_plane * ZP4   # the incident beam assumed to be a plane wave

# Propagate the exit beam to the focal plane
E_foc4 = propTF(E_ext, L_m, wavelength_m, z_m)

# Plot the results
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

ax1.plot(x_m*1e3, np.abs(E_ext4)**2)
ax1.set_xlabel('Position (mm)')
ax1.set_ylabel('Intensity')
ax1.set_xlim(-8, 8)
ax1.set_title('Intensity of the Exit Beam')

ax2.plot(x_m*1e3, np.abs(E_foc4)**2)
ax2.set_xlabel('Position (mm)')
ax2.set_ylabel('Intensity')
ax2.set_xlim(-8, 8)
ax2.set_title('Intensity of the Beam at the Focal Plane')

ax3.plot(x_m*1e6, np.abs(E_foc4)**2)
ax3.set_xlabel('Position (um)')
ax3.set_ylabel('Intensity')
ax3.set_xlim(-1000, 1000)
ax3.set_title('Intensity of the Beam at the Focal Plane (Zoomed)')

plt.tight_layout()
plt.show()

# %%
# Create single beam (original ZP) propagation
Nz = 1001  # number of z positions to calculate
zs_m = np.linspace(0, 2, Nz)*z_m  # range from 0 to focal length

time_start = time.time()

# Propagate to different z positions
Ez_single = np.zeros((len(x_m), Nz), dtype=complex)
for i_z in range(Nz):
    Ez_single[:, i_z] = propTF(E_ext4, L_m, wavelength_m, zs_m[i_z])

time_end = time.time()
time_spent = time_end - time_start
print(f"Time spent: {time_spent:.2f} seconds")

# Create intensity map
intensity_map = np.abs(Ez_single)**2
normalized_map = (intensity_map/np.max(intensity_map)) **0.1 # use 0.05 power for better visualization


# %%
# Plot propagation cross-section
plt.figure(figsize=(10, 4))
extent = (zs_m[0]*1e3, zs_m[-1]*1e3, x_m[0]*1e3, x_m[-1]*1e3)
           # z in mm, x in μm

plt.imshow(normalized_map, extent=extent, 
          aspect='auto',  # let matplotlib handle the aspect ratio
          origin='lower',
          cmap='viridis')
plt.xlabel('Distance from ZP [mm]')
plt.ylabel('Radial position [mm]')
plt.title('Single Beam Propagation (Zoomed)')
plt.colorbar(label='Normalized Intensity (0.05 power)')
# plt.grid(True)
# plt.ylim(-30, 30)   # Zoom in
plt.show()

# %% [markdown]
# We need a central stop?
# 

# %%


# %%


# %%


# %%


# %% [markdown]
# # Old codes

# %% [markdown]
# ## Generate an image and a desired object

# %%
# image 
def gaussfunc(x, mean_x, sigma_x):
    gaussF = np.exp(-((x-mean_x)/(np.sqrt(2)*sigma_x))**2)
    return gaussF
x_m  = np.linspace(-20e-3, 20e-3, 40000)
size_rms_m = 10e-6

E0 = np.sqrt(gaussfunc(x_m, 0, size_rms_m))
plt.plot(x_m*1e3, np.abs(E0)**2)  
plt.xlabel('position (mm)')
plt.ylabel('Intensity')
plt.xlim(-.02,.02)
plt.show()  

# %% [markdown]
# ## propagate the beam
# 

# %%
L_m = x_m[-1]-x_m[0]
z_m = 0.5
Ez = propTF(E0, L_m, lambda_m, z_m)
plt.plot(x_m*1e3, np.abs(Ez)**2)
# plt.xlim(-.1,.1)

# %%
# ZP_gen = np.angle(Ez + np.conj(Ez))
# ZP_gen = (Ez + np.conj(Ez))>0
ZP_gen = np.angle(Ez)
ZP = ZP_gen > 0
plt.plot(x_m*1e3, ZP)
plt.xlim([-5,5])


# %%
ZP_gen = np.angle(Ez + np.conj(Ez))
ZP2 = ZP_gen > 0

plt.plot(x_m*1e3, ZP2)
plt.xlim([-5,5])

# %%
E_parallel = np.sqrt(gaussfunc(x_m, 0, L_m/10))

Ef3 = propTF(E_parallel*ZP2, L_m, lambda_m, +z_m/2)

plt.plot(x_m*1e3, np.abs(Ef3)**2)

# %%
#ZP_gen = np.abs(Ez + np.conj(Ez))**2
ZP_gen = np.angle(Ez + np.conj(Ez))
ZP = ZP_gen>np.pi/2
plt.plot(x_m*1e3, ZP)
plt.xlim([-8,8])


# %%
Ezp = Ez*ZP
plt.plot(x_m*1e3, np.abs(Ezp)**2)
plt.xlim([-8,8])

# %%
Ef = propTF(Ezp, L_m, lambda_m, +z_m)
plt.plot(x_m*1e3, np.abs(Ef)**2)
# plt.xlim(-3,3)
plt.show()

# %%
E_parallel = np.sqrt(gaussfunc(x_m, 0, L_m/10))

Ef2 = propTF(E_parallel*ZP, L_m, lambda_m, +z_m)

plt.plot(x_m*1e3, np.abs(Ef2)**2)

# %%
pl

# %%


# %%


# %%


# %%
# Enable debug mode to see calculations
# _DEBUG = True

# %%


# %%


# %%
# Parameters

foc_len_m = 100e-3  
lambda_m = 625e-9  
N = 50   # number of zones


# %%


# %%


# %%


# %%


# %%


# %%


# %%



