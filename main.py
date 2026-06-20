import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
import cv2
import time
from settings import (
    center_freq,
    fractional_bw,
    sector_angle_deg,
    curvature_radius_mm,
)

def anisotropic_diffusion(img, niter=5, kappa=30, gamma=0.1):
    """
    Perona-Malik anisotropic diffusion for edge-preserving speckle reduction.
    """
    img = img.astype(np.float32)
    for _ in range(niter):
        nablaN = np.roll(img, -1, axis=0) - img
        nablaS = np.roll(img,  1, axis=0) - img
        nablaE = np.roll(img, -1, axis=1) - img
        nablaW = np.roll(img,  1, axis=1) - img

        cN = np.exp(-(nablaN / kappa) ** 2)
        cS = np.exp(-(nablaS / kappa) ** 2)
        cE = np.exp(-(nablaE / kappa) ** 2)
        cW = np.exp(-(nablaW / kappa) ** 2)

        img += gamma * (cN * nablaN + cS * nablaS + cE * nablaE + cW * nablaW)
    return img


def load_rf_csv(filepath):
    """
    Load an RF data file into a numeric (num_beams, num_samples) array.

    Handles two layouts transparently:
      * The probe-app captures (``*RF.csv`` / ``*_rawRF.csv``): a header row
        ``Line,Sample_0,Sample_1,...`` plus a leading integer ``Line`` index
        column.  Both are stripped here.
      * Plain numeric dumps such as ``ae2RF.txt``: no header, no index column.

    (The old code called ``np.genfromtxt`` directly, which silently turned the
    header row into a row of zeros and treated the ``Line`` index as sample 0.)
    """
    with open(filepath, "r") as f:
        first_line = f.readline()
    has_header = any(ch.isalpha() for ch in first_line)

    data = np.genfromtxt(
        filepath,
        delimiter=",",
        skip_header=1 if has_header else 0,
        filling_values=0.0,
    )
    if data.ndim == 1:
        data = data[np.newaxis, :]

    # Drop a leading integer index column (0, 1, 2, ...) if one is present.
    if has_header and data.shape[1] > 1:
        col0 = data[:, 0]
        if np.allclose(col0, np.arange(len(col0))):
            data = data[:, 1:]

    return data


def reconstruct_bmode_array(
    data,
    *,
    fast=True,
    center_freq=3e6,          # Probe center frequency in Hz
    fractional_bw=0.6,        # Fractional bandwidth of the bandpass filter
    sector_angle_deg=70,
    curvature_radius_mm=30,
    dynamic_range=60,
    tgc_exponent=1.5,
    output_resolution=None,
    fs=30e6,
    c=1540,
):
    """
    Convert a raw RF array into an 8-bit grayscale convex B-mode image.

    ``data`` is the RF echo matrix; orientation is normalized internally to
    (num_beams, num_samples).  Returns a ``uint8`` HxW grayscale image.

    ``fast=True`` (real-time path): renders at a lower internal resolution and
    skips the expensive Perona-Malik diffusion pass.  ``fast=False`` reproduces
    the original offline pipeline used by the CLI.
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[1] < data.shape[0]:
        data = data.T  # Ensure shape is (num_beams, num_samples)

    num_beams, num_samples = data.shape

    if output_resolution is None:
        output_resolution = (512, 512) if fast else (800, 800)

    # -------------------------
    # Bandpass filter around probe center frequency
    # -------------------------
    low  = (center_freq * (1 - fractional_bw / 2)) / (fs / 2)
    high = (center_freq * (1 + fractional_bw / 2)) / (fs / 2)
    low  = np.clip(low,  1e-4, 0.9999)
    high = np.clip(high, 1e-4, 0.9999)
    b, a = butter(4, [low, high], btype="band")
    data = filtfilt(b, a, data, axis=1)

    analytic = hilbert(data, axis=1)   # axis=1 = fast-time (samples) axis
    envelope = np.abs(analytic)

    depths_idx = np.arange(num_samples)
    tgc = (depths_idx / depths_idx[-1]) ** tgc_exponent + 1.0
    envelope *= tgc[np.newaxis, :]
    envelope /= np.max(envelope) + 1e-12

    bmode = 20 * np.log10(envelope + 1e-6)
    bmode = np.clip(bmode, -dynamic_range, 0.0)
    bmode = (bmode + dynamic_range) / dynamic_range  # Normalize to [0, 1]
    bmode = gaussian_filter(bmode, sigma=0.5)

    # -------------------------
    # Convex scan conversion (polar -> Cartesian)
    # -------------------------
    sector_angle = np.deg2rad(sector_angle_deg)
    dr = c / (2 * fs)
    depths_m = np.arange(num_samples) * dr
    Rcurv = curvature_radius_mm / 1000.0

    theta = np.linspace(-sector_angle / 2, sector_angle / 2, num_beams)

    max_depth = depths_m[-1]
    x_max = (Rcurv + max_depth) * np.sin(sector_angle / 2)

    x_lin = np.linspace(-x_max, x_max, output_resolution[1])
    y_lin = np.linspace(0, max_depth, output_resolution[0])
    Xg, Yg = np.meshgrid(x_lin, y_lin)

    r_grid     = np.sqrt(Xg ** 2 + (Yg + Rcurv) ** 2)
    theta_grid = np.arctan2(Xg, Yg + Rcurv)

    interp = RegularGridInterpolator(
        (theta, depths_m),
        bmode,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    query_pts = np.column_stack((theta_grid.ravel(), r_grid.ravel() - Rcurv))
    cart = interp(query_pts).reshape(output_resolution)

    mask = (
        (np.abs(theta_grid) <= sector_angle / 2) &
        (r_grid >= Rcurv) &
        (r_grid <= Rcurv + max_depth)
    )
    cart_masked = np.where(mask, cart, 0.0)

    img8 = np.clip(cart_masked * 255, 0, 255).astype(np.uint8)

    # -------------------------
    # Post-processing.  The Perona-Malik diffusion pass is the slowest step,
    # so it is skipped in the real-time (fast) path.
    # -------------------------
    if fast:
        img_proc = img8
    else:
        img_proc = np.clip(anisotropic_diffusion(img8, niter=5, kappa=30, gamma=0.1),
                           0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_proc)

    kernel_sharp = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_sharp = cv2.filter2D(img_clahe, -1, kernel_sharp)

    # -------------------------
    # Aspect-correct final resize (width preserved, height from physical ratio)
    # -------------------------
    width_cm  = abs(x_lin[-1] - x_lin[0]) * 100
    height_cm = abs(y_lin[-1] - y_lin[0]) * 100
    aspect_ratio = width_cm / height_cm if height_cm else 1.0

    output_width  = output_resolution[1]
    output_height = max(1, int(output_width / aspect_ratio))

    img_8bit = cv2.resize(img_sharp, (output_width, output_height),
                          interpolation=cv2.INTER_LINEAR)
    return img_8bit


def rf_to_convex_bmode_fixed(
    filepath,
    center_freq=3e6,
    fractional_bw=0.6,
    sector_angle_deg=70,
    curvature_radius_mm=30,
    dynamic_range=60,
    tgc_exponent=1.5,
    output_resolution=(800, 800),
    fs=30e6,
    c=1540,
):
    """
    CLI / offline wrapper: read an RF file, run the full-quality reconstruction,
    and write the grayscale + JET-heatmap PNGs (unchanged outputs).
    """
    data = load_rf_csv(filepath)
    num_beams, num_samples = (data.shape if data.ndim == 2 else (1, data.shape[0]))
    print(f"RF data shape: {data.shape}  ->  {num_beams} beams x {num_samples} samples")

    img_8bit = reconstruct_bmode_array(
        data,
        fast=False,
        center_freq=center_freq,
        fractional_bw=fractional_bw,
        sector_angle_deg=sector_angle_deg,
        curvature_radius_mm=curvature_radius_mm,
        dynamic_range=dynamic_range,
        tgc_exponent=tgc_exponent,
        output_resolution=output_resolution,
        fs=fs,
        c=c,
    )

    img_color = cv2.applyColorMap(img_8bit, cv2.COLORMAP_JET)
    cv2.imwrite("Fixed_Convex_B-mode_Reconstruction.png", img_8bit)
    cv2.imwrite("Fixed_Convex_B-mode_Reconstruction_heatmap.png", img_color)

    return img_8bit.astype(np.float32) / 255.0


if __name__ == "__main__":
    start_time = time.perf_counter()
    img = rf_to_convex_bmode_fixed(
        "ae2RF.txt",
        center_freq=center_freq,           # <-- SET THIS to your probe's center frequency
        fractional_bw=fractional_bw,
        sector_angle_deg=sector_angle_deg,
        curvature_radius_mm=curvature_radius_mm,
    )
    end_time = time.perf_counter()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")
