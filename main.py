import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # For 3D plots
from scipy.signal import correlate, find_peaks

# Define audio file paths
audio_files = {
    "Suave": "./content/suave.wav",
    "Medio": "./content/medio.wav",
    "Fuerte": "./content/fuerte.wav",
}


# Function to extract envelopes
def hl_envelopes_idx(s, dmin=1, dmax=1):
    """Finds high and low envelopes of a signal."""
    # Local minima
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    # Local maxima
    lmax = (np.diff(np.sign(np.diff(s))) < 0).nonzero()[0] + 1

    # Global minima within dmin-sized chunks
    lmin = lmin[
        [i + np.argmin(s[lmin[i : i + dmin]]) for i in range(0, len(lmin), dmin)]
    ]
    # Global maxima within dmax-sized chunks
    lmax = lmax[
        [i + np.argmax(s[lmax[i : i + dmax]]) for i in range(0, len(lmax), dmax)]
    ]

    return lmin, lmax


# Helper function to compute FFT
def compute_fft(signal, sr):
    """Computes the FFT of a signal and returns the amplitude spectrum and frequencies."""
    n = len(signal)
    T = 1 / sr
    fft_signal = np.fft.fft(signal)
    magnitude = np.abs(fft_signal)[: n // 2]
    frequencies = np.fft.fftfreq(n, T)[: n // 2]
    return magnitude, frequencies


def fft_correlate(signal1, signal2, sr):
    """
    Performs cross-correlation of the Fourier Transforms of two signals.
    """
    # Compute Fourier Transforms
    # fft1, _ = compute_fft(signal1, sr)
    # fft2, _ = compute_fft(signal2, sr)

    fft1 = np.fft.fft(signal1)
    fft2 = np.fft.fft(signal2)

    # Normalize Fourier Transforms
    fft1_norm = np.abs(fft1) / np.linalg.norm(fft1)
    fft2_norm = np.abs(fft2) / np.linalg.norm(fft2)
    # fft1_norm = fft1
    # fft2_norm = fft2
    # Compute cross-correlation in frequency domain
    corr = correlate(fft1_norm, fft2_norm, mode="full")

    # Normalize the cross-correlation result
    corr_norm = np.abs(corr) / np.linalg.norm(corr)
    corr_norm /= np.max(corr_norm)

    # Frequency axis for plotting
    n = len(corr_norm)  # Total length of the correlation array
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1 / (sr * 2)))  # Frequency axis in Hz

    # return corr_norm, freqs
    return corr_norm, freqs


# Function to plot the envelope and FFT side-by-side
def process_signal(path, name):
    """Processes and visualizes signal data with envelopes and FFT."""
    signal, sr = librosa.load(path, sr=None)
    length = len(signal) / sr
    time = np.linspace(0.0, length, len(signal))

    # Calculate the envelopes
    lmin_idx, lmax_idx = hl_envelopes_idx(signal, dmin=32, dmax=32)
    low_envelope = np.interp(time, time[lmin_idx], signal[lmin_idx])
    high_envelope = np.interp(time, time[lmax_idx], signal[lmax_idx])

    # Compute FFT
    fft_magnitude, fft_frequencies = compute_fft(signal, sr)

    # Calculate RMS
    rms = np.sqrt(np.mean(signal**2))

    # Plot envelope and FFT side-by-side
    plt.style.use("seaborn-v0_8")
    fig, axs = plt.subplots(1, 2, figsize=(18, 5), gridspec_kw={"width_ratios": [1, 2]})
    fig.suptitle(f"{name}", fontsize=24)

    # Envelope plot
    axs[0].plot(time, signal, label="Señal", alpha=0.6, color="blue")
    axs[0].plot(time, low_envelope, label="Envolvente Inferior", color="green")
    axs[0].plot(time, high_envelope, label="Envolvente Superior", color="red")
    axs[0].set_title(f"Envolvente señal '{name}' | RMS = {rms:.3f}")
    axs[0].set_xlabel("Tiempo [s]")
    axs[0].set_ylabel("Amplitud")
    axs[0].set_xlim(0, 2.25)
    axs[0].set_ylim(-0.4, 0.4)
    axs[0].legend()

    # FFT plot
    axs[1].plot(fft_frequencies, fft_magnitude, label="Magnitud FFT", color="r")
    axs[1].set_title("Transformada de Fourier")
    axs[1].set_xlabel("Frecuencia [Hz]")
    axs[1].set_ylabel("Magnitud")
    axs[1].set_xlim(0, 4000)
    axs[1].set_ylim(0, 1100)
    axs[1].legend()

    plt.tight_layout()
    plt.show()


# Function to plot the spectrogram
def process_spectrogram(name, audio_path):
    """Plots the spectrogram for an audio file."""
    y, sr = librosa.load(audio_path, sr=None)

    # Compute the STFT
    n_fft = 2048
    hop_length = 512
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)

    # Convert to decibels
    Sxx = np.abs(D) ** 2
    Sxx_dB = librosa.amplitude_to_db(Sxx, ref=np.max)

    # Grid layout for plots
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, width_ratios=[1, 1, 2], height_ratios=[1, 1])
    fig.suptitle(f"{name}", fontsize=32)

    # Time-domain signal plot
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(np.linspace(0, len(y) / sr, len(y)), y, color="b")
    ax1.set_title("Señal de Audio en el Dominio del Tiempo")
    ax1.set_xlabel("Tiempo [s]")
    ax1.set_ylabel("Amplitud")
    ax1.margins(x=0)

    # 2D spectrogram plot
    ax2 = fig.add_subplot(gs[1, :2], sharex=ax1)
    times = np.arange(Sxx_dB.shape[1]) * hop_length / sr
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    cax = ax2.pcolormesh(times, frequencies, Sxx_dB, shading="gouraud", cmap="inferno")
    ax2.set_title("Power Spectrogram (en dB)")
    ax2.set_xlabel("Tiempo [s]")
    ax2.set_ylabel("Frecuencia [Hz]")
    ax2.set_ylim(0, 4000)
    ax2.margins(x=0)

    # 3D spectrogram plot
    ax3 = fig.add_subplot(gs[:, 2], projection="3d")
    cota = int(len(frequencies) * 0.175)
    X, Y = np.meshgrid(times, frequencies[:cota])
    Z = Sxx_dB[:cota]
    ax3.plot_surface(X, Y, Z, cmap="inferno", edgecolor="none")
    ax3.set_title("Spectrograma 3D de Potencia")
    ax3.set_xlabel("Tiempo [s]", labelpad=10)
    ax3.set_ylabel("Frecuencia [Hz]", labelpad=10)
    ax3.set_zlabel("Potencia [dB]")

    plt.tight_layout()
    plt.show()


def compute_cross_correlation(audio_files):
    """Computes cross-correlation for all file pairs, finds local maxima, and displays results in a DataFrame."""
    results = []

    # Compute cross-correlation for all file pairs and ranges
    file_names = list(audio_files.keys())
    for i, name1 in enumerate(file_names):
        for name2 in file_names[i + 1 :]:
            path1, path2 = audio_files[name1], audio_files[name2]
            signal1, sr1 = librosa.load(path1, sr=None)
            signal2, sr2 = librosa.load(path2, sr=None)

            if sr1 != sr2:
                raise ValueError(
                    "La frecuencia de muestreo debe ser la misma en ambos archivos."
                )

            # Compute FFT-based cross-correlation
            corr_fft, freqs = fft_correlate(signal1, signal2, sr1)

            # Find local maxima
            peaks, properties = find_peaks(
                corr_fft, height=0.2, distance=150
            )  # Adjust parameters as needed
            peak_freqs = freqs[peaks]
            peak_values = corr_fft[peaks]

            # Discard negative side peaks
            peak_freqs = peak_freqs[len(peak_freqs) // 2 :]
            peak_values = peak_values[len(peak_values) // 2 :]

            # Discard peak at 0 Hz, take the other first 9 peaks
            peak_freqs = peak_freqs[1:10]
            peak_values = peak_values[1:10]

            results.append(
                {
                    "Audio 1": name1,
                    "Audio 2": name2,
                    "Max Peak Frequency (Hz)": peak_freqs[np.argmax(peak_values)],
                    "Max Peak Correlation": max(peak_values),
                    "All Peaks (Hz)": list(peak_freqs),
                    "All Peak Values": list(peak_values),
                }
            )

            # Plot the result
            plt.figure(figsize=(24, 6))
            plt.plot(
                freqs, corr_fft, label="Correlacion Cruzada"
            )  # Cross-correlation function
            plt.scatter(
                peak_freqs, peak_values, color="red", label="Peaks"
            )  # Highlight peaks

            # Annotate peaks
            for freq, value in zip(peak_freqs, peak_values):
                plt.text(
                    freq,
                    value + 0.05,
                    f"{freq:.1f} Hz",
                    color="blue",
                    fontsize=12,
                    ha="center",
                )

            plt.title(f"Cross-Correlation of Fourier Transforms {name1} vs {name2}")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Normalized Correlation")
            plt.xlim(0, 4000)
            # plt.ylim(0,0.2)
            plt.legend()
            plt.grid()
            plt.show()


# Main function
def main():
    # Process and plot all envelopes and FFTs first
    for name, file in audio_files.items():
        print(f"Processing envelope and FFT for {name}...")
        process_signal(file, name)

    # Compute and display cross-correlation results
    compute_cross_correlation(audio_files)

    # Process and plot all spectrograms
    for name, file in audio_files.items():
        print(f"Processing spectrogram for {name}...")
        process_spectrogram(name, file)


if __name__ == "__main__":
    main()
