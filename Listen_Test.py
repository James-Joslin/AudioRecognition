class Recorder(object):

    def __init__(self, sampling_rate = 16000, num_channels = 1, sample_width = 2):    
        self.sampling_rate = sampling_rate
        self.num_channels = num_channels # Remember to drop channels from training files
        self.sample_width = sample_width # The width of each sample in bytes. Each group of ``sample_width`` bytes represents a single audio sample. 

    def pyaudio_stream_callback(self, in_data, frame_count, time_info, status):
        self.raw_audio_bytes_array.extend(in_data)
        return (in_data, pyaudio.paContinue)

    def start_recording(self):

        self.raw_audio_bytes_array = bytearray()

        pa = pyaudio.PyAudio()
        self.pyaudio_stream = pa.open(format=pyaudio.paInt16,
                                      rate=self.sampling_rate,
                                      channels=self.num_channels,
                                      input=True,
                                      stream_callback=self.pyaudio_stream_callback)

        self.pyaudio_stream.start_stream()

    def stop_recording(self):

        self.pyaudio_stream.stop_stream()
        self.pyaudio_stream.close()

        speech_recognition_audio_data = speech_recognition.AudioData(self.raw_audio_bytes_array,
                                                                     self.sampling_rate,
                                                                     self.sample_width)
        # convert the audio represented by the ``AudioData`` object to
        # a byte string representing the contents of a WAV file
        speech_recognition_audio_data = speech_recognition_audio_data.get_wav_data()
        return speech_recognition_audio_data

def decode_audio(audio_binary):
    # Decode WAV-encoded audio files to `float32` tensors, normalized
    # # to the [-1.0, 1.0] range. Return `float32` audio and a sample rate.
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    # Data is single channel (mono), drop the `channels` axis from the array.
    return tf.squeeze(audio, axis=-1)

def get_spectrogram(waveform, num_seconds):
    position = tfio.audio.trim(waveform, axis=0, epsilon=0.1)
    start = position[0]
    stop = position[1]

    waveform = waveform[start:stop]

    input_len = 16000 * num_seconds
    zero_padding = tf.zeros(
        [input_len] - tf.shape(waveform),
        dtype=tf.float32)
    zero_padding = zero_padding[0:int(len(zero_padding)/2)]
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with `zero_padding`, which ensures all audio
    # clips are of the same length.
    waveform = tf.concat([zero_padding, waveform, zero_padding], 0)
    
    spectrogram = tfio.audio.spectrogram(
        waveform, nfft=512, window=512, stride=256)
    spectrogram = tfio.audio.melscale(
        spectrogram, rate=16000, mels=256, fmin=0, fmax=8000)
    spectrogram = tfio.audio.dbscale(
        spectrogram, top_db=45)

    # Obtain the magnitude of the STFT - not necassary?
    # spectrogram = tf.abs(spectrogram)
    '''
    Add a `channels` dimension, so that the spectrogram can be used
    as image-like input data with convolution layers (which expect
    shape (`batch_size`, `height`, `width`, `channels`).
    '''
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = (spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
    import pyaudio
    import speech_recognition
    from time import sleep
    import tensorflow as tf
    import tensorflow_io as tfio
    import matplotlib.pyplot as plt
    import numpy as np

    rate = 16000
    recorder = Recorder(sampling_rate=rate)

    # start recording
    recorder.start_recording()
    print('Listening')
    duration = 3
    sleep(duration)
    wav_data = recorder.stop_recording()
    print('Ended')

    input_audio = decode_audio(wav_data)
    print('Audio input shape:', input_audio.shape)
    
    spectrogram = get_spectrogram(input_audio, duration)

    print('Waveform shape:', input_audio.shape)
    print('Spectrogram shape:', spectrogram.shape)

    fig, axes = plt.subplots(2, figsize=(12, 8))
    axes[0].set_title('Whole Waveform')
    axes[0].plot(input_audio)

    plot_spectrogram(spectrogram.numpy(), axes[1])
    axes[1].set_title('Centred dbScale Mel-Spectrogram')
    plt.tight_layout()
    plt.show()
    fig.savefig("./Close1.png")