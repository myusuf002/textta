import os
import sox
import wave
import tempfile
import collections

DEFAULT_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_WIDTH = 2
DEFAULT_FORMAT = (DEFAULT_RATE, DEFAULT_CHANNELS, DEFAULT_WIDTH)

def info_audio(src_audio_path):
    print("file name   :", src_audio_path)
    print("bit rate    :", sox.file_info.bitrate(src_audio_path))
    print("sample rate :", sox.file_info.sample_rate(src_audio_path))
    print("channels    :", sox.file_info.channels(src_audio_path))
    print("duration    :", sox.file_info.duration(src_audio_path))
    print("encoding    :", sox.file_info.encoding(src_audio_path))
    print("file type   :", sox.file_info.file_type(src_audio_path))
    print("num samples :", sox.file_info.num_samples(src_audio_path))
    
def convert_audio(src_audio_path, dst_audio_path, file_type=None, audio_format=DEFAULT_FORMAT):
    sample_rate, channels, width = audio_format
    transformer = sox.Transformer()
    transformer.set_output_format(file_type=file_type, rate=sample_rate, channels=channels, bits=width*8)
    transformer.build(src_audio_path, dst_audio_path)
