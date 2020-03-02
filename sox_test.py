import sox

file_name = 'common_voice_id_19059577.wav'
print("file name   :", file_name)
print("bit rate    :", sox.file_info.bitrate(file_name))
print("sample rate :", sox.file_info.sample_rate(file_name))
print("channels    :", sox.file_info.channels(file_name))
print("duration    :", sox.file_info.duration(file_name))
print("encoding    :", sox.file_info.encoding(file_name))
print("file type   :", sox.file_info.file_type(file_name))
print("num samples :", sox.file_info.num_samples(file_name))
