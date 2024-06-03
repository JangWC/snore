# 파일이름 설정---------------------------------------------------------------------------

sound_file_name = 'TEST'
sound_foramt = 'm4a'

pressure_file_name = 'LPS22HH_1.csv'

# model load ---------------------------------------------------------------------------

from tensorflow.keras.models import load_model
best_model = load_model('CNNLSTM_tanh_best.h5')
loaded_model = load_model('sound_model.h5')

# Sound file slice  ---------------------------------------------------------------------------

from pydub import AudioSegment
import os
import pandas as pd

def convert_time_to_seconds(t):
    return t.hour * 3600 + t.minute * 60 + t.second

for filename in os.listdir('.'):
    ## 파일들 이름이 같은 글자로 시작되어야함
    ## 예시) 스마트~1, 스마트~2인경우 = filename.startswith("스마트")
    ## 예시) LG~1, LG~2인경우 = filename.startswith("LG")
    if filename.endswith(".m4a") and filename.startswith(sound_file_name):
        # 오디오 파일과 동일한 이름의 엑셀 파일을 찾음
#         excel_filename = filename.replace('.wav', '.xlsx')
#         if os.path.exists('목업/'+excel_filename):
#             print(filename)
#             # 엑셀 파일 로드
#             df = pd.read_excel('목업/'+excel_filename, header=None, engine='openpyxl')
#             df = df.iloc[:, [0,1]].dropna(axis=0)
#             df.iloc[:, 0] = df.iloc[:, 0].apply(convert_time_to_seconds)
            
        # 오디오 파일 로드
        audio_name = os.path.join(filename)
        audio = AudioSegment.from_file(audio_name, format=sound_foramt)

        # 10초 단위로 오디오 파일 자르기
        ten_seconds = 10 * 1000  # 10 seconds in milliseconds

        for i in range(0, len(audio), ten_seconds):
            slice = audio[i:i + ten_seconds]
            # 해당 시간 구간의 라벨링 확인
            start_time = i // 1000
            end_time = (i + ten_seconds) // 1000
            # labels = df[(df.iloc[:, 0] >= start_time) & (df.iloc[:, 0] < end_time)].iloc[:, 1]

            # 조건에 따라 폴더 선택
            folder = 'sound_slice'
            if not os.path.exists(folder):
                os.makedirs(folder)
                
            index = i // ten_seconds
            # 오디오 파일 저장
            slice.export(os.path.join(folder, f"{filename[:-4]}_{index}.wav"), format="wav")
            
# 모델 실행 전 단계  ---------------------------------------------------------------------------
import os
from matplotlib import pyplot as plt
import tensorflow as tf 
import tensorflow_io as tfio
import math
from IPython.display import Audio
from string import ascii_uppercase
from pandas import DataFrame

import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.io.wavfile import write
#import librosa.display

from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
from scipy.signal import firwin, lfilter


def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels) 
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    
    wav.set_shape(tf.TensorShape([None,]))  # Here you might want to specify a shape

    return wav




def adjust_values(sample, label):
    # 0.5 이상의 값들을 0.5로, 0.1 이하의 값들을 0으로 조정
    sample = tf.where(sample >= 1.0, 1.0, sample)
    sample = tf.where(sample < 0.15, 0.0, sample)
    return sample, label

def resize_batch_spectrograms(spectrograms, labels):
    # spectrograms의 시간 축을 너비로 간주하고 리사이즈 (batch_size, 161, 500, 1)
    resized_spectrograms = tf.image.resize(spectrograms, [500, 161])
    return resized_spectrograms, labels
def high_pass_filter(wav, cutoff=0.5, sample_rate=16000):
    # Perform FFT
    fft = tf.signal.fft(tf.cast(wav, tf.complex64))
    
    # Create frequency bins
    freq_bins = tf.linspace(0.0, float(sample_rate) / 2.0, num=tf.shape(fft)[0] // 2 + 1)
    
    # Create mask for high-pass filter
    mask = tf.cast(freq_bins > cutoff, tf.complex64)
    mask = tf.concat([mask, tf.reverse(mask[1:-1], axis=[0])], axis=0)
    
    # Apply mask
    fft_filtered = fft * mask
    
    # Perform inverse FFT
    wav_filtered = tf.signal.ifft(fft_filtered)
    
    return tf.cast(tf.math.real(wav_filtered), tf.float32)

def preprocess(file_path, label):
    wav = load_wav_16k_mono(file_path)
    wav = wav[:160000]
    
    # Apply high-pass filter
    wav = high_pass_filter(wav, cutoff=0.5)
    
    zero_padding = tf.zeros([160000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    
    window = np.kaiser(320, 10)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32, fft_length=320, window_fn=lambda frame_size, dtype: window,
                                 pad_end=True,
                                 name=None)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    
    spectrogram_3_channels = tf.tile(spectrogram, [1, 1, 3])
    print("Spectrogram shape:", spectrogram_3_channels.shape)
    return spectrogram_3_channels, label
       
# 모델 사용  ---------------------------------------------------------------------------
import random
import re
import numpy as np

#group_names = ['피험자a', '피험자b', '피험자c', '피험자d','피험자e', '피험자f','피험자g','피험자h','피험자i','피험자j']  # 사용할 그룹 이름
group_names = [sound_file_name]  # 사용할 그룹 이름
SNORING_DATA_PATH = os.path.join('sound_slice')
ex_val = tf.data.Dataset.from_tensor_slices([])
first = True

def extract_number_from_filename(filename):
    match = re.search(r'_(\d+)', filename)
    return int(match.group(1)) if match else 0

def sorted_list_files(path_pattern):
    files = tf.io.gfile.glob(path_pattern)
    files.sort(key=extract_number_from_filename)
    return files


for i in group_names:
    files = sorted_list_files(SNORING_DATA_PATH + '/' + i + '*.wav')
    pos = tf.data.Dataset.from_tensor_slices(files)    #neg = tf.data.Dataset.list_files(NOT_SNORING_DATA_PATH + '/'+ i +'*.wav')
    positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
    #negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
    data = positives

    # preprocess 함수 적용
    data = data.map(preprocess)

    # 값 조정 함수 적용
    data = data.map(adjust_values)

    # 나머지 데이터셋 파이프라인
    data = data.cache()
    data = data.batch(64)
    data = data.prefetch(8)
    data = data.map(resize_batch_spectrograms)
    if first:
        ex_val = data
        first = False
    else:
        ex_val = ex_val.concatenate(data)

        
data = tf.data.Dataset.from_tensor_slices([])

# 이미지 데이터만을 추출하여 NumPy 배열로 저장
images = []
labels = []

for batch_images, batch_labels in ex_val.as_numpy_iterator():
    images.append(batch_images)
    labels.append(batch_labels)

# 모든 배치를 하나의 배열로 결합
x_ex_val = np.concatenate(images, axis=0)
y_ex_val = np.concatenate(labels, axis=0)

images = []
labels = []

sound_test_prob = best_model.predict(x_ex_val)

import pandas as pd

df = pd.read_csv(pressure_file_name, usecols=[1])
b_column_list = df.iloc[:, 0].tolist()

def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        chunk = lst[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk.extend([0] * (chunk_size - len(chunk)))
        yield chunk

# B 열 리스트를 100개씩 나누기
chunks = list(chunk_list(b_column_list, 100))
from sklearn.preprocessing import MinMaxScaler
X = np.array(chunks)
X = X.reshape((X.shape[0], X.shape[1], 1))  
scaler = MinMaxScaler()
X = scaler.fit_transform(X.reshape(-1, 1)).reshape(X.shape[0], X.shape[1], 1)
press_sound_proba = loaded_model.predict(X)

# ensmeble 및 최종 결과 -------------------------------------------------------------------------------------
target_length = sound_test_prob.shape[0]

if len(press_sound_proba) > target_length:
    press_sound_proba = press_sound_proba[:target_length]*0.01
elif len(press_sound_proba) < target_length:
    padding = np.zeros((target_length - len(press_sound_proba), 1))
    press_sound_proba = np.vstack((press_sound_proba, padding))
    press_sound_proba = press_sound_proba*0.01

ensemble_proba = press_sound_proba + sound_test_prob

test_pred = np.where(ensemble_proba > 0.5, 1, 0)

indices_of_1 = np.where(test_pred == 1)[0]
marked_indices = indices_of_1 * 10
ratio_of_1 = np.mean(test_pred == 1)

print('파일명 : ', sound_file_name)
print("코골이 발생 위치 :", marked_indices)
print("2회 이상 코골이 발생 퍼센트:", ratio_of_1*100, '%')