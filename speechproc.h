#ifndef SPEECHPROC_H
#define SPEECHPROC_H

#include <vector>
#include <utility> // std::pair

#include <torch/script.h>

#include <torchaudio/csrc/sox/io.h>
#include <torchaudio/csrc/sox/effects.h>

#include <sndfile.h>

/**
 * @brief read SF_INFO instance to understand codec type and depth
 * @param info - SF_INFO instance
 * @param name - codec name as string
 * @param depth - bit depth os samples
 */
void check_sf_format(const SF_INFO &info, std::string &name, int &depth);

/**
 * @brief read audio from file to torch::Tensor
 * @param filename
 * @param sampling_rate - target sampling rate
 * @param ok - optional argument to check if data has been red successfully
 * @return audio in torch::Tensor format
 */
torch::Tensor read_audio_sndfile(const std::string &filename, int target_sampling_rate, SF_INFO &sfinfo, bool *ok=nullptr);

/**
 * @brief read audio from memory buffer to torch::Tensor
 * @param content - pointer to binary data in memory
 * @param content_size - binary data size in bytes
 * @param ok - optional argument to check if data has been red successfully
 * @return audio in torch::Tensor format
 */
torch::Tensor read_audio_sndfile(const uint8_t *content, uint64_t content_size, int target_sampling_rate, SF_INFO &sfinfo, bool *ok=nullptr);

/**
 * @brief read audio from file to torch::Tensor
 * @param filename
 * @param sampling_rate - target sampling rate
 * @return audio in torch::Tensor format
 * @note call torchaudio::sox_effects::initialize_sox_effects() once before use this
 */
torch::Tensor read_audio_torchaudio(const std::string &filename, int target_sampling_rate, int target_channels);

/**
 * @brief calculate record duration
 * @param audio - audio tensor
 * @param sampling_rate - audio sampling rate in Hz
 * @return resord duration in seconds
 */
float record_duration(const torch::Tensor &audio, int sampling_rate);

/**
 * @brief process audio tensor and return vector of speech timestamps
 * @param audio - input audio tensor with single channel, i.e.: 1xN
 * @param model - VAD model
 * @param threshold - VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH
 * @param neg_threshold - speech detection hysteresis
 * @param sampling_rate - VAD model sample rate
 * @param min_speech_duration_ms - final speech chunks shorter min_speech_duration_ms are thrown out
 * @param min_silence_duration_ms - in the end of each speech chunk wait for min_silence_duration_ms before separating it
 * @param window_size_samples - audio chunks of window_size_samples size are fed to the VAD model (WARNING! Use 512, 1024, 1536 for 16000 sample rate and 256, 512, 768 for 8000 sample rate)
 * @param speech_pad_ms - final speech chunks are padded by speech_pad_ms each side
 * @return speech timestamps vector
 * @note if you do not want to experiment with parameters simply use apply_vad_8khz()
*/
std::vector<std::pair<int,int>> silero_speech_stamps(const torch::Tensor &audio,
                                             const torch::jit::script::Module &model,
                                             float threshold,
                                             float neg_threshold,
                                             int sampling_rate,
                                             int min_speech_duration_ms,
                                             int min_silence_duration_ms,
                                             int window_size_samples,
                                             int speech_pad_ms );
/**
 * @brief apply VAD to 8 kHz audio
 * @param audio - input audio tensor with single channel (i.e. 1xN) and 8000 Hz sample rate
 * @param model - VAD model
 * @return speech timestamps vector
 */
std::vector<std::pair<int,int>> apply_silero_vad_8khz(const torch::Tensor &audio, const torch::jit::script::Module &model);

/**
 * @brief process audio tensor and return vector of speech timestamps
 * @param audio - input audio tensor with single channel, i.e.: 1xN
 * @param model
 * @param threshold
 * @param neg_threshold
 * @param sampling_rate
 * @param min_speech_duration_ms
 * @param min_silence_duration_ms
 * @param speech_pad_ms
 * @return speech timestamps vector
 * @note if you do not want to experiment with parameters simply use apply_bisolut_vad_8khz()
 */
std::vector<std::pair<int,int>> bisolut_speech_stamps(const at::Tensor &iaudio,
                                              torch::jit::script::Module &model,
                                              float threshold,
                                              float neg_threshold,
                                              int sampling_rate,
                                              int min_speech_duration_ms,
                                              int min_silence_duration_ms,
                                              int speech_pad_ms);

/**
 * @brief apply VAD to 8 kHz audio
 * @param audio - input audio tensor with single channel (i.e. 1xN) and 8000 Hz sample rate
 * @param model - VAD model
 * @return speech timestamps vector
 */
std::vector<std::pair<int,int>> apply_bisolut_vad_8khz(const at::Tensor &audio, torch::jit::script::Module &model);

/**
 * @brief calculate speech duration
 * @param timestamps - speech timestamps
 * @param sampling_rate - audio sampling rate in Hz
 * @return speech duration in seconds
 */
float speech_duration(const std::vector<std::pair<int,int>> &timestamps, int sampling_rate);

/**
 * @brief calculate snr
 * @param audio - input audio tensor with single channel, i.e.: 1xN
 * @param speech_timestamps - speech timestamps
 * @param sampling_rate - audio sampling rate in Hz
 * @return snr in dB
 */
float estimate_snr(const torch::Tensor &audio, const std::vector<std::pair<int,int>> &speech_timestamps, int sampling_rate);

/**
 * @brief calculate audio overload
 * @param audio - input audio tensor with single channel, i.e.: 1xN
 * @param sampling_rate - audio sampling rate in Hz
 * @return overload power (0.0 - not at all, 1.0 - overloaded on whole length)
 */
float estimate_overload(const torch::Tensor &audio, int sampling_rate);

/**
 * @brief calculate relative energy of signal's harmonics below frequency
 * @param audio - input audio tensor with single channel, i.e.: 1xN
 * @param sampling_rate - audio sampling rate in Hz
 * @param frequency - threshold frequency in Hz
 * @return relative energy from 0.0 to 1.0
 */
float estimate_energy_below_frequency(const torch::Tensor &audio, int sampling_rate, float frequency);

/**
 * @brief process audio tensor and return Russian language probability estiamtion
 * @param audio - input audio tensor with single channel and 16000 Hz sample rate
 * @param model - lang  prediction model (SILERO 4 languages)
 * @return model's confidence about Russian language in audio
 */
float russian_language_prob(const torch::Tensor &audio, torch::jit::script::Module &model);

/**
 * @brief process audio tensor and return Russian language probability estiamtion
 * @param audio - input audio tensor with single channel and 8000 Hz sample rate
 * @param model - lang  prediction model (BISOLUT 7 languages)
 * @return model's confidence about Russian language in audio
 */
float ru_prob(const torch::Tensor &audio, torch::jit::script::Module &model);

/**
 * @brief process audio tensor and return many voices on record probability
 * @param audio - input audio tensor with single channel and 8000 Hz sample rate
 * @param model - model that predicts many voices
 * @return model's confidence about many voices on record
 */
float many_voices_prob(const torch::Tensor &audio, torch::jit::script::Module &model);

/**
 * @brief process audio tensor and return numbers sequence pronounced in record
 * @param audio - input audio tensor with single channel and 8000 Hz sample rate
 * @param speech_timestamps - speech timestamps for audio
 * @param model - model that predicts numbers
 * @return numbers sequence pronounced in record
 */
std::vector<std::string> predict_sequence(const torch::Tensor &audio, const std::vector<std::pair<int,int>> &speech_timestamps, torch::jit::script::Module &model);

#endif // SPEECHPROC_H
