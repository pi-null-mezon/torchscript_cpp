#ifndef SPEECHPROC_H
#define SPEECHPROC_H

#include <vector>
#include <utility> // std::pair

#include <torch/script.h>

#include <torchaudio/csrc/sox/io.h>
#include <torchaudio/csrc/sox/effects.h>

/**
 * @brief read audio from file to torch::Tensor
 * @param filename
 * @param sampling_rate - target sampling rate
 * @return audio in torch::Tensor format
 * @note call torchaudio::sox_effects::initialize_sox_effects() once before use this
 */
torch::Tensor read_audio(const std::string &filename, int target_sampling_rate, int target_channels);

/**
 * @brief process audio and return vector of speech timestamps
 * @param single_channel_audio - one dimensional float torch.Tensor
 * @param model - VAD model
 * @param threshold - VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH
 * @param neg_threshold - speech detection hysteresis
 * @param sampling_rate - VAD model sample rate
 * @param min_speech_duration_ms - final speech chunks shorter min_speech_duration_ms are thrown out
 * @param min_silence_duration_ms - in the end of each speech chunk wait for min_silence_duration_ms before separating it
 * @param window_size_samples - audio chunks of window_size_samples size are fed to the VAD model (WARNING! Use 512, 1024, 1536 for 16000 sample rate and 256, 512, 768 for 8000 sample rate)
 * @param speech_pad_ms - final speech chunks are padded by speech_pad_ms each side
 * @return timestamps vector
*/
std::vector<std::pair<int,int>> speech_stamps(torch::Tensor single_channel_audio,
                                             const torch::jit::script::Module &vad_model,
                                             float threshold,
                                             float neg_threshold,
                                             int sampling_rate,
                                             int min_speech_duration_ms,
                                             int min_silence_duration_ms,
                                             int window_size_samples,
                                             int speech_pad_ms );

std::vector<std::pair<int,int>> apply_vad_8khz(torch::Tensor single_channel_audio, const torch::jit::script::Module &vad_model);

std::vector<std::pair<int,int>> apply_vad_16khz(torch::Tensor single_channel_audio, const torch::jit::script::Module &vad_model);


#endif // SPEECHPROC_H
