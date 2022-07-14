#include "speechproc.h"


at::Tensor read_audio(const std::string &filename, int target_sampling_rate, int target_channels)
{
    std::tuple<torch::Tensor, int64_t> data = torchaudio::sox_io::load_audio_file(filename,0,-1,true,true,"wav");
    const torch::Tensor &raw_tensor = std::get<0>(data);
    const int64_t raw_sample_rate = std::get<1>(data);

    // DEBUG
    /*std::cout << "INPUT AUDIO" << std::endl;
    std::cout << " - size: ";
    for(size_t i = 0; i < raw_tensor.sizes().size(); ++i) {
        if(i != 0)
            std::cout << "x";
        std::cout << raw_tensor.sizes()[i];
    }
    std::cout << std::endl;
    std::cout << " - sample rate: " << raw_sample_rate << std::endl;*/

    if(raw_tensor.sizes()[0] != target_channels || raw_sample_rate != target_sampling_rate) {

        std::vector<std::vector<std::string>> effects;
        effects.push_back({"channels", std::to_string(target_channels)});
        effects.push_back({"rate", "-a", std::to_string(target_sampling_rate)});
        data = torchaudio::sox_effects::apply_effects_tensor(raw_tensor,raw_sample_rate,effects,true);

        const torch::Tensor &tensor = std::get<0>(data);
        const int64_t sample_rate = std::get<1>(data);

        /*std::cout << "RESAMPLED AUDIO" << std::endl;
        std::cout << " - size: ";
        for(size_t i = 0; i < tensor.sizes().size(); ++i) {
            if(i != 0)
                std::cout << "x";
            std::cout << tensor.sizes()[i];
        }
        std::cout << std::endl;
        std::cout << " - sample rate: " << sample_rate << std::endl;*/

        assert(tensor.sizes()[0] == target_channels);
        assert(sample_rate == target_sampling_rate);
        return tensor;
    }
    assert(raw_tensor.sizes()[0] == target_channels);
    assert(raw_sample_rate == target_sampling_rate);
    return raw_tensor;
}



std::vector<std::pair<int,int>> speech_stamps(torch::Tensor audio,
                                              const torch::jit::script::Module &model,
                                              float threshold,
                                              float neg_threshold,
                                              int sampling_rate,
                                              int min_speech_duration_ms,
                                              int min_silence_duration_ms,
                                              int window_size_samples,
                                              int speech_pad_ms )
{
    // reset model state - it should be done for this specific DNN
    torch::jit::script::Module vad = model.deepcopy();

    // trying to squeeze empty dimensions
    if(audio.sizes().size() > 1)
        if(audio.sizes()[0] == 1 and audio.sizes()[1] > 0)
            audio = torch::squeeze(audio,0);   

    int min_speech_samples = sampling_rate * min_speech_duration_ms / 1000;
    int min_silence_samples = sampling_rate * min_silence_duration_ms / 1000;
    int speech_pad_samples = sampling_rate * speech_pad_ms / 1000;

    int audio_length_samples = audio.sizes()[0];

    std::vector<float> speech_probs;
    speech_probs.reserve(audio_length_samples / window_size_samples + 1);

    c10::InferenceMode guard;
    for(int current_start_sample = 0; current_start_sample < audio_length_samples; current_start_sample += window_size_samples) {
        torch::Tensor chunk = torch::slice(audio,0,current_start_sample,current_start_sample + window_size_samples);        
        if(chunk.sizes()[0] < window_size_samples)
            chunk = torch::constant_pad_nd(chunk, torch::IntList{0, int(window_size_samples - chunk.sizes()[0])}, 0);
        float prob = vad.forward({chunk, sampling_rate}).toTensor()[0].item().toFloat();
        speech_probs.emplace_back(prob);
    }

    bool triggered = false;
    std::vector<std::pair<int,int>> speeches;
    std::pair<int,int> current_speech;
    int temp_end = 0;

    for(int i = 0; i < (int)speech_probs.size(); ++i) {
        const float &speech_prob = speech_probs[i];
        if(speech_prob >= threshold && temp_end
                )
            temp_end = 0;
        if(speech_prob >= threshold && triggered == false) {
            triggered = true;
            current_speech.first = window_size_samples * i;
            continue;
        }
        if(speech_prob < neg_threshold && triggered == true) {
            if(temp_end == 0)
                temp_end = window_size_samples * i;
            if((window_size_samples * i) - temp_end < min_silence_samples)
                continue;
            else {
                current_speech.second = temp_end;
                if((current_speech.second - current_speech.first) > min_speech_samples)
                    speeches.emplace_back(current_speech);
                temp_end = 0;
                current_speech = std::pair<int,int>();
                triggered = false;
                continue;
            }
        }
    }
    if(current_speech.second == audio_length_samples)
        speeches.emplace_back(current_speech);

    for(int i = 0; i < (int)speeches.size(); ++i) {
        std::pair<int,int> &speech = speeches[i];
            if(i == 0)
                speech.first = int(std::max(0, speech.first - speech_pad_samples));
            if(i != (int)speeches.size() - 1) {
                int silence_duration = speeches[i + 1].first - speech.second;
                if(silence_duration < 2 * speech_pad_samples) {
                    speech.second += int(silence_duration / 2);
                    speeches[i + 1].first = int(std::max(0, speeches[i + 1].first - silence_duration / 2));
                } else
                    speech.second += int(speech_pad_samples);
            } else
                speech.second = int(std::min(audio_length_samples, speech.second + speech_pad_samples));
    }

    return speeches;
}

std::vector<std::pair<int,int>> apply_vad_8khz(at::Tensor audio, const torch::jit::script::Module &model)
{
    return speech_stamps(audio,model,0.9f,0.9f,8000,10,10,256,30);
}

std::vector<std::pair<int,int>> apply_vad_16khz(at::Tensor audio, const torch::jit::script::Module &model)
{
    return speech_stamps(audio,model,0.9f,0.9f,16000,10,10,512,30);
}

