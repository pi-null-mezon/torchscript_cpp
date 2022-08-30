#include "speechproc.h"

#include <soxr.h>

torch::Tensor normalize(const torch::Tensor& tensor) {
    assert(tensor.sizes().size() == 1);
    return tensor / torch::max(tensor.abs());
}

torch::Tensor read_audio_torchaudio(const std::string &filename, int target_sampling_rate, int target_channels)
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


std::vector<std::pair<int,int>> silero_speech_stamps(const at::Tensor &iaudio,
                                              const torch::jit::script::Module &model,
                                              float threshold,
                                              float neg_threshold,
                                              int sampling_rate,
                                              int min_speech_duration_ms,
                                              int min_silence_duration_ms,
                                              int window_size_samples,
                                              int speech_pad_ms )
{
    // reset model state - it should be done for this specific DNN to get reproduceable output
    torch::jit::script::Module vad = model.deepcopy();

    // slice single audio channel
    const torch::Tensor &audio = iaudio[0];

    int min_speech_samples = sampling_rate * min_speech_duration_ms / 1000;
    int min_silence_samples = sampling_rate * min_silence_duration_ms / 1000;
    int speech_pad_samples = sampling_rate * speech_pad_ms / 1000;

    int audio_length_samples = audio.sizes()[0];

    std::vector<float> speech_probs;
    speech_probs.reserve(audio_length_samples / window_size_samples + 1);

    c10::InferenceMode guard;
    for(int current_start_sample = 0; current_start_sample < audio_length_samples; current_start_sample += window_size_samples) {
        torch::Tensor chunk = torch::slice(audio,0,current_start_sample,current_start_sample + window_size_samples);        
        if(chunk.sizes()[0] < window_size_samples) {
            torch::Tensor padded = torch::zeros({window_size_samples});
            padded.slice(0,0,chunk.sizes()[0]) = chunk;
            chunk = std::move(padded);
        }
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

std::vector<std::pair<int,int>> apply_silero_vad_8khz(const at::Tensor &audio, const torch::jit::script::Module &model)
{
    return silero_speech_stamps(audio,model,0.9f,0.9f,8000,100,10,256,30);
}

std::vector<std::pair<int,int>> bisolut_speech_stamps(const at::Tensor &iaudio,
                                              torch::jit::script::Module &model,
                                              float threshold,
                                              float neg_threshold,
                                              int sampling_rate,
                                              int min_speech_duration_ms,
                                              int min_silence_duration_ms,
                                              int speech_pad_ms)
{
    // slice single audio channel
    const torch::Tensor &audio = iaudio[0];

    const int min_speech_samples = sampling_rate * min_speech_duration_ms / 1000;
    const int min_silence_samples = sampling_rate * min_silence_duration_ms / 1000;
    const int speech_pad_samples = sampling_rate * speech_pad_ms / 1000;
    const int audio_length_samples = audio.sizes()[0];
    const int window_size_samples = 256;

    c10::InferenceMode guard;
    std::vector<torch::Tensor> batch;
    batch.reserve(audio_length_samples / window_size_samples + 1);
    for(int current_start_sample = 0; current_start_sample < audio_length_samples; current_start_sample += window_size_samples) {
        torch::Tensor chunk = torch::slice(audio,0,current_start_sample,current_start_sample + window_size_samples);
        if(chunk.sizes()[0] < window_size_samples) {
            torch::Tensor padded = torch::zeros({window_size_samples});
            padded.slice(0,0,chunk.sizes()[0]) = chunk;
            chunk = std::move(padded);
        }
        batch.emplace_back(chunk);
    }
    std::vector<float> speech_probs;
    speech_probs.reserve(batch.size());
    if(batch.size() > 0) {
        c10::InferenceMode guard;
        torch::Tensor batch_tensor = torch::stack(batch).unsqueeze(1);
        torch::Tensor probs = torch::softmax(model.forward({batch_tensor}).toTensor(),1);
        for(int i = 0; i < probs.sizes()[0]; ++i)
            speech_probs.emplace_back(probs[i][1].item().toFloat());
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

std::vector<std::pair<int,int>> apply_bisolut_vad_8khz(const at::Tensor &audio, torch::jit::script::Module &model)
{
    return bisolut_speech_stamps(audio,model,0.75f,0.75f,8000,100,30,20);
}


float russian_language_prob(const torch::Tensor &audio, torch::jit::script::Module &model)
{
    c10::InferenceMode guard;
    torch::Tensor prediction = model.forward({audio}).toTuple()->elements()[2].toTensor();
    torch::Tensor probs = torch::softmax(prediction, 1).squeeze(0);
    return probs[0].item().toFloat();
}

float ru_prob(const torch::Tensor &audio, torch::jit::script::Module &model)
{
    c10::InferenceMode guard;
    torch::Tensor prediction = model.forward({normalize(audio.squeeze(0)).unsqueeze(0).unsqueeze(0)}).toTensor();
    torch::Tensor probs = torch::softmax(prediction, 1).squeeze(0);
    return probs[4].item().toFloat();
}

float record_duration(const at::Tensor &audio, int sampling_rate)
{
    return static_cast<float>(audio.sizes()[1]) / sampling_rate;
}

float speech_duration(const std::vector<std::pair<int, int> > &timestamps, int sampling_rate)
{
    int samples = 0;
    for(const auto &item: timestamps)
        samples += item.second - item.first;
    return static_cast<float>(samples) / sampling_rate;
}

float estimate_snr(const at::Tensor &audio, const std::vector<std::pair<int,int>> &speech_timestamps, int sampling_rate)
{
    // filter voice harmonics
    torch::Tensor harmonics = torch::fft_rfft(audio[0]);
    float f_step = static_cast<float>(sampling_rate / 2) / harmonics.sizes()[0];
    int low = static_cast<int>(50.0f / f_step);
    int high = static_cast<int>(3850.0f / f_step);
    harmonics.slice(0,0,low) = 0.0f;
    harmonics.slice(0,high,harmonics.sizes()[0]) = 0.0f;
    torch::Tensor voice_squared = torch::fft_irfft(harmonics).square();
    // estimate signal energy
    float s_energy = 0.0f;
    int s_counts = 0;
    for(const auto &speech: speech_timestamps) {
        s_energy += torch::sum(voice_squared.slice(0,speech.first,speech.second)).item().toFloat();
        s_counts += speech.second - speech.first;
    }
    // estimate noise energy
    torch::Tensor audio_squared = torch::square(audio[0]);
    float n_energy = (torch::sum(audio_squared).item().toFloat() - s_energy) / (audio_squared.sizes()[0] - s_counts + 1.0E-6f);
    s_energy /= (s_counts + 1.0E-6f);
    return std::min(100.0f, 10.0f * std::log10((s_energy / (n_energy + 1.0E-6f)) + 1.0E-5f));
}

float estimate_overload(const at::Tensor &audio, int sampling_rate)
{
    int window_size = static_cast<int>(0.25f * sampling_rate);
    std::vector<float> envelope;
    envelope.reserve(audio.sizes()[1] / window_size + 1);
    if(window_size < audio.sizes()[1])
        for(int current_start_sample = 0; current_start_sample < (int)audio.sizes()[1]; current_start_sample += window_size)
            envelope.emplace_back(audio.slice(1,current_start_sample,current_start_sample + window_size).abs().max().item().toFloat());
    if(envelope.size() > 0) {
        float max = *std::max_element(envelope.begin(),envelope.end());
        int max_occurencies = 0;
        for(size_t i = 0; i < envelope.size(); ++i)
            if(envelope[i] == max)
                max_occurencies++;
        return max_occurencies == 1 ? 0.0f : float(max_occurencies) / envelope.size();
    }
    return 0.0f;
}

float estimate_energy_below_frequency(const at::Tensor &audio, int sampling_rate, float frequency)
{
    torch::Tensor amplitude_spectrum = torch::fft_rfft(audio[0]).abs();
    float f_step = static_cast<float>(sampling_rate / 2) / amplitude_spectrum.sizes()[0];
    int index = static_cast<int>(frequency / f_step);
    if(index >= amplitude_spectrum.sizes()[0])
        return 1.0f;
    float energy_below = torch::sum(amplitude_spectrum.slice(0,0,index)).item().toFloat();
    float energy_above = torch::sum(amplitude_spectrum.slice(0,index,amplitude_spectrum.sizes()[0])).item().toFloat();
    return energy_below / (energy_above + energy_below + 1.0E-6f);
}

torch::Tensor fix_length(const torch::Tensor& tensor, int target_length) {
    assert(tensor.sizes().size() == 1);
    torch::Tensor out;
    if(tensor.sizes()[0] < target_length) {
        int gap = (target_length - tensor.sizes()[0] ) / 2;
        out = torch::zeros({target_length});
        out.slice(0,gap,gap + tensor.sizes()[0]) = tensor;
    } else if(tensor.sizes()[0] > target_length) {
        int shift = (tensor.sizes()[0] - target_length) / 2;
        out = tensor.slice(0, shift, shift + target_length);
    }
    return out;
}


float many_voices_prob(const at::Tensor &audio, torch::jit::script::Module &model)
{
    // split into chunks
    int window_size = 5 * 8000; // hyperparameters
    std::vector<torch::Tensor> chunks;
    chunks.reserve(audio.sizes()[1] / window_size + 1);
    for(int start = 0; start < (int)audio.sizes()[1]; start += window_size) {
        torch::Tensor chunk = normalize(audio[0].slice(0,start,start + window_size));
        if(chunk.sizes()[0] > 0.75 * window_size || chunks.size() == 0) {
            if(chunk.sizes()[0] < window_size) {
                torch::Tensor tmp = torch::zeros({window_size});
                tmp.slice(0,0,chunk.sizes()[0]) = chunk;
                chunks.emplace_back(tmp);
            } else {
                chunks.emplace_back(chunk);
            }
        }
    }
    torch::Tensor batch_tensor = torch::stack(chunks).unsqueeze(1);
    c10::InferenceMode guard;
    torch::Tensor prediction = model({batch_tensor}).toTensor();
    // softmax over labels dimension
    torch::Tensor probs = torch::softmax(prediction, 1);
    // average over batch dimension
    return torch::mean(probs,0)[1].item().toFloat();
}


std::vector<std::string> predict_sequence(const at::Tensor &audio, const std::vector<std::pair<int,int>> &speech_timestamps, torch::jit::script::Module &model)
{
    std::vector<std::string> sequence;
    if(speech_duration(speech_timestamps,8000) > 0.1) {
        std::vector<torch::Tensor> batch;
        batch.reserve(speech_timestamps.size());
        for(const auto &part: speech_timestamps) {
            int duration = part.second - part.first;
            if(duration > 1500)
                batch.emplace_back(normalize(fix_length(audio[0].slice(0,part.first,part.second), 8000)));
        }
        if(batch.size() > 0) {
            c10::InferenceMode guard;
            torch::Tensor batch_tensor = torch::stack(batch).unsqueeze(1);
            torch::Tensor prediction = model.forward({batch_tensor}).toTensor();
            for(int i = 0; i < prediction.sizes()[0]; ++i) {
                int label = torch::argmax(prediction[i]).item().toInt();
                sequence.emplace_back(label == 10 ? "!" : std::to_string(label));
            }
        }
    }
    return sequence;
}

void check_sf_format(const SF_INFO &info, std::string &name, int &depth)
{
    if((info.format & 0xFF0000) == 0x010000) {
        name = "WAV";
        depth = 0;
        if((info.format & 0xFFFF) == 0x0001) {
            name += ":PCM_S8";
            depth = 8;
        } else if((info.format & 0xFFFF) == 0x0002) {
            name += ":PCM_16";
            depth = 16;
        } else if((info.format & 0xFFFF) == 0x0003) {
            name += ":PCM_24";
            depth = 24;
        } else if((info.format & 0xFFFF) == 0x0004) {
            name += ":PCM_32";
            depth = 32;
        } else if((info.format & 0xFFFF) == 0x0005) {
            name += ":PCM_U8";
            depth = 8;
        }
    } else {
        name = "encoded";
        depth = -1;
    }
}

at::Tensor read_audio_sndfile(const std::string &filename, int target_sampling_rate, SF_INFO &sfinfo, bool *ok)
{
    if(ok)
        *ok = false;

    memset (&sfinfo, 0, sizeof (sfinfo)) ;
    SNDFILE	*infile;
    if (! (infile = sf_open (filename.c_str(), SFM_READ, &sfinfo)))
        return torch::Tensor();

    std::vector<float> waveform, buffer;
    waveform.reserve(sfinfo.channels*sfinfo.frames);
    buffer.resize(4096);
    int	readcount;
    while((readcount = (int) sf_read_float (infile, buffer.data(), buffer.size())))
        waveform.insert(waveform.end(),buffer.begin(),buffer.begin()+readcount);
    sf_close (infile) ;

    std::vector<float> in;
    in.resize(sfinfo.frames);
    float acc;
    for(int j = 0; j < sfinfo.frames; ++j) {
        acc = 0.0f;
        for(int i = 0; i < sfinfo.channels; ++i)
            acc += waveform[j*sfinfo.channels + i];
        in[j] = acc / sfinfo.channels;
    }
    std::vector<float> out;
    double orate = target_sampling_rate, irate = sfinfo.samplerate;
    out.resize((size_t)(in.size() * orate / irate + .5));
    size_t odone;

    soxr_error_t error = soxr_oneshot(irate, orate, 1,
    in.data(), in.size(), NULL,
    out.data(), out.size(), &odone,
    NULL, NULL, NULL);

    if(!error) {
        if(ok)
            *ok = true;
        out.resize(odone);
        return torch::from_blob(out.data(),{static_cast<int64_t>(out.size())},torch::TensorOptions().dtype(torch::kFloat)).clone().unsqueeze(0);
    }
    return torch::Tensor();
}

// READ FROM MEMORY BUFFER FACILITIES START

struct CustomMemoryBuffer {
    CustomMemoryBuffer(const uint8_t * _data, sf_count_t _length) : data(_data), length(_length), pos(0) {}
    const uint8_t *data;
    sf_count_t length;
    sf_count_t pos;
};

static sf_count_t buffer_get_filelen (void *user_data)
{
    CustomMemoryBuffer *buff = (CustomMemoryBuffer *) user_data ;
    return buff->length;
}

static sf_count_t buffer_seek (sf_count_t offset, int whence, void *user_data)
{
    CustomMemoryBuffer *buff = (CustomMemoryBuffer *) user_data;
    switch (whence) {
        case SEEK_SET:
            buff->pos = offset;
            break;
        case SEEK_CUR:
            buff->pos += offset;
            break ;
        case SEEK_END:
            buff->pos = buff->length+offset;
            break ;
        default:
            break ;
    }
    return buff->pos;
}

static sf_count_t buffer_read (void *ptr, sf_count_t count, void *user_data)
{
    CustomMemoryBuffer *buff = (CustomMemoryBuffer *) user_data;
    if((buff->pos + count) > buff->length)
        count = buff->length - buff->pos;
    memcpy(ptr,(void *)(buff->data + buff->pos), count);
    buff->pos += count;
    return count;
}

static sf_count_t buffer_tell (void *user_data)
{
    CustomMemoryBuffer *buff = (CustomMemoryBuffer *) user_data;
    return buff->pos;
}

at::Tensor read_audio_sndfile(const uint8_t *content, uint64_t content_size, int target_sampling_rate, SF_INFO &sfinfo, bool *ok)
{
    if(ok)
        *ok = false;

    SF_VIRTUAL_IO sfvirtualio ;
    sfvirtualio.get_filelen = buffer_get_filelen;
    sfvirtualio.seek = buffer_seek;
    sfvirtualio.read = buffer_read;
    sfvirtualio.tell = buffer_tell;

    CustomMemoryBuffer mem_buff(content, content_size);

    memset (&sfinfo, 0, sizeof (sfinfo)) ;
    SNDFILE	*infile;
    if(! (infile = sf_open_virtual( &sfvirtualio, SFM_READ, &sfinfo, (void*)(&mem_buff))))
        return torch::Tensor();

    std::vector<float> waveform, buffer;
    waveform.reserve(sfinfo.channels*sfinfo.frames);
    buffer.resize(4096);
    int	readcount;
    while((readcount = (int) sf_read_float (infile, buffer.data(), buffer.size())))
        waveform.insert(waveform.end(),buffer.begin(),buffer.begin()+readcount);
    sf_close (infile) ;

    std::vector<float> in;
    in.resize(sfinfo.frames);
    float acc;
    for(int j = 0; j < sfinfo.frames; ++j) {
        acc = 0.0f;
        for(int i = 0; i < sfinfo.channels; ++i)
            acc += waveform[j*sfinfo.channels + i];
        in[j] = acc / sfinfo.channels;
    }
    std::vector<float> out;
    double orate = target_sampling_rate, irate = sfinfo.samplerate;
    out.resize((size_t)(in.size() * orate / irate + .5));
    size_t odone;

    soxr_error_t error = soxr_oneshot(irate, orate, 1,
    in.data(), in.size(), NULL,
    out.data(), out.size(), &odone,
    NULL, NULL, NULL);

    if(!error) {
        if(ok)
            *ok = true;
        out.resize(odone);
        return torch::from_blob(out.data(),{static_cast<int64_t>(out.size())},torch::TensorOptions().dtype(torch::kFloat)).clone().unsqueeze(0);
    }
    return torch::Tensor();
}

// READ FROM MEMORY BUFFER FACILITIES END

