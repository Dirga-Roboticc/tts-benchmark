import torch
from parler_tts import ParlerTTSForConditionalGeneration, ParlerTTSStreamer
from transformers import AutoTokenizer
import soundfile as sf
import sounddevice as sd
import time
from threading import Thread
import argparse
import statistics
import numpy as np

def count_words(text):
    return len(text.split())

def generate(model, tokenizer, text, description, use_streaming=False, play_steps_in_s=0.5):
    inputs = tokenizer(description, return_tensors="pt").to(model.device)
    prompt = tokenizer(text, return_tensors="pt").to(model.device)
    
    if use_streaming:
        play_steps = int(model.audio_encoder.config.frame_rate * play_steps_in_s)
        streamer = ParlerTTSStreamer(model, device=model.device, play_steps=play_steps)
        generation_kwargs = dict(
            input_ids=inputs.input_ids,
            prompt_input_ids=prompt.input_ids,
            attention_mask=inputs.attention_mask,
            streamer=streamer,
            do_sample=True,
            temperature=1.0,
            min_new_tokens=10,
        )
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        stream = sd.OutputStream(samplerate=model.config.sampling_rate, channels=1)
        stream.start()
        
        for new_audio in streamer:
            if new_audio.shape[0] == 0:
                break
            audio_chunk = new_audio if isinstance(new_audio, torch.Tensor) else torch.from_numpy(new_audio)
            audio_chunk_np = audio_chunk.cpu().numpy().astype(np.float32)
            stream.write(audio_chunk_np)
            yield audio_chunk
        
        stream.stop()
        stream.close()
    else:
        generation = model.generate(
            input_ids=inputs.input_ids,
            prompt_input_ids=prompt.input_ids,
            attention_mask=inputs.attention_mask,
            do_sample=True,
            temperature=1.0,
            min_new_tokens=10,
        )
        yield generation

def run_benchmark(use_streaming=False, print_results=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
    tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

    prompt = "In a quiet village, Maya found an ancient key buried beneath her grandmother's garden. Every night after that, she dreamt of a glowing door deep in the forest. One evening, she followed her dreams and found the door. Trembling, she unlocked it, stepping into a realm where time paused, and forgotten memories came alive."
    description = "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."

    start_time = time.time()

    audio_chunks = []
    for audio_chunk in generate(model, tokenizer, prompt, description, use_streaming=use_streaming):
        audio_chunks.append(audio_chunk)
    
    if use_streaming:
        audio_arr = torch.cat(audio_chunks, dim=-1).cpu().numpy()
    else:
        audio_arr = audio_chunks[0].cpu().numpy().squeeze()

    end_time = time.time()

    sf.write("output/parler_tts_out.wav", audio_arr, model.config.sampling_rate)

    total_time = end_time - start_time
    word_count = count_words(prompt)
    audio_duration = len(audio_arr) / model.config.sampling_rate
    words_per_second = word_count / total_time
    real_time_factor = audio_duration / total_time

    if print_results:
        print(f"Benchmark results for Parler TTS ({'Streaming' if use_streaming else 'Non-streaming'}):")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Number of words: {word_count}")
        print(f"Audio duration: {audio_duration:.2f} seconds")
        print(f"Processing speed: {words_per_second:.2f} words per second")
        print(f"Real-time factor: {real_time_factor:.2f}x")
        print(f"Audio saved to output/parler_tts_out.wav")

        if not use_streaming:
            print("Playing audio...")
            sd.play(audio_arr, model.config.sampling_rate)
            sd.wait()
            print("Audio playback finished.")

    return total_time, word_count, audio_duration, words_per_second, real_time_factor

def run_multiple_benchmarks(num_runs=5, use_streaming=False):
    total_times = []
    words_per_second_list = []
    real_time_factors = []

    print(f"\nRunning initial benchmark (not counted in averages)")
    run_benchmark(use_streaming=use_streaming, print_results=False)

    for i in range(num_runs):
        print(f"\nRunning benchmark {i+1}/{num_runs}")
        total_time, word_count, audio_duration, words_per_second, real_time_factor = run_benchmark(use_streaming=use_streaming, print_results=False)
        
        if i > 0:  # Skip the first run
            total_times.append(total_time)
            words_per_second_list.append(words_per_second)
            real_time_factors.append(real_time_factor)

    print(f"\nBenchmark results for Parler TTS ({'Streaming' if use_streaming else 'Non-streaming'}):")
    print(f"Number of runs (excluding initial run): {num_runs - 1}")
    print(f"Average total processing time: {statistics.mean(total_times):.2f} seconds")
    print(f"Average processing speed: {statistics.mean(words_per_second_list):.2f} words per second")
    print(f"Average real-time factor: {statistics.mean(real_time_factors):.2f}x")
    print(f"Audio saved to output/parler_tts_out.wav")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parler TTS Benchmark")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of benchmark runs (including initial run)")
    args = parser.parse_args()

    run_multiple_benchmarks(args.num_runs, use_streaming=args.streaming)
