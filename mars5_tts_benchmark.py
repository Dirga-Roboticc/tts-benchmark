import torch
import librosa
import time
import statistics
import numpy as np
import soundfile as sf
import soundfile as sf
import soundfile as sf

mars5, config_class = torch.hub.load('Camb-ai/mars5-tts', 'mars5_english', trust_repo=True)

def count_words(text):
    return len(text.split())

def run_benchmark(mars5, config_class, text, wav, ref_transcript, print_results=True):
    cfg = config_class(deep_clone=False, rep_penalty_window=100, top_k=100, temperature=0.7, freq_penalty=3)
    
    start_time = time.time()
    ar_codes, wav_out = mars5.tts(text, wav, ref_transcript, cfg=cfg)
    output_path = f"output/mars5_output_{int(time.time())}.wav"
    sf.write(output_path, wav_out.cpu().numpy(), mars5.sr)
    end_time = time.time()
    
    duration = end_time - start_time
    word_count = count_words(text)
    words_per_second = word_count / duration
    
    if print_results:
        print(f"Time taken: {duration:.2f} seconds")
        print(f"Words generated: {word_count}")
        print(f"Words per second: {words_per_second:.2f}")
        print(f"Audio saved to: {output_path}")
    
    return duration, word_count, words_per_second

def run_multiple_benchmarks(num_runs=5):
    wav, sr = librosa.load('./example.wav', sr=mars5.sr, mono=True)
    wav = torch.from_numpy(wav)
    ref_transcript = "We actually haven't managed to meet demand."
    
    text = "In a quiet village, Maya found an ancient key buried beneath her grandmother's garden. Every night after that, she dreamt of a glowing door deep in the forest. One evening, she followed her dreams and found the door. Trembling, she unlocked it, stepping into a realm where time paused, and forgotten memories came alive."
    results = []
    total_times = []
    words_per_second_list = []
    real_time_factors = []
    
    print("Running benchmarks...")
    for i in range(num_runs):
        print(f"\nRun {i+1}:")
        duration, word_count, words_per_second = run_benchmark(mars5, config_class, text, wav, ref_transcript, print_results=True)
        if i > 0:  # Skip the first run
            results.append((duration, word_count, words_per_second))
            total_times.append(duration)
            words_per_second_list.append(words_per_second)
            real_time_factors.append(duration / (word_count / 150))  # Assuming 150 words per minute for natural speech
    
    print("\nAverage results (excluding first run):")
    print(f"Number of runs (excluding initial run): {num_runs - 1}")
    print(f"Average total processing time: {statistics.mean(total_times):.2f} seconds")
    print(f"Average processing speed: {statistics.mean(words_per_second_list):.2f} words per second")
    print(f"Average real-time factor: {statistics.mean(real_time_factors):.2f}x")
    
    return {
        "model": "Mars5",
        "num_runs": num_runs - 1,
        "avg_processing_time": statistics.mean(total_times),
        "avg_processing_speed": statistics.mean(words_per_second_list),
        "avg_real_time_factor": statistics.mean(real_time_factors)
    }

def print_formatted_output(result):
    print(f"Model: {result['model']}")
    print(f"Number of runs: {result['num_runs']}")
    print(f"Average processing time: {result['avg_processing_time']:.2f} seconds")
    print(f"Average processing speed: {result['avg_processing_speed']:.2f} words/second")
    print(f"Average real-time factor: {result['avg_real_time_factor']:.2f}x")

if __name__ == "__main__":
    result = run_multiple_benchmarks(6)  # Run 6 times, but only count the last 5
    print("\nFormatted Output:")
    print_formatted_output(result)
