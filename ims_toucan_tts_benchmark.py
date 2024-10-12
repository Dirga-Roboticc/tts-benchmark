import torch
import time
import os
import sys

# Add the IMS-Toucan directory to the Python path
sys.path.append('IMS-Toucan')

from InferenceInterfaces.ToucanTTSInterface import ToucanTTSInterface

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

def count_words(text):
    return len(text.split())

def run_benchmark(tts, text, print_results=True):
    start_time = time.time()
    tts.read_to_file(text_list=[text], file_location="output/ims-toucan-tts-output.wav")
    end_time = time.time()
    
    duration = end_time - start_time
    word_count = count_words(text)
    words_per_second = word_count / duration
    
    if print_results:
        print(f"Time taken: {duration:.2f} seconds")
        print(f"Words generated: {word_count}")
        print(f"Words per second: {words_per_second:.2f}")
    
    return duration, word_count, words_per_second

def run_multiple_benchmarks(num_runs=5):
    # Init IMS-Toucan TTS
    tts = ToucanTTSInterface(device=device)
    
    text = "In a quiet village, Maya found an ancient key buried beneath her grandmother's garden. Every night after that, she dreamt of a glowing door deep in the forest. One evening, she followed her dreams and found the door. Trembling, she unlocked it, stepping into a realm where time paused, and forgotten memories came alive."
    
    results = []
    
    print("Running benchmarks...")
    for i in range(num_runs):
        print(f"\nRun {i+1}:")
        duration, word_count, words_per_second = run_benchmark(tts, text, print_results=True)
        if i > 0:  # Skip the first run
            results.append((duration, word_count, words_per_second))
    
    if results:
        avg_duration = sum(r[0] for r in results) / len(results)
        avg_words_per_second = sum(r[2] for r in results) / len(results)
        
        print("\nAverage results (excluding first run):")
        print(f"Average time taken: {avg_duration:.2f} seconds")
        print(f"Average words per second: {avg_words_per_second:.2f}")
    else:
        print("\nNo valid results to average.")

if __name__ == "__main__":
    run_multiple_benchmarks(6)  # Run 6 times, but only count the last 5
