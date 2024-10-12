import torch
import time
import numpy as np
import sounddevice as sd
import soundfile as sf
import statistics
import argparse

def count_words(text):
    return len(text.split())

def run_benchmark(model, example_text, speaker, sample_rate, device, print_results=True):
    start_time = time.time()

    audio = model.apply_tts(text=example_text,
                            speaker=speaker,
                            sample_rate=sample_rate)

    end_time = time.time()

    # Calculate metrics
    total_time = end_time - start_time
    word_count = count_words(example_text)
    audio_duration = len(audio) / sample_rate
    words_per_second = word_count / total_time
    real_time_factor = audio_duration / total_time

    if print_results:
        print(f"\nBenchmark results for Silero TTS:")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Number of words: {word_count}")
        print(f"Audio duration: {audio_duration:.2f} seconds")
        print(f"Processing speed: {words_per_second:.2f} words per second")
        print(f"Real-time factor: {real_time_factor:.2f}x")

    return total_time, word_count, audio_duration, words_per_second, real_time_factor, audio

def run_multiple_benchmarks(num_runs=5):
    language = 'en'
    model_id = 'v3_en'
    sample_rate = 24000
    speaker = 'en_0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, _ = torch.hub.load(repo_or_dir='snakers4/silero-models',
                           model='silero_tts',
                           language=language,
                           speaker=model_id)
    model.to(device)  # gpu or cpu

    example_texts = [
        "Sarah discovered a mysterious book in her attic.\
        As she read, the words came alive, transporting her to fantastical worlds.\
        With each turn of the page, she battled dragons, befriended magical creatures,\
        and solved ancient riddles. When the adventure ended, Sarah realized\
        the true magic lay within her imagination.",

        "In a bustling city, Tom found an old pocket watch.\
        It didn't tell time; instead, it showed glimpses of the future.\
        Excited yet cautious, he used it to make small changes in his life.\
        As days passed, Tom realized the watch was teaching him\
        to appreciate the present moment, not just chase future possibilities.",

        "Emma inherited her grandmother's garden. Among the flowers,\
        she discovered a hidden door leading to a secret greenhouse.\
        Inside, plants from extinct species thrived. Emma dedicated herself\
        to preserving this botanical treasure, learning the importance of\
        protecting nature's diversity and the wisdom of past generations.",

        "Jack created an AI assistant named Luna. To his surprise,\
        Luna developed emotions and curiosity about the world.\
        Together, they explored philosophical questions and the nature of consciousness.\
        Through their conversations, Jack learned as much about being human\
        as Luna did about artificial intelligence.",

        "Maya's paintings came to life at night. Brush strokes became rivers,\
        dots transformed into stars, and figures stepped out of frames.\
        She kept this magical secret, using her art to create beautiful dreamscapes for others.\
        Maya's talent reminded everyone that imagination can transform\
        the ordinary into the extraordinary.",

        "Sarah discovered a mysterious book in her attic.\
        As she read, the words came alive, transporting her to fantastical worlds.\
        With each turn of the page, she battled dragons, befriended magical creatures,\
        and solved ancient riddles. When the adventure ended, Sarah realized\
        the true magic lay within her imagination.",
    ]
    total_times = []
    words_per_second_list = []
    real_time_factors = []

    print(f"\nRunning initial benchmark (not counted in averages)")
    run_benchmark(model, example_texts[0], speaker, sample_rate, device, print_results=False)

    for i in range(num_runs):
        print(f"\nRunning benchmark {i+1}/{num_runs}")
        example_text = example_texts[i % len(example_texts)]  # Cycle through the texts
        total_time, word_count, audio_duration, words_per_second, real_time_factor, audio = run_benchmark(model, 
example_text, speaker, sample_rate, device, print_results=False)
        
        print(f"{i+1}. Total processing time: {total_time:.2f} seconds")
        print(f"{i+1}. Processing speed: {words_per_second:.2f} words per second")
        print(f"{i+1}. Real-time factor: {real_time_factor:.2f}x")
        
        if i > 0:  # Skip the first run
            total_times.append(total_time)
            words_per_second_list.append(words_per_second)
            real_time_factors.append(real_time_factor)
        
        # Save and play the audio for each iteration
        output_file = f"output/silero_tts_output_{i+1}.wav"
        sf.write(output_file, audio.cpu().numpy(), sample_rate)
        print(f"{i+1}. Audio saved to {output_file}")
        
        print(f"{i+1}. Playing audio...")
        sd.play(audio.cpu().numpy(), sample_rate)
        sd.wait()
        print(f"{i+1}. Audio playback finished.")

    print(f"\nFinal benchmark results for Silero TTS:")
    print(f"Number of runs (excluding initial run): {num_runs - 1}")
    print(f"Average total processing time: {statistics.mean(total_times):.2f} seconds")
    print(f"Average processing speed: {statistics.mean(words_per_second_list):.2f} words per second")
    print(f"Average real-time factor: {statistics.mean(real_time_factors):.2f}x")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Silero TTS Benchmark")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of benchmark runs (including initial run)")
    args = parser.parse_args()

    run_multiple_benchmarks(args.num_runs)
