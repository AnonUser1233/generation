import subprocess
import time

if __name__ == "__main__":
    # Set these variables before running this file to the GPUs and cpu ranges accessible for your device.
    # Example ranges and parameters have been given
    GPUS = [0, 1, 2, 3]
    taskset_range = ["0-4", "5-9", "10-14", "15-19"]
    python = "/path/to/python/"
    cores = 2
    # End of variables to set

    jobs = []
    for model in ["davinci", "gpt-3.5-turbo", "curie", "babbage", "ada"]:
        for dataset in ["AGNews", "eli5", "SST", "goemotions"]:
            jobs.append(f"{python} scripts/meta.py {model} --title {dataset} --parent_dir2 {dataset} --cores {cores}")

    
    processes = [None for _ in range(len(GPUS))]
    current_job = 0
    while True:
        for i, process in enumerate(processes):
            if process is None or process.poll() is not None:
                if current_job < len(jobs):
                    print(f"Starting job {current_job}")
                    processes[i] = subprocess.Popen(["taskset", "-c", taskset_range[i]] + jobs[current_job].split() 
                                                    , env={"CUDA_VISIBLE_DEVICES": str(GPUS[i])}, )
                    current_job += 1
        
        time.sleep(10)