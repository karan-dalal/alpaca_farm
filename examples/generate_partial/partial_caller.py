import subprocess
import shutil
import time
import os

def main():
    """
    Define variables.
    """
    t = 512
    chunk_size = 5
    max_instances = 100
    counter = 0
    generate_16_and_rank = "/home/yusun/code/karan/alpaca_farm/examples/generate_partial/generate_16_rank.py"
    refit_model = "/home/yusun/code/karan/alpaca_farm/examples/generate_partial/refit_model.sh"
    dump_directory = "/home/yusun/code/karan/alpaca_farm/examples/generate_partial/results"

    while t > 0:
        """
        Generate 16 responses and rank them. Results are saved to dump_directory.
        """
        generate_16_and_rank_args = ['--current_t', f'{t}', '--max_instances', f'{max_instances}', '--chunk_size', f'{chunk_size}', '--dump_directory', f'{dump_directory}']    
        completed_process = subprocess.run(["python3", generate_16_and_rank, *generate_16_and_rank_args], stdout=subprocess.PIPE, text=True)
        output = completed_process.stdout
        t = int(output)
        print("-----------")
        print("Current t: ", t)
        print("Timestep: ", counter)
        print("-----------")
        """
        Refit model to generated data. 
        """
        refit_model_args = [f'{t}', f'{dump_directory}', f'{counter}']
        subprocess.run(["bash", refit_model, *refit_model_args])
        """
        Continue iteration. Move trainer_state.json file under t folder result.
        """
        time.sleep(90)
        original_path = os.path.join(dump_directory, "trainer_state.json")
        if os.path.exists(original_path):
            new_path = os.path.join(dump_directory, f"t={t}", "trainer_state.json")
            shutil.move(original_path,new_path)
        
        t -= chunk_size
        counter += 1

if __name__ == "__main__":
    main()