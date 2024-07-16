
import csv
import re

# Function to parse the text file and extract relevant data
def parse_benchmarks(file_path):
    with open(file_path, 'r') as file:
        data = file.read().strip().split('\n\n')
    
    parsed_data = []
    for block in data:
        lines = block.strip().split('\n')
        fmap_info = lines[0]
        total_energy_cost = float(re.search(r"Total energy cost: ([\d\.]+)", lines[1]).group(1))
        volatile_energy = float(re.search(r"In volatile : ([\d\.]+)", lines[1]).group(1))
        non_volatile_energy = float(re.search(r"In non-volatile : ([\d\.]+)", lines[1]).group(1))
        total_memory_accesses = int(re.search(r"Total memory accesses: (\d+)", lines[2]).group(1))
        volatile_accesses = int(re.search(r"In volatile : (\d+)", lines[2]).group(1))
        non_volatile_accesses = int(re.search(r"In non-volatile : (\d+)", lines[2]).group(1))
        
        parsed_data.append({
            "fmap_info": fmap_info,
            "total_energy_cost": total_energy_cost,
            "volatile_energy": volatile_energy,
            "non_volatile_energy": non_volatile_energy,
            "total_memory_accesses": total_memory_accesses,
            "volatile_accesses": volatile_accesses,
            "non_volatile_accesses": non_volatile_accesses
        })
    
    return parsed_data

# Function to compute differences and save to a CSV file
def compute_differences_and_save(parsed_data, output_file):
    differences = []

    for i in range(1, len(parsed_data)):
        prev = parsed_data[i - 1]
        curr = parsed_data[i]

        diff = {
            "fmap_info": curr["fmap_info"],
            "total_energy_cost_diff": curr["total_energy_cost"] - prev["total_energy_cost"],
            "volatile_energy_diff": curr["volatile_energy"] - prev["volatile_energy"],
            "non_volatile_energy_diff": curr["non_volatile_energy"] - prev["non_volatile_energy"],
            "total_memory_accesses_diff": curr["total_memory_accesses"] - prev["total_memory_accesses"],
            "volatile_accesses_diff": curr["volatile_accesses"] - prev["volatile_accesses"],
            "non_volatile_accesses_diff": curr["non_volatile_accesses"] - prev["non_volatile_accesses"]
        }

        differences.append(diff)

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = [
            "fmap_info",
            "total_energy_cost_diff",
            "volatile_energy_diff",
            "non_volatile_energy_diff",
            "total_memory_accesses_diff",
            "volatile_accesses_diff",
            "non_volatile_accesses_diff"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for diff in differences:
            writer.writerow(diff)

# Main function to execute the script
def main(file_path : str,output_file_path : str):
    input_file_path = file_path
    
    parsed_data = parse_benchmarks(input_file_path)
    compute_differences_and_save(parsed_data, output_file_path)
    print(f"Differences computed and saved to {output_file_path}")

if __name__ == "__main__":
    main()
