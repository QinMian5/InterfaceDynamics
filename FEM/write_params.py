# Author: Mian Qin
# Date Created: 2025/5/20
import json


parameters = {
    "flat": {
        "45": [70, 40],
        "60": [65, 50],
        "70": [55, 50],
        "80": [55, 55],
        "90": [55, 60],
        "120": [45, 70],
        "150": [45, 80],
    },
    "pillar": {
        "45": [70 + 10, 40],
        "60": [65 + 10, 50],
        "70": [55 + 10, 50],
        "80": [55 + 10, 55],
        "90": [55 + 10, 60],
        "120": [45 + 10, 70],
        "150": [45 + 10, 80],
    }
}


def main():
    # system, theta, job_name
    jobs = [
        ["flat", "45", "l1000_to_l3000", ["--lambda_0 1000", "--lambda_star 3000"]],
        ["flat", "60", "l1000_to_l3000", ["--lambda_0 1000", "--lambda_star 3000"]],
        ["flat", "70", "l1000_to_l3000", ["--lambda_0 1000", "--lambda_star 3000"]],
        ["flat", "80", "l1000_to_l3000", ["--lambda_0 1000", "--lambda_star 3000"]],
        ["flat", "90", "l1000_to_l3000", ["--lambda_0 1000", "--lambda_star 3000"]],
        ["flat", "120", "l1000_to_l3000", ["--lambda_0 1000", "--lambda_star 3000"]],
        ["flat", "150", "l1000_to_l3000", ["--lambda_0 1000", "--lambda_star 3000"]],
        ["pillar", "45", "l1000_to_l3000_sym", ["--lambda_0 1000", "--lambda_star 3000"]],
        ["pillar", "60", "l1000_to_l3000_sym", ["--lambda_0 1000", "--lambda_star 3000"]],
        ["pillar", "70", "l1000_to_l3000_sym", ["--lambda_0 1000", "--lambda_star 3000"]],
        ["pillar", "80", "l1000_to_l3000_sym", ["--lambda_0 1000", "--lambda_star 3000"]],
        ["pillar", "90", "l1000_to_l3000_sym", ["--lambda_0 1000", "--lambda_star 3000"]],
        ["pillar", "120", "l1000_to_l3000_sym", ["--lambda_0 1000", "--lambda_star 3000"]],
        ["pillar", "150", "l1000_to_l3000_sym", ["--lambda_0 1000", "--lambda_star 3000"]],
        ["pillar", "45", "l1000_to_l3000_asym", ["--lambda_0 1000", "--lambda_star 3000", "--asymmetric", "--x_offset 40"]],
        ["pillar", "60", "l1000_to_l3000_asym", ["--lambda_0 1000", "--lambda_star 3000", "--asymmetric", "--x_offset 35"]],
        ["pillar", "70", "l1000_to_l3000_asym", ["--lambda_0 1000", "--lambda_star 3000", "--asymmetric", "--x_offset 30"]],
        ["pillar", "80", "l1000_to_l3000_asym", ["--lambda_0 1000", "--lambda_star 3000", "--asymmetric", "--x_offset 30"]],
        ["pillar", "90", "l1000_to_l3000_asym", ["--lambda_0 1000", "--lambda_star 3000", "--asymmetric", "--x_offset 30"]],
        ["pillar", "120", "l1000_to_l3000_asym", ["--lambda_0 1000", "--lambda_star 3000", "--asymmetric", "--x_offset 25"]],
        ["pillar", "150", "l1000_to_l3000_asym", ["--lambda_0 1000", "--lambda_star 3000", "--asymmetric", "--x_offset 25"]],
    ]
    job_params = []
    for job in jobs:
        system, theta, job_name, extra_commands = job
        r_box, z_box = parameters[system][theta]
        job_params.append({
            "SYSTEM": system,
            "THETA": theta,
            "JOB_NAME": job_name,
            "EXTRA_COMMANDS": " ".join(extra_commands),
        })
    with open("job_params.json", 'w') as file:
        json.dump(job_params, file, indent='\t')


if __name__ == "__main__":
    main()
