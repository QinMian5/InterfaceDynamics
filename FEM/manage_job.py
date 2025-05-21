# Author: Mian Qin
# Date Created: 2025/5/20
from pathlib import Path
import shutil
from itertools import count
import subprocess
import shlex
import json
import argparse


_templates = {
    "job.sh": "job_${SYSTEM}_${THETA}_${JOB_NAME}.sh",
}

_path_data_dir = Path("./jobs")
_path_template_dir = Path("./template")


def _load_params() -> list[dict[str, str]]:
    with open("job_params.json", 'r') as file:
        job_params = json.load(file)
    return job_params


def _replace_params(content: str, params: dict) -> str:
    for k, v in params.items():
        content = content.replace(f"${{{k}}}", str(v))
    return content


def create() -> None:
    _path_data_dir.mkdir(exist_ok=True)

    filename2content = {}
    for filename in _templates:
        path_template = _path_template_dir / filename
        content = path_template.read_text()
        filename2content[filename] = content

    job_params = _load_params()
    for job_param in job_params:
        system = job_param["SYSTEM"]
        theta = job_param["THETA"]
        job_name = job_param["JOB_NAME"]
        path_dir = _path_data_dir
        path_dir.mkdir(parents=True, exist_ok=True)

        for src_filename, dst_filename in _templates.items():
            dst_path = path_dir / _replace_params(dst_filename, job_param)
            dst_path.parent.mkdir(exist_ok=True, parents=True)
            template_content = filename2content[src_filename]
            content = _replace_params(template_content, job_param)
            dst_path.write_text(content)


def submit() -> None:
    job_params = _load_params()
    for job_param in job_params:
        system = job_param["SYSTEM"]
        theta = job_param["THETA"]
        job_name = job_param["JOB_NAME"]
        path_dir = _path_data_dir
        working_dir = path_dir
        job_file_name = _replace_params("job_${SYSTEM}_${THETA}_${JOB_NAME}.sh", job_param)
        command = f"sbatch {str(path_dir / job_file_name)}"
        subprocess.run(shlex.split(command), cwd=working_dir)


def main():
    parser = argparse.ArgumentParser(description="Control the program functions.")
    parser.add_argument("--create", action="store_true", help="Execute the create function")
    parser.add_argument("--submit", action="store_true", help="Execute the submit function")
    # parser.add_argument("--gather", action="store_true", help="Execute the gather function")
    # parser.add_argument("--clean", action="store_true", help="Execute the clean function")

    args = parser.parse_args()
    if args.create:
        print("Creating job folders...")
        create()
        print("Done.")
    if args.submit:
        print("Submitting job.sh...")
        submit()
        print("Done.")
    # if args.gather:
    #     print("Gathering results...")
    #     gather()
    #     print("Done.")
    # if args.clean:
    #     print("Cleaning backup files...")
    #     clean()
    #     print("Done.")


if __name__ == "__main__":
    main()
