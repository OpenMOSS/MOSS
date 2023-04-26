import subprocess
from argparse import ArgumentParser

"""WARNING: this scripts may only works on linux"""


pip_dependencies = [
    # change version based on your situation
    "torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116",
    "transformers==4.25.1",
    "sentencepiece",
    "datasets",
    "accelerate",
    "matplotlib",
    "huggingface_hub",
    "gradio",
    "auto-gptq -i https://pypi.org/simple",
    "mdtex2html"
]


def setup_env():
    parser = ArgumentParser()
    parser.add_argument("--conda_home", type=str, default="/root/miniconda3/bin")
    parser.add_argument("--init_conda", action="store_true")
    parser.add_argument("--conda_name", type=str, default="moss")
    parser.add_argument("--python_version", type=str, default="3.10")
    args = parser.parse_args()

    if args.init_conda:
        print(
            subprocess.run(
                f"./conda create -n {args.conda_name} python={args.python_version} -y".split(),
                check=True,
                stdout=subprocess.PIPE,
                cwd=args.conda_home
            ).stdout.decode()
        )
    for pip_dependency in pip_dependencies:
        print(
            subprocess.run(
                f"./conda run -n {args.conda_name} pip install -U {pip_dependency}".split(),
                check=True,
                stdout=subprocess.PIPE,
                cwd=args.conda_home
            ).stdout.decode()
        )


if __name__ == "__main__":
    setup_env()