import subprocess
from argparse import ArgumentParser

"""WARNING: this scripts may only works on linux"""

# change version based on your situation
pip_torch = "torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116"

pip_dependencies = [
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
    parser.add_argument(
        "--conda_home",
        type=str,
        default="/root/miniconda3/bin",
        help="path to where your conda executable installed"
    )
    parser.add_argument("--conda_name", type=str, default="moss")
    parser.add_argument(
        "--init_conda",
        action="store_true",
        help="whether to create a new conda environment whose name is 'conda_name', make sure it's not exists."
    )
    parser.add_argument(
        "--python_version",
        type=str,
        default="3.10",
        help="python version used when creating conda env"
    )
    parser.add_argument("--reinstall_torch", action="store_true", help="whether to reinstall pytorch or not.")
    parser.add_argument(
        "--no_cuda_ext_for_auto_gptq",
        action="store_true",
        help="whether to not install CUDA extension for auto-gptq"
    )
    parser.add_argument("--install_triton", action="store_true", help="whether to install triton")
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

    if args.reinstall_torch:
        print(
            subprocess.run(
                f"./conda run -n {args.conda_name} pip uninstall torch -y".split(),
                check=True,
                stdout=subprocess.PIPE,
                cwd=args.conda_home
            ).stdout.decode()
        )
        print(
            subprocess.run(
                f"./conda run -n {args.conda_name} pip install {pip_torch}".split(),
                check=True,
                stdout=subprocess.PIPE,
                cwd=args.conda_home
            ).stdout.decode()
        )

    for pip_dependency in pip_dependencies:
        command = f"./conda run -n {args.conda_name} pip install {pip_dependency}"
        if "auto-gptq" in pip_dependency and args.no_cuda_ext_for_auto_gptq:
            command = f"./conda run -n {args.conda_name} BUILD_CUDA_EXT=0 pip install {pip_dependency}"
        if "auto-gptq" in pip_dependency and args.install_triton:
            command = command.replace("auto-gptq", "auto-gptq[triton]")
        print(
            subprocess.run(
                command.split(),
                check=True,
                stdout=subprocess.PIPE,
                cwd=args.conda_home
            ).stdout.decode()
        )


if __name__ == "__main__":
    setup_env()
