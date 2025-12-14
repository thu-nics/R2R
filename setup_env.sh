conda create -n r2r python=3.10
conda activate r2r
pip install -e .
pip install packaging annotated_types psutil pyyaml
pip install setproctitle
pip install sglang==0.4.6
pip install sgl-kernel==0.1.0
pip install flashinfer-python==0.2.3 -i https://flashinfer.ai/whl/cu124/torch2.6/