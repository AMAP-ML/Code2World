
python -u run.py --agent_name='m3a_gemini25f_Code2World'  --console_port=5556 --grpc_port=8554  --checkpoint_dir='ckpt/m3a_gemini25f_Code2World' 

CUDA_VISIBLE_DEVICES=0,1,4 nohup stdbuf -oL -eL python -u run.py --agent_name='m3a_gemini25f_Code2World'  --console_port=5556 --grpc_port=8554  --checkpoint_dir='ckpt/m3a_gemini25f_Code2World'  2>&1 | tee log/m3a_gemini25f_Code2World.txt &
https://tao.plus7.plus/v1/chat/completions


nohup stdbuf -oL -eL python -u run.py --agent_name='m3a_gemini_2.5f'  --console_port=5556 --grpc_port=8554  --checkpoint_dir='/home/zyh/zyh_documents/zla/ViMo/ViMo-Empowered_Agent/ckpt/gemini_2.5f_20251216' 2>&1 | tee log/gemini_2.5f_20251216.txt &
nohup stdbuf -oL -eL python -u run.py --agent_name='m3a_gpt4o'  --console_port=5558 --grpc_port=8564  --checkpoint_dir='/home/zyh/zyh_documents/zla/ViMo/ViMo-Empowered_Agent/ckpt/m3a_gpt4o' 2>&1 | tee log/m3a_gpt4o.txt &
nohup stdbuf -oL -eL python -u run.py --agent_name='m3a_qw3'  --console_port=5556 --grpc_port=8554  --checkpoint_dir='/home/zyh/zyh_documents/zla/ViMo/ViMo-Empowered_Agent/ckpt/m3a_qw3_new' 2>&1 | tee log/m3a_qw3.txt &
nohup stdbuf -oL -eL python -u run.py --agent_name='m3a_glm46vf'  --console_port=5556 --grpc_port=8554  --checkpoint_dir='/home/zyh/zyh_documents/zla/ViMo/ViMo-Empowered_Agent/ckpt/m3a_glm46vf' 2>&1 | tee log/m3a_glm46vf.txt &

CUDA_VISIBLE_DEVICES=2,3,4 nohup stdbuf -oL -eL python -u run.py --agent_name='m3a_gemini25f_wm_qw3sft600'  --console_port=5556 --grpc_port=8554  --checkpoint_dir='/home/zyh/zyh_documents/zla/ViMo/ViMo-Empowered_Agent/ckpt/m3a_gemini25f_wm_qw3sft600' 2>&1 | tee log/m3a_gemini25f_wm_qw3sft600.txt &
CUDA_VISIBLE_DEVICES=2,3,4 nohup stdbuf -oL -eL python -u run.py --agent_name='m3a_gpt4o_wm_qw3sft600'  --console_port=5556 --grpc_port=8554  --checkpoint_dir='/home/zyh/zyh_documents/zla/ViMo/ViMo-Empowered_Agent/ckpt/m3a_gpt4o_wm_qw3sft600' 2>&1 | tee log/m3a_gpt4o_wm_qw3sft600.txt &
CUDA_VISIBLE_DEVICES=2,3,4 nohup stdbuf -oL -eL python -u run.py --agent_name='m3a_qw3_wm_qw3sft600'  --console_port=5556 --grpc_port=8554  --checkpoint_dir='/home/zyh/zyh_documents/zla/ViMo/ViMo-Empowered_Agent/ckpt/m3a_qw3_wm_qw3sft600' 2>&1 | tee log/m3a_qw3_wm_qw3sft600.txt &
CUDA_VISIBLE_DEVICES=2,3,4 nohup stdbuf -oL -eL python -u run.py --agent_name='m3a_glm46vf_wm_qw3sft600'  --console_port=5556 --grpc_port=8554  --checkpoint_dir='/home/zyh/zyh_documents/zla/ViMo/ViMo-Empowered_Agent/ckpt/m3a_glm46vf_wm_qw3sft600' 2>&1 | tee log/m3a_glm46vf_wm_qw3sft600.txt &

