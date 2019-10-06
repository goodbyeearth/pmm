# Note
1、统一中文注释吧。

2、不要改 pommerman 和 stable_baseline 里的东西，要改的话直接在 my_*.py 里进行；
（如果修改涉及内容较多，如 pommerman 里的地图大小、道具等等，新建一个 my_pommerman 文件夹吧。）

# 运行方法（命令行）
python run.py --alg=a2c --num_timesteps=1e5

(具体参数暂时先看 my_cmd_utils.py 吧～～)

# 对 pommerman 或 stable_baselines 的相关修改
1、pommerman/forward_model.py: 修改了 act 函数里 is_communicative 的默认值，False->True。

作用：设置其他智能体做出动作的格式，使动作包含 message