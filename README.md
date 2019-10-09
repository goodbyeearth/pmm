# Note
1、统一中文注释吧。

2、把个人测试的东西放到一个文件夹里，务必将该文件夹的名称定义为 ×××_test。（.gitignore 已添加 *_test/ 这一项。）

3、不要改 pommerman 和 stable_baseline 里的东西，要改的话直接在 my_*.py 里进行；
（如果修改涉及内容较多，如 pommerman 里的地图大小、道具等等，新建一个 my_pommerman 文件夹吧。）

# 运行方法（命令行）
python run.py --alg=ppo2 --policy_type=CustomPolicy --num_timesteps=1e5 --log_path=./log_test/

(具体参数暂时先看 my_cmd_utils.py 吧～～)

# 对 pommerman 或 stable_baselines 的相关修改
无