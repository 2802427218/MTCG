import sys
import subprocess
import argparse


def main(file_loc, output_loc):
    # 打印输入和输出文件位置，以便于调试
    print(f"Input file location: {file_loc}, Output log location: {output_loc}")

    # 以追加模式打开输出文件
    with open(output_loc, 'a') as output_file:
        # 双重循环，生成特定的字符串参数
        for i in range(2):  # 循环变量i从0到1
            for j in range(4):  # 循环变量j从0到3
                specify_str = f"[{i},{j},1]"
                output_file.write(specify_str + '\n')
                # 执行eval.py脚本，传递必要的参数，并将标准输出重定向到文件
                subprocess.run(['python', 'eval_implement.py', '--file_loc', file_loc, '--specify', specify_str],
                               stdout=output_file)

        # 在文件中写入“average”，然后再次运行eval.py脚本但不带specify参数
        output_file.write('average\n')
        subprocess.run(['python', 'eval_implement.py', '--file_loc', file_loc],
                       stdout=output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to automate eval.py execution')
    parser.add_argument('--file_loc', type=str, help='Location of the input file',
                        default='../results/predict_4500.txt')
    parser.add_argument('--output_loc', type=str, help='Location of the output log file',
                        default='../results/eval_4500.txt')

    args = parser.parse_args()

    # 直接使用通过argparse解析的参数调用main函数
    main(args.file_loc, args.output_loc)

# python eval.py ../results/predict_final.txt results/eval_final.txt
