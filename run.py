import argparse
import numpy as np
from modules.experiment.debate_factory import debate_factory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rounds', type=int, default=9,
                          help='number of rounds')
    parser.add_argument('--n_exp', type=int, default=3,
                          help='number of independent experiments')
    parser.add_argument('--mode', type=str, default="scalar",
                          help='测试模式')

    parser.add_argument('--agents', type=int, default=2,
                          help='number of agents')
    parser.add_argument('--n_stubborn', type=int, default=0,
                          help='number of stubborn agents')
    parser.add_argument('--n_suggestible', type=int, default=0,
                          help='number of suggestible agents')

    parser.add_argument('--out_file', type=str, default='',
                          help='path to save the output')
    parser.add_argument('--summarize_mode', type=str, default="last_round",
                          help='all_rounds or last_round: summarize all rounds memories or last round memories')

    parser.add_argument('--not_full_connected', action="store_true",
                          help='True if each agent knows all the position of other agents')
    parser.add_argument('--random_agent', action="store_true",
                          help='True if use random agent selection')
    # parse and set arguments
    args = parser.parse_args()
    # define connectivity matrix
    N = args.agents

    # 默认的是除了对角线区域部分全部是 1
    m = np.ones((N, N), dtype=bool)
    if(args.not_full_connected):
        m = np.random.randint(0, 2, size=(N, N), dtype=np.bool)
        # m = np.array(
        #   # [
        #   #   [False, False, False],
        #   #   [True, False, False],
        #   #   [True, False, False],
        #   # ]
        #   # 默认的是3*3
        #   [
        #     [False, True , True ],
        #     [True , False, False],
        #     [True , False, False],
        #   ]
        #)
    np.fill_diagonal(m, False)
    print(args.mode)
    print(type(args.mode))
    # mode = ["2d", "scalar"]
    exp = debate_factory(args.mode, args, connectivity_matrix=m)
    exp.run()

if __name__ == "__main__":
    main()