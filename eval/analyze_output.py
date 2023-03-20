from argparse import ArgumentParser

def set_arguments(parser):
    
    #### new args 
    parser.add_argument("--test_file", type=str, default="test.txt")
    parser.add_argument("--output_file", type=str, default="output")

    args = parser.parse_args()

    return args

def main() :
    parser = ArgumentParser()
    args  = set_arguments(parser)
    
    scores = [0, 0, 0, 0]
    count = 0
    with open(args.test_file, 'r') as fp :
        for line in fp.read().splitlines() :
            idx = line.split()[0]
            scores[int(idx) + 1] += 1
            count += 1
    

    _len = float(count)
    result = []
    for i in range(len(scores)) :
        tmp = (scores[i] / _len) * 100
        # print(scores[i], _len)
        format(tmp, '.2f')
        result.append(tmp)
    a = "=" * 10 + "Result for " + args.test_file.split('/')[-1] + "=" * 10
    print(a)
    print("{:.2f}% negative\n{:.2f}% netural\n{:.2f}% positve\n{:.2f}% others".format(result[0], result[1], result[2], result[3]))
    print("=" * len(a))

if __name__ == '__main__' :
    main()


